"""
 Copyright (c) 2023, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""
import copy
import logging
import math

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.cuda.amp import autocast as autocast
from torch.nn import functional as F

from lavis.common.registry import registry
from lavis.models.base_model import all_gather_with_grad, concat_all_gather
from lavis.models.blip2_models.blip2 import (
    Blip2Base,
    compute_sim_matrix,
    disabled_train,
)
from lavis.models.blip_models.blip_outputs import BlipOutput, BlipOutputFeatures


@registry.register_model("blip2_cir_align_prompt")
class Blip2QformerCirAlignPrompt(Blip2Base):
    """
    BLIP2 first-stage model with Q-former and ViT.
    Supported model types:
        - pretrained: pretrained model with vit-g
        - pretrain_vitL: pretrained model with vit-large
        - coco: fintuned model on coco
    Usage:
        >>> from lavis.models import load_model
        >>> model = load_model("blip2", "pretrain")
    """

    PRETRAINED_MODEL_CONFIG_DICT = {
        "pretrain": "configs/models/blip2/blip2_pretrain.yaml",
        "pretrain_vitL": "configs/models/blip2/blip2_pretrain_vitL.yaml",
        "coco": "configs/models/blip2/blip2_coco.yaml",
    }

    def __init__(
            self,
            vit_model="eva_clip_g",
            img_size=224,
            drop_path_rate=0,
            use_grad_checkpoint=False,
            vit_precision="fp16",
            freeze_vit=True,
            num_query_token=32,
            cross_attention_freq=2,
            embed_dim=256,
            max_txt_len=32,
    ):
        super().__init__()

        self.tokenizer = self.init_tokenizer()

        self.visual_encoder, self.ln_vision = self.init_vision_encoder(
            vit_model, img_size, drop_path_rate, use_grad_checkpoint, vit_precision
        )
        if freeze_vit:
            for name, param in self.visual_encoder.named_parameters():
                param.requires_grad = False
            self.visual_encoder = self.visual_encoder.eval()
            self.visual_encoder.train = disabled_train
            logging.info("freeze vision encoder")
        self.Qformer, self.query_tokens = self.init_Qformer(
            num_query_token, self.visual_encoder.num_features, cross_attention_freq
        )
        self.Qformer.resize_token_embeddings(len(self.tokenizer))
        state_dict = self.Qformer.state_dict()
        for name, param in self.Qformer.named_parameters():
            if "_query" in name:
                key_orig = name.replace("_query", "")
                param.data.copy_(state_dict[key_orig])

        self.vision_proj = nn.Linear(self.Qformer.config.hidden_size, embed_dim)
        self.text_proj = nn.Linear(self.Qformer.config.hidden_size, embed_dim)

        self.itm_head = nn.Linear(self.Qformer.config.hidden_size, 2)

        self.temp = nn.Parameter(0.07 * torch.ones([]))

        self.max_txt_len = max_txt_len
        # new tokens
        self.prompt_tokens = nn.Parameter(
            torch.zeros(1, num_query_token, self.Qformer.config.hidden_size)
        )
        self.prompt_tokens.data.normal_(mean=0.0, std=self.Qformer.config.initializer_range)

    def init_stage2(self, tau=0.05):
        self.tau = tau
        self.Qformer_query = copy.deepcopy(self.Qformer)
        self.text_proj_q = copy.deepcopy(self.text_proj)
        for param in self.parameters():
            param.requires_grad = False
        for param in self.Qformer_query.parameters():
            param.requires_grad = True
        for param in self.text_proj_q.parameters():
            param.requires_grad = True
        self.temp = nn.Parameter(tau * torch.ones([], device=self.device))
        self.query_type = 1
        self.crossentropy_criterion = nn.CrossEntropyLoss()

    @torch.no_grad()
    def get_bank_feats(self, image, text, target):
        with torch.no_grad():
            image_embeds = self.ln_vision(self.visual_encoder(image))
            image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
                self.device
            )
            # query tokens
            query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
            query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(
                self.device
            )
            # text tokens
            text_tokens = self.tokenizer(
                text,
                padding="max_length",
                truncation=True,
                max_length=self.max_txt_len,
                return_tensors="pt",
            ).to(self.device)
            # fusion reference image and text tokens into a set of multi-modal tokens
            attention_mask = torch.cat([query_atts, text_tokens.attention_mask], dim=1)
            fusion_output = self.Qformer.bert(
                text_tokens.input_ids,
                query_embeds=query_tokens,
                attention_mask=attention_mask,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )
            text_output = self.Qformer.bert(
                text_tokens.input_ids,
                query_embeds=fusion_output.last_hidden_state[:, : query_tokens.size(1), :],
                attention_mask=attention_mask,
                return_dict=True,
            )
            fusion_feats = self.text_proj(text_output.last_hidden_state[:, 32, :])

            fusion_hidden_feats = fusion_output.last_hidden_state[:, : query_tokens.size(1), :]
            ###============== Fusion-target Contrastive ===================###
            # reference image feature
            target_embeds = self.ln_vision(self.visual_encoder(target))
            target_atts = torch.ones(target_embeds.size()[:-1], dtype=torch.long).to(
                self.device
            )
            target_output = self.Qformer.bert(
                query_embeds=query_tokens,
                encoder_hidden_states=target_embeds,
                encoder_attention_mask=target_atts,
                use_cache=True,
                return_dict=True,
            )
            target_feats = F.normalize(
                self.vision_proj(target_output.last_hidden_state), dim=-1
            )

            refer_output = self.Qformer.bert(
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                use_cache=True,
                return_dict=True,
            )
            refer_feats = F.normalize(
                self.vision_proj(refer_output.last_hidden_state), dim=-1
            )
            return fusion_hidden_feats, target_feats, refer_feats, fusion_feats

    @torch.no_grad()
    def get_refer_bank_feats(self, image, text, target):
        with torch.no_grad():
            image_embeds = self.ln_vision(self.visual_encoder(image))
            image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
                self.device
            )
            # query tokens
            query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
            query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(
                self.device
            )
            # text tokens
            text_tokens = self.tokenizer(
                text,
                padding="max_length",
                truncation=True,
                max_length=self.max_txt_len,
                return_tensors="pt",
            ).to(self.device)
            # fusion reference image and text tokens into a set of multi-modal tokens
            attention_mask = torch.cat([query_atts, text_tokens.attention_mask], dim=1)
            fusion_output = self.Qformer.bert(
                text_tokens.input_ids,
                query_embeds=query_tokens,
                attention_mask=attention_mask,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )
            fusion_hidden_feats = fusion_output.last_hidden_state[:, : query_tokens.size(1), :]
            ###============== Fusion-target Contrastive ===================###
            # reference image feature
            target_embeds = self.ln_vision(self.visual_encoder(target))
            target_atts = torch.ones(target_embeds.size()[:-1], dtype=torch.long).to(
                self.device
            )
            target_fusion_output = self.Qformer.bert(
                text_tokens.input_ids,
                query_embeds=query_tokens,
                attention_mask=attention_mask,
                encoder_hidden_states=target_embeds,
                encoder_attention_mask=target_atts,
                return_dict=True,
            )
            target_fusion_hidden_feats = target_fusion_output.last_hidden_state[:, : query_tokens.size(1), :]
            return fusion_hidden_feats, target_fusion_hidden_feats

    def forward_stage2(self, text, target_feats, fusion_hidden_states, target_indexs):
        bs = target_indexs.shape[0]
        # query tokens
        query_tokens = self.query_tokens.expand(bs, -1, -1)
        query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(
            self.device
        )
        # text tokens
        text_tokens = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_txt_len,
            return_tensors="pt",
        ).to(self.device)
        # fusion reference image and text tokens into a set of multi-modal tokens
        attention_mask = torch.cat([query_atts, text_tokens.attention_mask], dim=1)

        text_output = self.Qformer_query.bert(
            text_tokens.input_ids,
            query_embeds=fusion_hidden_states,
            attention_mask=attention_mask,
            return_dict=True,
        )

        fusion_feats = F.normalize(
            self.text_proj_q(text_output.last_hidden_state[:, 32, :]), dim=-1
        )  # 128*256
        loss_qtc = torch.tensor(0, device=self.device, dtype=torch.float)
        target_indexs = target_indexs.to(self.device)
        for i in range(bs):
            sim_t2q_ = torch.matmul(
                fusion_feats[i:i + 1].unsqueeze(1).unsqueeze(1), target_feats.permute(0, 2, 1)
            ).squeeze()  # 1*128*32
            sim_q2t, _ = sim_t2q_.max(-1)
            ground_truth_i = target_indexs[i]
            sim_q2t = sim_q2t / self.temp
            loss_qtc_i = F.cross_entropy(sim_q2t, ground_truth_i)
            loss_qtc += loss_qtc_i
        loss_qtc /= bs
        return {
            'loss_qtc': loss_qtc,
        }

    @torch.no_grad()
    def inference(self, reference_embeds, target_feats, text):
        image_atts = torch.ones(reference_embeds.size()[:-1], dtype=torch.long).to(
            self.device
        )
        # query tokens
        query_tokens = self.query_tokens.expand(reference_embeds.shape[0], -1, -1)
        query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(
            self.device
        )
        # text tokens
        text_tokens = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_txt_len,
            return_tensors="pt",
        ).to(self.device)

        attention_mask = torch.cat([query_atts, text_tokens.attention_mask], dim=1)
        fusion_output = self.Qformer.bert(
            text_tokens.input_ids,
            query_embeds=query_tokens,
            attention_mask=attention_mask,
            encoder_hidden_states=reference_embeds,
            encoder_attention_mask=image_atts,
            return_dict=True,
        )
        if self.query_type == 0:
            Qformer = self.Qformer
            text_proj = self.text_proj
        else:
            Qformer = self.Qformer_query
            text_proj = self.text_proj_q
        text_output = Qformer.bert(
            text_tokens.input_ids,
            query_embeds=fusion_output.last_hidden_state[:, : query_tokens.size(1), :],
            attention_mask=attention_mask,
            return_dict=True,
        )

        fusion_feats = F.normalize(
            text_proj(text_output.last_hidden_state[:, 32, :]), dim=-1
        )
        sim_t2q = torch.matmul(
            fusion_feats.unsqueeze(1).unsqueeze(1), target_feats.permute(0, 2, 1)
        ).squeeze()

        sim_i2t, _ = sim_t2q.max(-1)
        return sim_i2t

    @torch.no_grad()
    def extract_target_features(self, image, mode='mean'):
        with self.maybe_autocast():
            image_embeds_frozen = self.ln_vision(self.visual_encoder(image))
        image_embeds_frozen = image_embeds_frozen.float()
        image_atts = torch.ones(
            image_embeds_frozen.size()[:-1], dtype=torch.long
        ).to(self.device)
        query_tokens = self.query_tokens.expand(
            image_embeds_frozen.shape[0], -1, -1
        )

        query_output = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds_frozen,
            encoder_attention_mask=image_atts,
            return_dict=True,
        )
        image_embeds = query_output.last_hidden_state

        # return image_embeds
        image_features = F.normalize(self.vision_proj(image_embeds), dim=-1)
        return image_features, image_embeds_frozen

    @classmethod
    def from_config(cls, cfg):
        vit_model = cfg.get("vit_model", "eva_clip_g")
        img_size = cfg.get("image_size")
        num_query_token = cfg.get("num_query_token")
        cross_attention_freq = cfg.get("cross_attention_freq", 2)

        drop_path_rate = cfg.get("drop_path_rate", 0)
        use_grad_checkpoint = cfg.get("use_grad_checkpoint", False)
        vit_precision = cfg.get("vit_precision", "fp16")
        freeze_vit = cfg.get("freeze_vit", True)

        max_txt_len = cfg.get("max_txt_len", 32)

        model = cls(
            vit_model=vit_model,
            img_size=img_size,
            drop_path_rate=drop_path_rate,
            use_grad_checkpoint=use_grad_checkpoint,
            vit_precision=vit_precision,
            freeze_vit=freeze_vit,
            num_query_token=num_query_token,
            cross_attention_freq=cross_attention_freq,
            max_txt_len=max_txt_len,
        )
        model.load_checkpoint_from_config(cfg)

        return model
