'''
Modified for Candidate-Reranking CIR stage I
from https://github.com/salesforce/BLIP/blob/main/models/blip_retrieval.py
'''

from med import BertConfig, BertModel
from transformers import BertTokenizer

import torch
from torch import nn
import torch.nn.functional as F

from blip import create_vit, init_tokenizer, load_checkpoint


class BLIP_Retrieval(nn.Module):
    def __init__(self,
                 med_config='blip4cir/configs/med_config.json',
                 image_size=384,
                 vit='base',
                 vit_grad_ckpt=False,
                 vit_ckpt_layer=0,
                 embed_dim=256
                 ):
        """
        Modified BLIP_Retrieval model for CIR

        Args:
            med_config (str): path for the mixture of encoder-decoder model's configuration file
            image_size (int): input image size
            vit (str): model size of vision transformer
        """
        super().__init__()

        self.visual_encoder, vision_width = create_vit(vit, image_size, vit_grad_ckpt, vit_ckpt_layer)
        self.tokenizer = init_tokenizer()
        med_config = BertConfig.from_json_file(med_config)
        med_config.encoder_width = vision_width
        self.text_encoder = BertModel(config=med_config, add_pooling_layer=False)

        text_width = self.text_encoder.config.hidden_size

        self.vision_proj = nn.Linear(vision_width, embed_dim)  # 768 -> 256
        self.text_proj = nn.Linear(text_width, embed_dim)  # 768 -> 256

        self.temp = nn.Parameter(0.07 * torch.ones([]))

    def init_stage2(self):
        for param in self.visual_encoder.parameters():
            param.requires_grad = False
        for param in self.vision_proj.parameters():
            param.requires_grad = False

    def img_embed(self, image, atts=False, return_pool_and_normalized=False):
        '''
        If in train: return pooled_and_normalized features;
        if in val: return raw features (for computing txt-img fusion)
            and pooled_and_normalized_features (for all target images)
        '''
        image_embeds = self.visual_encoder(image)  # B x 577 x 768
        out = (image_embeds,)
        if return_pool_and_normalized:
            image_embeds_p = F.normalize(self.vision_proj(image_embeds[:, 0, :]), dim=-1)  # B x 256
            out += (image_embeds_p,)
        if atts:
            image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)
            out += (image_atts,)
        if len(out) == 1:
            out = out[0]  # if only one type of feature is returned, unwrap the tuple
        return out

    def img_embed_p(self, image):
        '''
        If in train: return pooled_and_normalized features;
        if in val: return raw features (for computing txt-img fusion)
            and pooled_and_normalized_features (for all target images)
        '''
        image_embeds = self.visual_encoder(image)  # B x 577 x 768
        image_embeds_p = F.normalize(self.vision_proj(image_embeds[:, 0, :]), dim=-1)  # B x 256
        return image_embeds_p

    def img_txt_fusion(self, r_image_embeds, t_image_embeds, text, train=False, return_raw=False):
        device = r_image_embeds.device

        r_image_atts = torch.ones(r_image_embeds.size()[:-1], dtype=torch.long).to(device)

        text = self.tokenizer(text, padding='longest', return_tensors="pt").to(device)
        text.input_ids[:, 0] = self.tokenizer.enc_token_id

        output_pos = self.text_encoder(text.input_ids,
                                       attention_mask=text.attention_mask,
                                       encoder_hidden_states=r_image_embeds,
                                       encoder_attention_mask=r_image_atts,
                                       return_dict=True,
                                       )

        # compute logits
        predicted_features = F.normalize(self.text_proj(output_pos.last_hidden_state[:, 0, :]), dim=-1)  # B x 256
        if not train:
            if return_raw:
                return output_pos
            else:
                return predicted_features
        else:
            target_features = t_image_embeds  # already normalized
            logits = predicted_features @ target_features.T / self.temp  # B x B
            return logits


def blip_cir(pretrained='', **kwargs):
    model = BLIP_Retrieval(**kwargs)
    if pretrained:
        model, msg = load_checkpoint(model, pretrained)
        print("missing keys:")
        print(msg.missing_keys)
    return model
