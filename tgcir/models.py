import copy
import math
import multiprocessing
import os.path
import random

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

import clip
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.checkpoint import checkpoint
from data_utils import targetpad_transform, CIRDataset
from utils import collate_fn


class SpatialAttention(nn.Module):
    def __init__(self, dim=512):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=dim, out_channels=1, kernel_size=1, stride=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        B, L, D = x.shape  # (B,49,1024)  text: (B,L,1024)
        x = x.permute(0, 2, 1)  # (B,1024,49)
        weight_map = self.conv(x)  # (B,1,L)
        out = torch.mean(x * weight_map, dim=-1)  # (b,1,49) * (b,1024,49) => (b,1024,49) =>(b,1024)
        return out, weight_map


class TokenLearner(nn.Module):
    def __init__(self, S, dim):
        super().__init__()
        self.S = S
        self.tokenizers = nn.ModuleList([SpatialAttention(dim) for _ in range(S)])

    def forward(self, x):
        B, L, C = x.shape
        Z = torch.Tensor(B, self.S, C).cuda()  # (B,8,1024)
        for i in range(self.S):
            Ai, _ = self.tokenizers[i](x)
            Z[:, i, :] = Ai
        return Z


class Backbone(nn.Module):
    def __init__(self, img_encoder='ViT-B/16', hidden_dim=512, dropout=0.0, local_token_num=8, global_token_num=4):
        super().__init__()
        self.clip, self.preprocess = clip.load(img_encoder, device='cuda', jit=False)
        self.clip = self.clip.float()
        self.image_backbone = self.clip.visual
        self.img_encoder = img_encoder
        self.tokenlearn = TokenLearner(S=local_token_num, dim=hidden_dim)
        self.hidden_dim = hidden_dim
        self.fc = nn.Linear(768, 512)
        self.text_fc = nn.Linear(512, 512)

        self.masks = torch.nn.Embedding(global_token_num, hidden_dim)
        mask_array = np.zeros([global_token_num, hidden_dim])
        mask_array.fill(0.1)
        mask_len = int(hidden_dim / global_token_num)
        for i in range(global_token_num):
            mask_array[i, i * mask_len:(i + 1) * mask_len] = 1
        self.masks.weight = torch.nn.Parameter(torch.Tensor(mask_array), requires_grad=True)

        self.local_token_num = local_token_num
        self.global_token_num = global_token_num

        self.tokenlearn_text = copy.deepcopy(self.tokenlearn)
        self.masks_text = copy.deepcopy(self.masks)

    def extract_img_fea(self, x):
        x = self.image_backbone.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([self.image_backbone.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1],
                                                                                     dtype=x.dtype,
                                                                                     device=x.device), x],
                      dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.image_backbone.positional_embedding.to(x.dtype)
        x = self.image_backbone.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.image_backbone.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        global_fea = self.image_backbone.ln_post(x[:, 0, :]) @ self.image_backbone.proj

        # mask_norm = None
        global_tokens = []
        for idx in range(self.global_token_num):
            concept_idx = np.zeros((len(x),), dtype=int)
            concept_idx += idx
            concept_idx = torch.from_numpy(concept_idx)
            concept_idx = concept_idx.cuda()
            concept_idx = Variable(concept_idx)
            self.mask = self.masks(concept_idx)
            self.mask = torch.nn.functional.relu(self.mask)
            masked_embedding = global_fea * self.mask  # batch_size, dim
            global_tokens.append(masked_embedding)
        global_tokens = torch.stack(global_tokens).permute(1, 0, 2).contiguous()

        local_fea = self.fc(x.float())
        local_tokens = self.tokenlearn(self.fc(x.float()))
        return torch.cat([global_tokens, local_tokens], dim=1)

    def extract_text_fea(self, txt):
        text = clip.tokenize(txt).cuda()

        x = self.clip.token_embedding(text).type(self.clip.dtype)  # [batch_size, n_ctx, d_model]

        x = x + self.clip.positional_embedding.type(self.clip.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.clip.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.clip.ln_final(x).type(self.clip.dtype)
        global_fea = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.clip.text_projection

        global_tokens = []
        for idx in range(self.global_token_num):
            concept_idx = np.zeros((len(x),), dtype=int)
            concept_idx += idx
            concept_idx = torch.from_numpy(concept_idx)
            concept_idx = concept_idx.cuda()
            concept_idx = Variable(concept_idx)
            self.mask = self.masks_text(concept_idx)
            self.mask = torch.nn.functional.relu(self.mask)
            masked_embedding = global_fea * self.mask  # batch_size, dim
            global_tokens.append(masked_embedding)
        global_tokens = torch.stack(global_tokens).permute(1, 0, 2).contiguous()
        local_tokens = self.tokenlearn_text(self.text_fc(x.float()))

        return torch.cat([global_tokens, local_tokens], dim=1)


class CIRPlus(nn.Module):
    def __init__(self, clip_model_name, tau=0.01,
                 transform="targetpad", target_ratio=1.25,
                 device=torch.device('cuda'), plus=False
                 , local_token_num=8, global_token_num=4):
        super().__init__()

        # initial main model
        self.device = device
        self.tau = tau
        self.backbone = Backbone(clip_model_name, 512, 0, local_token_num, global_token_num)
        self.input_dim = self.backbone.clip.visual.input_resolution
        print("image size:", self.input_dim)
        self.output_dim = self.backbone.clip.visual.output_dim
        self.crossentropy_criterion = nn.CrossEntropyLoss()
        if transform == 'targetpad':
            self.preprocess = targetpad_transform(target_ratio, self.input_dim)
        self.plus = plus
        hidden_dim = 512
        self.preprocess = self.backbone.preprocess
        self.local_weight = nn.Parameter(torch.tensor([1.0 for _ in range(local_token_num + global_token_num)]))

        self.s_remain_map = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        self.t_remain_map = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        self.t_replace_map = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def img_embed(self, image, return_pool_and_normalized=False):
        '''
        If in train: return pooled_and_normalized features;
        if in val: return raw features (for computing txt-img fusion)
            and pooled_and_normalized_features (for all target images)
        '''
        image_embeds = self.backbone.extract_img_fea(image)
        res = [image_embeds]
        if return_pool_and_normalized:
            image_feats = F.normalize(torch.mean(image_embeds, dim=1), p=2, dim=-1)
            res.append(image_feats)
        if len(res) == 1:
            return res[0]
        return res

    def img_txt_fusion(self, ref_token, mod):
        mod_token = self.backbone.extract_text_fea(mod)
        remain_mask = self.s_remain_map(torch.cat([ref_token, mod_token], dim=-1))
        # replace_mask = self.s_replace_map(torch.cat([ref_token, mod_token], dim=-1))
        replace_mask = 1 - remain_mask
        s_fuse_local = remain_mask * ref_token + replace_mask * mod_token
        s_fuse_local = F.normalize(torch.mean(s_fuse_local, dim=1), p=2, dim=-1)
        return s_fuse_local

    def load_ckpt(self, model_path, is_origin=False):
        saved_state_dict = torch.load(model_path, map_location=self.device)
        self.load_state_dict(saved_state_dict['state_dict'], strict=False)
        if is_origin:
            # for stage2
            self.backbone.tokenlearn_text = copy.deepcopy(self.backbone.tokenlearn)
            self.backbone.masks_text = copy.deepcopy(self.backbone.masks)
            for param in self.backbone.image_backbone.parameters():
                param.requires_grad = False
            for param in self.backbone.tokenlearn.parameters():
                param.requires_grad = False
            for param in self.backbone.fc.parameters():
                param.requires_grad = False
            for param in self.backbone.masks.parameters():
                param.requires_grad = False

    def extract_bank_features(self, cirDataset: CIRDataset, device, bank_path, reload_bank=False):
        self.eval().float()
        if not os.path.exists(bank_path) or reload_bank:
            self.refer_bank = torch.zeros(len(cirDataset), 12, 512)
            self.target_bank = torch.zeros(cirDataset.image_id, 512)
            data_loader = DataLoader(dataset=cirDataset, batch_size=32, num_workers=multiprocessing.cpu_count(),
                                     pin_memory=True, collate_fn=collate_fn)
            for reference_image, caption, target_image, index, \
                target_index, reference_index_all, target_index_all in tqdm(
                data_loader, desc='encoding bank features...'):
                reference_image = reference_image.to(device, non_blocking=True)
                target_image = target_image.to(device, non_blocking=True)

                with torch.no_grad():
                    batch_refer_features = self.img_embed(reference_image, return_pool_and_normalized=True)
                    batch_target_features_p = self.img_embed(target_image, return_pool_and_normalized=True)[
                        -1].detach().cpu()
                    batch_refer_features_p = batch_refer_features[1].detach().cpu()
                    batch_refer_features = batch_refer_features[0].detach().cpu()
                    self.refer_bank[index] = batch_refer_features
                    self.target_bank[reference_index_all] = batch_refer_features_p
                    self.target_bank[target_index_all] = batch_target_features_p
            torch.save([self.refer_bank, self.target_bank], bank_path)
        else:
            self.refer_bank, self.target_bank = torch.load(bank_path)
        print("load bank successfully")

    def extract_refer_bank_features(self, cirDataset: CIRDataset, device, bank_path, reload_bank=False):
        self.eval().float()
        if not os.path.exists(bank_path) or reload_bank:
            self.refer_bank = torch.zeros(cirDataset.image_id, 12, 512)
            data_loader = DataLoader(dataset=cirDataset, batch_size=32, num_workers=multiprocessing.cpu_count(),
                                     pin_memory=True, collate_fn=collate_fn)
            for reference_image, caption, target_image, index, \
                target_index, reference_index_all, target_index_all in tqdm(
                data_loader, desc='encoding bank features...'):
                reference_image = reference_image.to(device, non_blocking=True)
                target_image = target_image.to(device, non_blocking=True)
                with torch.no_grad():
                    batch_refer_features = self.img_embed(reference_image).detach().cpu()
                    batch_target_features = self.img_embed(target_image).detach().cpu()
                    self.refer_bank[reference_index_all] = batch_refer_features
                    self.refer_bank[target_index_all] = batch_target_features
            torch.save(self.refer_bank, bank_path)
        print("load reference bank successfully")

    def load_refer_bank(self, bank_path):
        self.refer_bank = torch.load(bank_path)

    def bank_large_step(self, loss, text, indexs, target_indexs,
                        refer_indexs=None):
        if self.plus:
            reference_image_feats = self.refer_bank[refer_indexs].detach().to(self.device)
        else:
            reference_image_feats = self.refer_bank[indexs].detach().to(self.device)
        query_feats = self.img_txt_fusion(reference_image_feats, text)
        target_feats_all = self.target_bank.detach().to(self.device)
        target_indexs = target_indexs.to(self.device)
        target_neg_loss = self.infonce_loss(query_feats, target_feats_all, target_indexs, tau=self.tau)
        loss['bank_loss'] = target_neg_loss

    def forward(self, text, indexs, target_indexs, refer_indexs, refer_image=None, target_image=None):
        loss = dict()
        self.bank_large_step(loss, text, indexs, target_indexs, refer_indexs)
        return loss

    def infonce_loss(self, query_features, target_features, labels=None, tau=0.01):
        logits = (query_features @ target_features.T) / tau
        if labels is None:
            labels = torch.arange(query_features.shape[0]).long().to(self.device)
        return self.crossentropy_criterion(logits, labels)
