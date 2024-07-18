import math
import multiprocessing
import os.path
import random

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


class CIRPlus(nn.Module):
    def __init__(self, clip_model_name, tau=0.01,
                 transform="targetpad", target_ratio=1.25,
                 device=torch.device('cuda'), plus=False, neg_num=-1,
                 wo_bank=True, ):
        super().__init__()

        # initial main model
        self.device = device
        self.clip, self.preprocess = clip.load(clip_model_name, device=device, jit=False)
        self.clip = self.clip.float()
        self.combining_function = self.element_wise_sum
        if not wo_bank:
            for param in self.clip.visual.parameters():
                param.requires_grad = False
        self.tau = tau
        self.input_dim = self.clip.visual.input_resolution
        print("image size:", self.input_dim)
        self.output_dim = self.clip.visual.output_dim
        self.crossentropy_criterion = nn.CrossEntropyLoss()
        if transform == 'targetpad':
            self.preprocess = targetpad_transform(target_ratio, self.input_dim)
        self.plus = plus
        self.neg_num = neg_num
        self.wo_bank = wo_bank

    def encode_image(self, image):
        image_feats = self.clip.encode_image(image)
        return image_feats

    def encode_text(self, text):
        text = clip.tokenize(text).to(self.device)
        text_feats = self.clip.encode_text(text)
        return text_feats

    def element_wise_sum(self, refer_image_feats, text_feats):
        query_features = refer_image_feats + text_feats
        return query_features

    def load_ckpt(self, model_path, is_origin=False):
        saved_state_dict = torch.load(model_path, map_location=self.device)
        if is_origin:
            self.clip.load_state_dict(saved_state_dict[self.clip.__class__.__name__], strict=False)
        else:
            self.load_state_dict(saved_state_dict['state_dict'], strict=False)

    def extract_bank_features(self, cirDataset: CIRDataset, device, bank_path, reload_bank=False):
        if not os.path.exists(bank_path) or reload_bank:
            self.refer_bank = torch.zeros(len(cirDataset), self.output_dim)
            self.target_bank = torch.zeros(cirDataset.image_id, self.output_dim)
            data_loader = DataLoader(dataset=cirDataset, batch_size=32, num_workers=multiprocessing.cpu_count(),
                                     pin_memory=True, collate_fn=collate_fn)
            for reference_image, caption, target_image, index, \
                target_index, reference_index_all, target_index_all in tqdm(
                data_loader, desc='encoding bank features...'):
                reference_image = reference_image.to(device, non_blocking=True)
                target_image = target_image.to(device, non_blocking=True)

                with torch.no_grad():
                    batch_refer_features = self.encode_image(reference_image).detach().cpu()
                    batch_target_features = self.encode_image(target_image).detach().cpu()
                    batch_target_features = F.normalize(batch_target_features)
                    self.refer_bank[index] = batch_refer_features
                    self.target_bank[reference_index_all] = F.normalize(batch_refer_features)
                    self.target_bank[target_index_all] = F.normalize(batch_target_features)
            torch.save([self.refer_bank, self.target_bank], bank_path)
        else:
            self.refer_bank, self.target_bank = torch.load(bank_path)

    def extract_refer_bank_features(self, cirDataset: CIRDataset, device, bank_path, reload_bank=False):
        if not os.path.exists(bank_path) or reload_bank:
            self.refer_bank = torch.zeros(cirDataset.image_id, self.output_dim, device=self.device)
            data_loader = DataLoader(dataset=cirDataset, batch_size=32, num_workers=multiprocessing.cpu_count(),
                                     pin_memory=True, collate_fn=collate_fn)
            for reference_image, caption, target_image, index, \
                target_index, reference_index_all, target_index_all in tqdm(
                data_loader, desc='encoding bank features...'):
                reference_image = reference_image.to(device, non_blocking=True)
                target_image = target_image.to(device, non_blocking=True)
                with torch.no_grad():
                    batch_refer_features = self.encode_image(reference_image)
                    batch_target_features = self.encode_image(target_image)
                    self.refer_bank[reference_index_all] = batch_refer_features
                    self.refer_bank[target_index_all] = batch_target_features
            self.refer_bank = self.refer_bank.to(self.device)
            torch.save(self.refer_bank, bank_path)

    def load_refer_bank(self, bank_path):
        self.refer_bank = torch.load(bank_path)

    def get_neg_id(self, i, N, k):
        range_target = list(range(N))
        range_target.remove(i)
        j = random.sample(range_target, k)
        return j

    def part_infonce_loss(self, query_feats, target_feats_all, target_indexs):
        M = target_feats_all.shape[0]
        N = query_feats.shape[0]
        target_loss = torch.tensor(0, device=self.device, dtype=torch.float)
        for i in range(N):
            pos_id = target_indexs[i].item()
            neg_ids = self.get_neg_id(pos_id, M, self.neg_num)
            neg_feats = target_feats_all[neg_ids]
            pos_feats = target_feats_all[pos_id].unsqueeze(0)
            target_feats = torch.cat([pos_feats, neg_feats])
            target_loss += self.infonce_loss(query_feats[i], target_feats,
                                             torch.tensor(0, device=self.device, dtype=torch.long), tau=self.tau)
        target_loss /= N
        return target_loss

    def bank_large_step(self, loss, text_feats, indexs, target_indexs,
                        refer_indexs=None):
        if self.plus:
            reference_image_feats = self.refer_bank[refer_indexs].detach().to(self.device)
        else:
            reference_image_feats = self.refer_bank[indexs].detach().to(self.device)
        query_feats = self.combining_function(reference_image_feats, text_feats)
        query_feats = F.normalize(query_feats)
        target_feats_all = self.target_bank.detach().to(self.device)
        target_indexs = target_indexs.to(self.device)
        if self.neg_num > 0:
            target_neg_loss = self.part_infonce_loss(query_feats, target_feats_all, target_indexs)
        else:
            target_neg_loss = self.infonce_loss(query_feats, target_feats_all, target_indexs, tau=self.tau)
        loss['bank_loss'] = target_neg_loss

    def forward(self, text, indexs, target_indexs, refer_indexs, refer_image=None, target_image=None):
        loss = dict()
        if not self.wo_bank:
            text_feats = self.encode_text(text)
            self.bank_large_step(loss, text_feats, indexs, target_indexs, refer_indexs)
        else:
            refer_image = Variable(refer_image, requires_grad=True)
            target_image = Variable(target_image, requires_grad=True)
            text_feats = self.encode_text(text)
            refer_image_feats = checkpoint(self.clip.encode_image, refer_image)
            target_image_feats = checkpoint(self.clip.encode_image, target_image)
            target_image_feats = checkpoint(F.normalize, target_image_feats)
            query_feats = checkpoint(self.combining_function, refer_image_feats, text_feats)
            query_feats = checkpoint(F.normalize, query_feats)
            loss['bbc_loss'] = self.infonce_loss(query_feats, target_image_feats, tau=self.tau)
        return loss

    def infonce_loss(self, query_features, target_features, labels=None, tau=0.01):
        logits = (query_features @ target_features.T) / tau
        if labels is None:
            labels = torch.arange(query_features.shape[0]).long().to(self.device)
        return self.crossentropy_criterion(logits, labels)
