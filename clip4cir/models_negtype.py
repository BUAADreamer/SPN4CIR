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
                 device=torch.device('cuda'), neg_type=4):
        # neg_type: 1-7
        super().__init__()

        # initial main model
        self.device = device
        self.clip, self.preprocess = clip.load(clip_model_name, device=device, jit=False)
        self.clip = self.clip.float()
        self.combining_function = self.element_wise_sum
        self.tau = tau
        self.input_dim = self.clip.visual.input_resolution
        print("image size:", self.input_dim)
        self.output_dim = self.clip.visual.output_dim
        self.crossentropy_criterion = nn.CrossEntropyLoss()
        if transform == 'targetpad':
            self.preprocess = targetpad_transform(target_ratio, self.input_dim)
        self.neg_type = neg_type

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

    def text_neg_loss(self, refer_feats, text_feats, target_feats):
        text_loss = torch.tensor(0, dtype=torch.float).to(self.device)
        bs = 0
        for i, refer_feat in enumerate(refer_feats):
            ground_truth_i = torch.tensor(i, dtype=torch.long, device=self.device)
            query_feats_i = refer_feat + text_feats
            query_feats_i = F.normalize(query_feats_i)
            simmat_i = query_feats_i @ target_feats[i] / (
                self.tau)
            text_loss_i = self.crossentropy_criterion(simmat_i, ground_truth_i)
            text_loss += text_loss_i
            bs += 1
        text_loss /= bs
        return text_loss

    def refer_neg_loss(self, refer_feats, text_feats, target_feats):
        refer_loss = torch.tensor(0, dtype=torch.float).to(self.device)
        bs = 0
        for i, text_feat in enumerate(text_feats):
            ground_truth_i = torch.tensor(i, dtype=torch.long, device=self.device)
            query_feats_i = refer_feats + text_feat
            query_feats_i = F.normalize(query_feats_i)
            simmat_i = query_feats_i @ target_feats[i] / (self.tau)
            refer_loss_i = self.crossentropy_criterion(simmat_i, ground_truth_i)
            refer_loss += refer_loss_i
            bs += 1
        refer_loss /= bs
        return refer_loss

    def query_neg_loss(self, refer_feats, text_feats, target_feats):
        query_feats = F.normalize(refer_feats + text_feats)
        query_loss = self.infonce_loss(target_feats, query_feats, tau=self.tau)
        return query_loss

    def load_ckpt(self, model_path, is_origin=False):
        saved_state_dict = torch.load(model_path, map_location=self.device)
        if is_origin:
            self.clip.load_state_dict(saved_state_dict[self.clip.__class__.__name__], strict=False)
        else:
            self.load_state_dict(saved_state_dict['state_dict'], strict=False)

    def forward(self, text, indexs, target_indexs, refer_indexs, refer_image=None, target_image=None):
        loss_dict = dict()
        refer_image = Variable(refer_image, requires_grad=True)
        target_image = Variable(target_image, requires_grad=True)
        text_feats = self.encode_text(text)
        refer_image_feats = checkpoint(self.clip.encode_image, refer_image)
        target_image_feats = checkpoint(self.clip.encode_image, target_image)
        target_image_feats = checkpoint(F.normalize, target_image_feats)
        query_feats = checkpoint(self.combining_function, refer_image_feats, text_feats)
        query_feats = checkpoint(F.normalize, query_feats)
        target_loss = self.infonce_loss(query_feats, target_image_feats, tau=self.tau)
        refer_loss = self.refer_neg_loss(refer_image_feats, text_feats, target_image_feats)
        text_loss = self.text_neg_loss(refer_image_feats, text_feats, target_image_feats)
        query_loss = self.infonce_loss(target_image_feats, query_feats, tau=self.tau)
        loss = torch.tensor(0, dtype=torch.float).to(self.device)
        cnt = 0
        neg_type = self.neg_type
        if neg_type // 8 == 1:
            loss += query_loss
            cnt += 1
        neg_type = neg_type % 8
        if neg_type // 4 == 1:
            loss += target_loss
            cnt += 1
        neg_type = neg_type % 4
        if neg_type // 2 == 1:
            loss += text_loss
            cnt += 1
        neg_type = neg_type % 2
        if neg_type == 1:
            loss += refer_loss
            cnt += 1
        loss /= cnt
        loss_dict['bbc_loss'] = loss
        return loss_dict

    def infonce_loss(self, query_features, target_features, labels=None, tau=0.01):
        logits = (query_features @ target_features.T) / tau
        if labels is None:
            labels = torch.arange(query_features.shape[0]).long().to(self.device)
        return self.crossentropy_criterion(logits, labels)
