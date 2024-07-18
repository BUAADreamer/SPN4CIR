import math
import multiprocessing
import os.path
import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.checkpoint import checkpoint
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn.functional as F
from data_utils import targetpad_transform, CIRDataset
from utils import collate_fn
from blip_cir import blip_cir


class CIRPlus(nn.Module):
    def __init__(self, blip_model_name, tau=0.01,
                 transform="targetpad", target_ratio=1.25, encoder='both',
                 device=torch.device('cuda'), plus=False):
        super().__init__()

        # initial main model
        self.device = device
        self.plus = plus
        self.blip = blip_cir(
            pretrained=blip_model_name,
            image_size=384, vit='base',
            vit_grad_ckpt=True, vit_ckpt_layer=4).to(self.device)
        self.tau = nn.Parameter(tau * torch.ones([]))
        self.input_dim = 384
        print("image size:", self.input_dim)
        self.output_dim = 256
        self.crossentropy_criterion = nn.CrossEntropyLoss()
        if transform == 'targetpad':
            self.preprocess = targetpad_transform(target_ratio, self.input_dim)
        self.encoder = encoder

    def load_ckpt(self, model_path, is_origin=False):
        saved_state_dict = torch.load(model_path, map_location=self.device)
        if is_origin:
            self.blip.load_state_dict(saved_state_dict["BLIP_Retrieval"], strict=False)
        else:
            self.load_state_dict(saved_state_dict['state_dict'], strict=False)

    def extract_bank_features(self, cirDataset: CIRDataset, device, bank_path, reload_bank=False):
        self.blip.eval().float()
        if not os.path.exists(bank_path) or reload_bank:
            self.refer_bank = torch.zeros(len(cirDataset), 577, 768)
            self.target_bank = torch.zeros(cirDataset.image_id, self.output_dim)
            data_loader = DataLoader(dataset=cirDataset, batch_size=32, num_workers=multiprocessing.cpu_count(),
                                     pin_memory=True, collate_fn=collate_fn)
            for reference_image, caption, target_image, index, \
                target_index, reference_index_all, target_index_all in tqdm(
                data_loader, desc='encoding bank features...'):
                reference_image = reference_image.to(device, non_blocking=True)
                target_image = target_image.to(device, non_blocking=True)

                with torch.no_grad():
                    batch_refer_features = self.blip.img_embed(reference_image)
                    batch_target_features_p = self.blip.img_embed(target_image, return_pool_and_normalized=True)[
                        -1].detach().cpu()
                    batch_refer_features_p = F.normalize(
                        self.blip.vision_proj(batch_refer_features[:, 0, :])).detach().cpu()
                    batch_refer_features = batch_refer_features.detach().cpu()
                    self.refer_bank[index] = batch_refer_features
                    self.target_bank[reference_index_all] = batch_refer_features_p
                    self.target_bank[target_index_all] = batch_target_features_p
            torch.save([self.refer_bank, self.target_bank], bank_path)
        else:
            self.refer_bank, self.target_bank = torch.load(bank_path)
        print("load bank successfully")

    def extract_refer_bank_features(self, cirDataset: CIRDataset, device, bank_path, reload_bank=False):
        self.blip.eval().float()
        if not os.path.exists(bank_path) or reload_bank:
            self.refer_bank = torch.zeros(cirDataset.image_id, 577, 768)
            data_loader = DataLoader(dataset=cirDataset, batch_size=32, num_workers=multiprocessing.cpu_count(),
                                     pin_memory=True, collate_fn=collate_fn)
            for reference_image, caption, target_image, index, \
                target_index, reference_index_all, target_index_all in tqdm(
                data_loader, desc='encoding bank features...'):
                reference_image = reference_image.to(device, non_blocking=True)
                target_image = target_image.to(device, non_blocking=True)
                with torch.no_grad():
                    batch_refer_features = self.blip.img_embed(reference_image).detach().cpu()
                    batch_target_features = self.blip.img_embed(target_image).detach().cpu()
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
        query_feats = self.blip.img_txt_fusion(reference_image_feats, None, text)
        target_feats_all = self.target_bank.detach().to(self.device)
        target_indexs = target_indexs.to(self.device)
        target_neg_loss = self.infonce_loss(query_feats, target_feats_all, target_indexs, tau=self.tau)
        loss['bank_loss'] = target_neg_loss

    def forward(self, text, indexs, target_indexs, refer_indexs, reference_image=None, target_image=None):
        loss = dict()
        self.bank_large_step(loss, text, indexs, target_indexs, refer_indexs)
        return loss

    def get_likelihood_probs(self, mu, sigma, sims) -> torch.Tensor:
        factor_item = 1 / (sigma * math.sqrt(2 * math.pi))
        exp_item = torch.exp(-1 / (2 * sigma ** 2) * ((sims - mu) ** 2))
        return factor_item * exp_item

    def infonce_loss(self, query_features, target_features, labels=None, tau=0.01):
        logits = (query_features @ target_features.T) / tau
        if labels is None:
            labels = torch.arange(query_features.shape[0]).long().to(self.device)
        return self.crossentropy_criterion(logits, labels)
