import multiprocessing
import os.path
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

import torch.nn.functional as F
from data_utils import targetpad_transform, CIRDataset
from utils import collate_fn
from lavis.models import load_model_and_preprocess


class CIRPlus(nn.Module):
    def __init__(self, blip_model_name, tau=0.07,
                 transform="targetpad", target_ratio=1.25,
                 device=torch.device('cuda'), plus=False):
        super().__init__()

        # initial main model
        self.blip_model, self.vis_processors, self.txt_processors = load_model_and_preprocess(name=blip_model_name,
                                                                                              model_type="pretrain",
                                                                                              is_eval=False,
                                                                                              device='cuda')
        update_method = getattr(self.blip_model, '_update_f_former', None)
        if callable(update_method):
            self.blip_model._update_f_former()
        self.input_dim = 224
        self.device = device
        print("image size:", self.input_dim)
        self.crossentropy_criterion = nn.CrossEntropyLoss()
        if transform == 'targetpad':
            self.preprocess = targetpad_transform(target_ratio, self.input_dim)
        self.tau = tau
        self.plus = plus

    def load_ckpt(self, model_path, is_origin=False):
        print('Trying to load the model')
        saved_state_dict = torch.load(model_path, map_location=self.device)
        if is_origin:
            self.blip_model.load_state_dict(saved_state_dict["Blip2QformerCirAlignPrompt"], strict=False)
            self.blip_model.init_stage2(self.tau)
        else:
            self.load_state_dict(saved_state_dict['state_dict'], strict=False)
            self.blip_model.init_stage2(self.tau)
            self.load_state_dict(saved_state_dict['state_dict'], strict=False)
        print('model loaded successfully')

    def extract_bank_features(self, cirDataset: CIRDataset, device, bank_path, reload_bank=False):
        if not os.path.exists(bank_path) or reload_bank:
            self.refer_bank = None
            self.target_bank = torch.zeros((cirDataset.image_id, 256), dtype=torch.float, device=self.device)
            data_loader = DataLoader(dataset=cirDataset, batch_size=32, num_workers=multiprocessing.cpu_count(),
                                     pin_memory=True, collate_fn=collate_fn)
            self.query_bank = None
            self.blip_model.eval().float()
            for reference_image, captions, target_image, index, \
                target_index, reference_index_all, target_index_all in tqdm(
                data_loader, desc='encoding bank features...'):
                reference_image = reference_image.to(device, non_blocking=True)
                target_image = target_image.to(device, non_blocking=True)
                text = [self.txt_processors["eval"](caption) for caption in captions]
                refer_hidden_states, target_feats, refer_feats, fusion_feats = self.blip_model.get_bank_feats(
                    reference_image, text,
                    target_image)
                refer_hidden_states = refer_hidden_states
                target_feats = target_feats
                refer_feats = refer_feats
                fusion_feats = fusion_feats
                if self.refer_bank is None:
                    self.refer_bank = refer_hidden_states
                    self.query_bank = fusion_feats
                else:
                    self.refer_bank = torch.cat((self.refer_bank, refer_hidden_states))
                    self.query_bank = torch.cat((self.query_bank, fusion_feats))
                self.target_bank[target_index_all] = target_feats
                self.target_bank[reference_index_all] = refer_feats
            self.refer_bank = self.refer_bank.detach().cpu()
            self.target_bank = self.target_bank.detach().cpu()
            self.query_bank = self.query_bank.detach().cpu()
            torch.save([self.refer_bank, self.target_bank, self.query_bank], bank_path)
        else:
            items = torch.load(bank_path)
            if len(items) == 2:
                self.refer_bank, self.target_bank = items
                self.query_bank = None
            elif len(items) == 3:
                self.refer_bank, self.target_bank, self.query_bank = items
        print("load bank successfully")

    def extract_refer_bank_features(self, cirDataset: CIRDataset, device, bank_path, reload_bank=False):
        if not os.path.exists(bank_path) or reload_bank:
            self.refer_bank = torch.zeros((cirDataset.image_id, 32, 768), dtype=torch.float, device=self.device)
            data_loader = DataLoader(dataset=cirDataset, batch_size=32, num_workers=multiprocessing.cpu_count(),
                                     pin_memory=True, collate_fn=collate_fn)
            self.query_bank = None
            self.blip_model.eval().float()
            for reference_image, captions, target_image, index, \
                target_index, reference_index_all, target_index_all in tqdm(
                data_loader, desc='encoding bank features...'):
                reference_image = reference_image.to(device, non_blocking=True)
                target_image = target_image.to(device, non_blocking=True)
                text = [self.txt_processors["eval"](caption) for caption in captions]
                refer_hidden_states, target_hidden_states = self.blip_model.get_refer_bank_feats(
                    reference_image, text,
                    target_image)
                self.refer_bank[reference_index_all] = refer_hidden_states
                self.refer_bank[target_index_all] = target_hidden_states
            self.refer_bank = self.refer_bank.detach().cpu()
            torch.save(self.refer_bank, bank_path)

    def load_refer_bank(self, bank_path):
        self.refer_bank = torch.load(bank_path)
        print("load reference bank successfully")

    def forward(self, text, indexs, target_indexs, refer_indexs):
        self.blip_model.train()
        text = [self.txt_processors["eval"](caption) for caption in text]
        target_feats = self.target_bank.detach().to(self.device)
        if self.plus:
            fusion_hidden_states = self.refer_bank.detach()[refer_indexs].to(self.device)
        else:
            fusion_hidden_states = self.refer_bank.detach()[indexs].to(self.device)
        target_indexs = target_indexs.to(self.device)
        return self.blip_model.forward_stage2(text, target_feats, fusion_hidden_states, target_indexs)

    def infonce_loss(self, query_features, target_features, labels=None, tau=0.01):
        logits = (query_features @ target_features.T) / tau
        if labels is None:
            labels = torch.arange(query_features.shape[0]).long().to(self.device)
        return self.crossentropy_criterion(logits, labels)
