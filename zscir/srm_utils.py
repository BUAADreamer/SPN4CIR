import multiprocessing
import os.path
from argparse import ArgumentParser

import PIL
from torch.utils.data import DataLoader
from torchvision.transforms import transforms, Compose, Resize, InterpolationMode, CenterCrop, ToTensor, Normalize
from tqdm import tqdm

import clip
import torch
from torch import nn
from transformers import ViTModel, BertModel, BertTokenizer, AutoTokenizer, AutoModel
import torch.nn.functional as F
from PIL import Image
from utils import collate_fn

from data_utils import CIRDataset
import unicom


def _convert_image_to_rgb(image):
    return image.convert("RGB")


class WarpModule(torch.nn.Module):
    def __init__(self, model) -> None:
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model(x)


class SRMCMRModel(nn.Module):
    def __init__(self, device=torch.device('cuda')):
        super().__init__()
        self.device = device
        bge_path = 'BAAI/bge-base-en'
        unicom_name = 'ViT-L/14'
        print("srm load model begin")
        self.visual_encoder, self.transform = unicom.load(unicom_name)
        self.visual_encoder = self.visual_encoder.cuda()
        self.visual_encoder = WarpModule(self.visual_encoder)
        self.visual_encoder.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(bge_path)
        self.text_encoder = AutoModel.from_pretrained(bge_path).to(device).eval()
        print("srm load model successfully")
        self.text_dim = 768
        self.visual_dim = 768

    def encode_image(self, image):
        if not isinstance(image, torch.Tensor):
            image = self.transform(image)
            image = image.to(self.device)
        image_feats = self.visual_encoder(image)
        image_feats = F.normalize(image_feats)
        return image_feats

    def encode_text(self, text):
        if not isinstance(text, torch.Tensor):
            text_inputs = self.tokenizer(text, padding=True, truncation=True, return_tensors='pt')
        else:
            text_inputs = text
        model_output = self.text_encoder(input_ids=text_inputs['input_ids'].to(self.device),
                                         attention_mask=text_inputs['attention_mask'].to(self.device))
        text_feats = model_output[0][:, 0]
        text_feats = F.normalize(text_feats)
        return text_feats


def extract_cir_features(cmr_model, cirDataset: CIRDataset, device, model_name='clip'):
    visual_dim = cmr_model.visual_dim if model_name != 'clip' else cmr_model.visual.output_dim
    text_dim = cmr_model.text_dim if model_name != 'clip' else cmr_model.visual.output_dim
    image_features = torch.randn(cirDataset.image_id, visual_dim)
    text_features = torch.randn(len(cirDataset), text_dim)
    data_loader = DataLoader(dataset=cirDataset, batch_size=32, num_workers=multiprocessing.cpu_count(),
                             pin_memory=True, collate_fn=collate_fn)
    with torch.no_grad():
        for batch_data in tqdm(data_loader, desc='encoding triplets...'):
            reference_image, caption, target_image, index, \
            target_index, reference_index_all, target_index_all = batch_data
            reference_image = reference_image.to(device, non_blocking=True)
            target_image = target_image.to(device, non_blocking=True)
            if model_name == 'clip':
                caption = clip.tokenize(caption).to(device)
            batch_refer_feats = F.normalize(cmr_model.encode_image(reference_image)).detach().cpu().to(torch.float32)
            batch_target_feats = F.normalize(cmr_model.encode_image(target_image)).detach().cpu().to(torch.float32)
            batch_text_feats = F.normalize(cmr_model.encode_text(caption)).detach().cpu().to(torch.float32)
            text_features[index] = batch_text_feats
            image_features[target_index_all] = batch_target_feats
            image_features[reference_index_all] = batch_refer_feats
    return image_features, text_features


def get_features():
    if not os.path.exists(srm_feats_path):
        # SRM
        srm_model = SRMCMRModel()
        cirDataset_srm = CIRDataset(args.dataset, 'train', 'relative', srm_model.transform, args.data_path,
                                    dress_types)
        srm_image_feats, srm_text_feats = extract_cir_features(srm_model, cirDataset_srm, device, 'srm')
        # clip
        clip_model, preprocess = clip.load('ViT-L/14', device=device, jit=False)
        clip_model.eval()
        cirDataset_clip = CIRDataset(args.dataset, 'train', 'relative', preprocess, args.data_path,
                                     dress_types)
        clip_image_feats, clip_text_feats = extract_cir_features(clip_model, cirDataset_clip, device)
        # save
        torch.save([srm_image_feats, srm_text_feats, clip_image_feats, clip_text_feats], srm_feats_path)
        return clip_image_feats, clip_text_feats, srm_image_feats, srm_text_feats
    else:
        print("loading features...")
        clip_image_feats, clip_text_feats, srm_image_feats, srm_text_feats = torch.load(srm_feats_path)
        print("loading features successfully")
        return clip_image_feats, clip_text_feats, srm_image_feats, srm_text_feats


def calcu_sims():
    if not os.path.exists(sims_path):
        sims_cross_i2t = clip_image_feats @ clip_text_feats.T  # NxM
        sims_cross_t2i = sims_cross_i2t.T  # MxN
        sims_intra_i2i = srm_image_feats @ srm_image_feats.T  # NxN
        sims_intra_t2t = srm_text_feats @ srm_text_feats.T  # MxM
        torch.save([sims_cross_i2t, sims_cross_t2i, sims_intra_i2i, sims_intra_t2t], sims_path)
        return sims_cross_i2t, sims_cross_t2i, sims_intra_i2i, sims_intra_t2t
    else:
        print("loading sims...")
        sims_cross_i2t, sims_cross_t2i, sims_intra_i2i, sims_intra_t2t = torch.load(sims_path)
        print("loading sims successfully")
        return sims_cross_i2t, sims_cross_t2i, sims_intra_i2i, sims_intra_t2t


def trans_captions(captions):
    if isinstance(captions, list):
        caption = f"{captions[0].strip('.?, ')} and {captions[1].strip('.?, ')}"
    else:
        caption = captions
    return caption


def get_srm_out():
    cirDataset = CIRDataset(args.dataset, 'train', 'relative', None, args.data_path, dress_types)
    triplets = cirDataset.triplets
    print("sorting sims...")
    i2i_ranks = torch.argsort(sims_intra_i2i, descending=True)
    i2t_ranks = torch.argsort(sims_cross_i2t, descending=True)
    t2i_ranks = torch.argsort(sims_cross_t2i, descending=True)
    sims_intra_i2i.fill_diagonal_(-10000)
    sims_intra_t2t.fill_diagonal_(-10000)
    i2i_ranks_neg = torch.argsort(sims_intra_i2i, descending=True)
    t2t_ranks_neg = torch.argsort(sims_intra_t2t, descending=True)
    print("sorting sims successfully")
    N = len(triplets)
    K = 1000
    rt_scores, rm_scores, mt_scores = torch.ones(N, dtype=torch.long), \
                                      torch.ones(N, dtype=torch.long), \
                                      torch.ones(N, dtype=torch.long)
    m_fn_indexs = torch.ones((N, K), dtype=torch.long) * -10
    t_fn_indexs = torch.ones((N, K), dtype=torch.long) * -10
    r_fn_indexs = torch.ones((N, K), dtype=torch.long) * -10
    for index, triplet in tqdm(enumerate(triplets), desc='Get srm data...'):
        reference_index = cirDataset.imagename2id[triplet['reference_name']]
        target_index = cirDataset.imagename2id[triplet['target_name']]
        r2t_rank = torch.where(i2i_ranks[reference_index] == target_index)[0][0].item()
        t2r_rank = torch.where(i2i_ranks[target_index] == reference_index)[0][0].item()
        m2r_rank = torch.where(t2i_ranks[index] == reference_index)[0][0].item()
        r2m_rank = torch.where(i2t_ranks[reference_index] == index)[0][0].item()
        m2t_rank = torch.where(t2i_ranks[index] == target_index)[0][0].item()
        t2m_rank = torch.where(i2t_ranks[target_index] == index)[0][0].item()
        rt_scores[index] = (r2t_rank + t2r_rank)
        rm_scores[index] = (r2m_rank + m2r_rank)
        mt_scores[index] = (m2t_rank + t2m_rank)

        target_negs = i2i_ranks_neg[target_index][:K]
        text_negs = t2t_ranks_neg[index][:K]
        refer_negs = i2i_ranks_neg[reference_index][:K]
        t_fn_indexs[index] = target_negs
        m_fn_indexs[index] = text_negs
        r_fn_indexs[index] = refer_negs
    print("saveing srm...")
    torch.save([r_fn_indexs, m_fn_indexs, t_fn_indexs, rt_scores, rm_scores, mt_scores], srm_path)
    print("saveing srm successfully")


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--dataset", default='fiq')
    parser.add_argument("--data_path", default='')
    parser.add_argument("--output_path", default='mm_data')
    dress_types = ['dress', 'shirt', 'toptee']
    args = parser.parse_args()
    if args.data_path == '':
        if args.dataset == 'fiq':
            args.data_path = 'fashionIQ_dataset'
        else:
            args.data_path = 'cirr_dataset'
    args.output_path = os.path.join(args.output_path, args.dataset)
    device = torch.device('cuda')
    srm_feats_path = os.path.join(args.output_path, "srm_feats.pth")
    srm_path = os.path.join(args.output_path, "srm.pth")
    sims_path = os.path.join(args.output_path, 'sims.pth')
    clip_image_feats, clip_text_feats, srm_image_feats, srm_text_feats = get_features()
    sims_cross_i2t, sims_cross_t2i, sims_intra_i2i, sims_intra_t2t = calcu_sims()
    get_srm_out()
