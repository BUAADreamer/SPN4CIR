import argparse
import json
import multiprocessing
import os.path
from pathlib import Path
import shutil
from typing import Union, List

import torch
from tqdm import tqdm

import clip
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

base_path = Path(__file__).absolute().parents[1].absolute()


def collate_fn(batch: list):
    """
    Discard None images in a batch when using torch DataLoader
    :param batch: input_batch
    :return: output_batch = input_batch - None_values
    """
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)


class ImageDataset(Dataset):
    def __init__(self, image_path: Union[str, List], preprocess):
        if isinstance(image_path, str):
            if os.path.isdir(image_path):
                self.image_path_list = os.listdir(image_path)
                self.image_path_list = [os.path.join(image_path, image_path_i) for image_path_i in self.image_path_list]
            elif image_path.endswith('.json'):
                with open(image_path) as f:
                    obj_list = json.loads(f.read())
                self.image_path_list = []
                for obj in obj_list:
                    self.image_path_list.append(obj['image_path'])
            else:
                self.image_path_list = [image_path]
        else:
            obj_list = image_path
            self.image_path_list = []
            for obj in obj_list:
                self.image_path_list.append(obj['image_path'])

        self.preprocess = preprocess

    def __getitem__(self, index):
        image_path = self.image_path_list[index]
        image = self.preprocess(Image.open(image_path))
        return image, image_path

    def __len__(self):
        return len(self.image_path_list)


def extract_image_features(cmr_model, imageDataset: ImageDataset, device):
    image_features = None
    data_loader = DataLoader(dataset=imageDataset, batch_size=32, num_workers=multiprocessing.cpu_count(),
                             pin_memory=True, collate_fn=collate_fn)
    image_paths = []
    for images, paths in tqdm(data_loader, desc='编码图像中'):
        images = images.to(device, non_blocking=True)
        with torch.no_grad():
            batch_features = cmr_model.encode_image(images)
            if image_features is not None:
                image_features = torch.vstack((image_features, batch_features))
            else:
                image_features = batch_features
            image_paths.extend(paths)
    image_features = F.normalize(image_features, dim=-1)
    return image_features, image_paths


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--query_image_path", default='')
    parser.add_argument("--target_image_path", default='')
    parser.add_argument("--retrieval_type", default='i2i', choices=['i2i', 't2t', 'i2t', 't2i'])
    parser.add_argument("--topk", default=20, type=int)
    parser.add_argument("--save", action='store_true')
    parser.add_argument("--output", default='test')
    args = parser.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    clip_model, clip_preprocess = clip.load("ViT-B/16", device=device, jit=False)
    if args.retrieval_type == 'i2i':
        query_image_dataset = ImageDataset(args.query_image_path, clip_preprocess)
        target_image_dataset = ImageDataset(args.target_image_path, clip_preprocess)
        query_image_features, query_image_paths = extract_image_features(clip_model, query_image_dataset, device)
        target_image_features, target_image_paths = extract_image_features(clip_model, target_image_dataset, device)
        distances = query_image_features @ target_image_features.T  # bs*dim @ dim*num = bs*num
        sorted_indices = torch.argsort(distances, dim=-1, descending=True).cpu()[:, :args.topk]
        if args.save:
            if not os.path.exists("retrieval_results"):
                os.mkdir("retrieval_results")
            args.output = os.path.join("retrieval_results", args.output)
            if not os.path.exists(args.output):
                os.mkdir(args.output)
            for i, sorted_indice in tqdm(enumerate(sorted_indices), desc='移动文件到输出文件夹中'):
                query_image_path = query_image_paths[i]
                out_path = os.path.join(args.output, f'{i}')
                if not os.path.exists(out_path):
                    os.mkdir(out_path)
                else:
                    shutil.rmtree(out_path)
                    os.mkdir(out_path)
                shutil.copy(query_image_path, os.path.join(out_path, 'query.png'))
                for rank, j in enumerate(sorted_indice):
                    target_image_path = target_image_paths[j]
                    shutil.copy(target_image_path, os.path.join(out_path, f'target_{rank}_{j}.png'))
