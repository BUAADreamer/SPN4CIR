import json
import os
import random
from pathlib import Path
from typing import List

import PIL
import PIL.Image
import torchvision.transforms.functional as F
from torch.utils.data import Dataset
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize

base_path = Path(__file__).absolute().parents[1].absolute()


def _convert_image_to_rgb(image):
    return image.convert("RGB")


class SquarePad:
    """
    Square pad the input image with zero padding
    """

    def __init__(self, size: int):
        """
        For having a consistent preprocess pipeline with CLIP we need to have the preprocessing output dimension as
        a parameter
        :param size: preprocessing output dimension
        """
        self.size = size

    def __call__(self, image):
        w, h = image.size
        max_wh = max(w, h)
        hp = int((max_wh - w) / 2)
        vp = int((max_wh - h) / 2)
        padding = [hp, vp, hp, vp]
        return F.pad(image, padding, 0, 'constant')


class TargetPad:
    """
    Pad the image if its aspect ratio is above a target ratio.
    Pad the image to match such target ratio
    """

    def __init__(self, target_ratio: float, size: int):
        """
        :param target_ratio: target ratio
        :param size: preprocessing output dimension
        """
        self.size = size
        self.target_ratio = target_ratio

    def __call__(self, image):
        w, h = image.size
        actual_ratio = max(w, h) / min(w, h)
        if actual_ratio < self.target_ratio:  # check if the ratio is above or below the target ratio
            return image
        scaled_max_wh = max(w, h) / self.target_ratio  # rescale the pad to match the target ratio
        hp = max(int((scaled_max_wh - w) / 2), 0)
        vp = max(int((scaled_max_wh - h) / 2), 0)
        padding = [hp, vp, hp, vp]
        return F.pad(image, padding, 0, 'constant')


def squarepad_transform(dim: int):
    """
    CLIP-like preprocessing transform on a square padded image
    :param dim: image output dimension
    :return: CLIP-like torchvision Compose transform
    """
    return Compose([
        SquarePad(dim),
        Resize(dim, interpolation=PIL.Image.BICUBIC),
        CenterCrop(dim),
        _convert_image_to_rgb,
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])


def targetpad_transform(target_ratio: float, dim: int):
    """
    CLIP-like preprocessing transform computed after using TargetPad pad
    :param target_ratio: target ratio for TargetPad
    :param dim: image output dimension
    :return: CLIP-like torchvision Compose transform
    """
    return Compose([
        TargetPad(target_ratio, dim),
        Resize(dim, interpolation=PIL.Image.BICUBIC),
        CenterCrop(dim),
        _convert_image_to_rgb,
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])


def generate_randomized_fiq_caption(captions, type=-1):
    random_num = random.random()
    if type == 0:
        random_num = 0.12
    elif type == 1:
        random_num = 0.37
    elif type == 2:
        random_num = 0.62
    elif type == 3:
        random_num = 0.88
    if random_num < 0.25:
        caption = f"{captions[0].strip('.?, ')} and {captions[1].strip('.?, ')}"
    elif 0.25 < random_num < 0.5:
        caption = f"{captions[1].strip('.?, ')} and {captions[0].strip('.?, ')}"
    elif 0.5 < random_num < 0.75:
        caption = f"{captions[0].strip('.?, ')}"
    else:
        caption = f"{captions[1].strip('.?, ')}"
    return caption


class CIRDataset(Dataset):
    def __init__(self, data_name, split, mode, preprocess, data_path='./', dress_types=None, val_ret_train=False,
                 fiq_val_type=0, plus=False, llmcap=False):
        # verify dress_types for FashionIQ
        if dress_types is None:
            dress_types = ['dress', 'shirt', 'toptee']
        else:
            for dress_type in dress_types:
                assert dress_type in ['dress', 'shirt', 'toptee']
        self.data_name = data_name
        self.split = split
        self.mode = mode
        self.preprocess = preprocess
        self.data_path = data_path
        self.dress_types = dress_types  # FashionIQ
        self.triplets: List[dict] = []
        self.targetname2id = dict()
        self.use_bank = False
        self.val_ret_train = val_ret_train
        self.fiq_val_type = fiq_val_type
        self.imagename2id = dict()
        self.imagenames = []
        self.imagepaths = []
        if self.data_name == 'fiq':
            self.caption_path = os.path.join(self.data_path, 'captions')
            self.image_path = os.path.join(self.data_path, 'images')
            for dress_type in dress_types:
                with open(os.path.join(self.caption_path, f'cap.{dress_type}.{self.split}.json')) as f:
                    self.triplets.extend(json.load(f))
            self.N = len(self.triplets)
            if self.split == 'train' and plus:
                llm_extend = "_llm" if llmcap else ""
                with open(os.path.join(self.caption_path, f'cap.extend_clip{llm_extend}.train.json')) as f:
                    extend_triplets = json.load(f)
                    if llmcap:
                        for triplet in extend_triplets:
                            triplet['captions'] = [
                                triplet['llm_caption'],
                                # triplet['captions'][0],
                            ]
                    self.triplets.extend(extend_triplets)
            self.triplets = [
                {
                    "reference": os.path.join(self.image_path, f'{triplet["candidate"]}.png'),
                    "reference_name": triplet["candidate"],
                    "target": os.path.join(self.image_path, f'{triplet["target"]}.png'),
                    "target_name": triplet["target"],
                    "captions": triplet['captions']
                }
                for triplet in self.triplets
            ]
            self.image_names: list = []  # FashionIQ
            for dress_type in dress_types:
                with open(
                        os.path.join(self.data_path, 'image_splits', f'split.{dress_type}.{self.split}.json')) as f:
                    self.image_names.extend(json.load(f))
            self.val_image_names: list = []
            if self.fiq_val_type == 1 and self.split == 'val':
                for triplet in self.triplets:
                    self.val_image_names.append(triplet['reference_name'])
                    self.val_image_names.append(triplet['target_name'])
                self.val_image_names = list(set(self.val_image_names))
        elif self.data_name == 'cirr':
            self.caption_path = os.path.join(self.data_path, 'cirr/captions')
            self.image_splits_path = os.path.join(self.data_path, 'cirr/image_splits')
            self.image_path = self.data_path
            with open(os.path.join(self.caption_path, f'cap.rc2.{self.split}.json')) as f:
                self.triplets = json.load(f)
            with open(os.path.join(self.image_splits_path, f'split.rc2.{self.split}.json')) as f:
                self.name_to_relpath = json.load(f)
            self.N = len(self.triplets)
            if self.split == 'train' and plus:
                llm_extend = "_llm" if llmcap else ""
                with open(os.path.join(self.caption_path, f'cap.rc2.train.extend_clip{llm_extend}.json')) as f:
                    extend_triplets = json.load(f)
                    if llmcap:
                        for triplet in extend_triplets:
                            triplet['caption'] = [
                                triplet['llm_caption'],
                                # triplet['caption1'],
                                # triplet['caption2'],
                            ]
                    self.triplets.extend(extend_triplets)
            self.triplets = [
                {
                    "reference": os.path.join(self.image_path, self.name_to_relpath[
                        triplet["reference"]]),
                    "reference_name": triplet["reference"],
                    "target": os.path.join(self.image_path,
                                           self.name_to_relpath[
                                               triplet["target_hard"]]) if "target_hard" in triplet else "",
                    "target_name": triplet["target_hard"] if "target_hard" in triplet else "",
                    "captions": [triplet['caption']] if isinstance(triplet['caption'], str) else triplet['caption'],
                    "pairid": triplet['pairid'],
                    "group_members": triplet['img_set']['members']
                }
                for triplet in self.triplets
            ]
        if split == 'train':
            self.target_id = 0
            self.image_id = 0
            for triplet in self.triplets:
                refer_name = triplet['reference_name']
                target_name = triplet['target_name']
                if target_name not in self.targetname2id:
                    self.targetname2id[target_name] = self.target_id
                    self.target_id += 1
                if refer_name not in self.imagename2id:
                    self.imagename2id[refer_name] = self.image_id
                    self.image_id += 1
                    self.imagenames.append(refer_name)
                if target_name not in self.imagename2id:
                    self.imagename2id[target_name] = self.image_id
                    self.image_id += 1
                    self.imagenames.append(target_name)
            self.imagepaths = [
                os.path.join(self.image_path, f'{image_name}.png')
                if self.data_name == 'fiq' else
                os.path.join(self.image_path, self.name_to_relpath[image_name])
                for image_name in self.imagenames
            ]
            optimized_images_file = os.path.join(self.data_path, "optimized_images.json")
            if os.path.exists(optimized_images_file):
                with open(optimized_images_file) as f:
                    self.imagenames, self.imagepaths, self.imagename2id = json.loads(f.read())
                    self.image_id = len(self.imagenames)
            print("target number:", self.target_id)
            print("image number", self.image_id)
        print(f"CIRDataset {self.data_name} {self.split} {self.mode} initial successfully")

    def __getitem__(self, index):
        if self.mode == 'relative':
            triplet = self.triplets[index]
            reference_name = triplet['reference_name']
            reference_image_path = triplet['reference']
            captions = triplet['captions']
            if self.split == 'train':
                target_index = self.targetname2id[triplet['target_name']]
                reference_index_all = self.imagename2id[triplet['reference_name']]
                target_index_all = self.imagename2id[triplet['target_name']]
                if len(captions) > 1:
                    if self.data_name == 'fiq' and index < self.N:
                        caption = generate_randomized_fiq_caption(captions)
                    else:
                        caption = random.choice(captions)
                else:
                    caption = captions[0]
                if self.use_bank:
                    return caption, index, target_index, target_index_all, reference_index_all
                else:
                    reference_image = self.preprocess(PIL.Image.open(reference_image_path))
                    target_image_path = triplet['target']
                    target_image = self.preprocess(PIL.Image.open(target_image_path))
                    return reference_image, caption, target_image, index, target_index, reference_index_all, target_index_all
            elif self.split == 'val' and self.val_ret_train:
                if len(captions) > 1:
                    caption = generate_randomized_fiq_caption(captions, type=0)
                else:
                    caption = captions[0]
                reference_image_path = triplet['reference']
                reference_image = self.preprocess(PIL.Image.open(reference_image_path))
                target_image_path = triplet['target']
                target_image = self.preprocess(PIL.Image.open(target_image_path))
                return reference_image, caption, target_image
            elif self.split == 'val':
                target_name = triplet['target_name']
                if self.data_name == 'fiq':
                    return reference_name, target_name, captions
                elif self.data_name == 'cirr':
                    return reference_name, target_name, captions[0], triplet['group_members']
            elif self.split == 'test1':
                assert self.data_name == 'cirr'
                pair_id = triplet['pairid']
                return pair_id, reference_name, captions[0], triplet['group_members']


        elif self.mode == 'classic':
            if self.data_name == 'fiq':
                if self.fiq_val_type == 0:  # original
                    image_name = self.image_names[index]
                    image_path = os.path.join(self.image_path, f"{image_name}.png")
                    image = self.preprocess(PIL.Image.open(image_path))
                    return image_name, image
                elif self.fiq_val_type == 1:  # VAL set
                    assert self.split == 'val'
                    image_name = self.val_image_names[index]
                    image_path = os.path.join(self.image_path, f"{image_name}.png")
                    image = self.preprocess(PIL.Image.open(image_path))
                    return image_name, image
            elif self.data_name == 'cirr':
                image_name = list(self.name_to_relpath.keys())[index]
                image_path = base_path / 'cirr_dataset' / self.name_to_relpath[image_name]
                image = self.preprocess(PIL.Image.open(image_path))
                return image_name, image

    def __len__(self):
        if self.mode == 'relative':
            return len(self.triplets)
        elif self.mode == 'classic':
            if self.data_name == 'fiq':
                if self.fiq_val_type == 0:
                    return len(self.image_names)
                else:
                    return len(self.val_image_names)
            elif self.data_name == 'cirr':
                return len(self.name_to_relpath)
