import json
import os.path
from argparse import ArgumentParser

import cv2
import numpy as np
from tqdm import tqdm

from data_utils import CIRDataset
import copy


def search():
    images = []
    for image_path in tqdm(cirDataset.imagepaths):
        image = cv2.imread(image_path)
        images.append(image)
    cnt = 0
    same_imageids_list = []
    same_imageid2ap = dict()
    w = tqdm(enumerate(images))
    cnt_all = 0
    for i, image_query in w:
        same_imageids = [i]
        if not i in same_imageid2ap:
            same_imageid2ap[i] = 1
        else:
            continue
        flag = False
        for j, image_target in enumerate(images):
            if j in same_imageid2ap:
                continue
            if i == j:
                continue
            if np.equal(image_query.shape, image_target.shape).all() and np.equal(image_query, image_target).all():
                cnt_all += 1
                flag = True
                if not j in same_imageid2ap:
                    same_imageid2ap[j] = 1
                same_imageids.append(j)
        if flag is True:
            cnt += 1
            w.set_description(f'find {cnt} image sets，{cnt_all} images')

        same_imageids_list.append(same_imageids)
    print(cnt)
    with open(os.path.join(data_path, "same_image_list.json"), 'w') as f:
        f.write(json.dumps(same_imageids_list, ensure_ascii=False))


def check():
    with open(os.path.join(data_path, "same_image_list.json")) as f:
        same_imageids_list = json.loads(f.read())
        cnt = 0
        for same_imageids in tqdm(same_imageids_list):
            cnt += len(same_imageids)
        print(cnt, cirDataset.image_id)
        assert cnt == cirDataset.image_id
        print("successful!!")
    image_id = 0
    imagenames = []
    imagepaths = []
    imagename2id = dict()
    for same_imageids in tqdm(same_imageids_list):
        for same_imageid in same_imageids:
            image_name = cirDataset.imagenames[same_imageid]
            imagename2id[image_name] = image_id
        image_id += 1
        imagenames.append(cirDataset.imagenames[same_imageids[0]])
        imagepaths.append(cirDataset.imagepaths[same_imageids[0]])
    print(len(imagenames))
    with open(os.path.join(data_path, 'optimized_images.json'), 'w') as f:
        f.write(json.dumps((imagenames, imagepaths, imagename2id), ensure_ascii=False))


parser = ArgumentParser()
parser.add_argument("--dataset", default='fiq')
parser.add_argument("--data_path", default='fashionIQ_dataset')
dress_types = ['dress', 'shirt', 'toptee']
args = parser.parse_args()
data_path = args.data_path
dataset = args.dataset
cirDataset = CIRDataset(dataset, 'train', 'relative', None, data_path,
                        ['dress', 'shirt', 'toptee'])
search()
check()
