import argparse
import json
import random

import torch

from data_utils_gen import CIRDataset, targetpad_transform
from tqdm import tqdm
import clip


def get_captions(caption1, caption2):
    prompt_list = [
        "{1} instead of {0}",
        "Unlike {0}, I want {1}",
        "{1}"
    ]
    captions = []
    for prompt_id in prompt_ids:
        caption = prompt_list[prompt_id].format(caption1, caption2)
        try:
            clip.tokenize(caption)
        except:
            caption = caption2
        captions.append(caption)
    return captions


def get_fiq():
    if args.i2i_rank >= 0:
        print("calculate i2i rank")
        sims_path = 'mm_data/fiq/sims.pth'
        sims_cross_i2t, sims_cross_t2i, sims_intra_i2i, sims_intra_t2t = torch.load(sims_path)
        i2i_ranks = torch.argsort(sims_intra_i2i, descending=True)
    it_list = []
    json_file = f'mm_data/fiq/fashioniq_it_{args.mllm}_{args.word_num}.json'
    with open(json_file) as f:
        it_list_ = json.loads(f.read())
        it_list.extend(it_list_)
    name2caption = dict()
    for it in it_list:
        name2caption[it['image_id']] = it['caption']
    preprocess = targetpad_transform(1.25, 288)
    relative_train_dataset = CIRDataset('fiq', 'train', 'relative', preprocess, "fashionIQ_dataset")
    refer2target = dict()
    N = len(relative_train_dataset.imagenames)
    print(N)
    print(len(relative_train_dataset.triplets))
    for triplet in relative_train_dataset.triplets:
        refer_name = triplet['reference_name']
        target_name = triplet['target_name']
        if refer_name not in relative_train_dataset.imagename2id or target_name not in relative_train_dataset.imagename2id:
            continue
        if refer_name not in refer2target:
            refer2target[refer_name] = dict()
        refer2target[refer_name][target_name] = 1

    def get_diff_id(i, N, k):
        if args.i2i_rank >= 0:
            if args.i2i_rank_max > args.i2i_rank:
                range_target = i2i_ranks[i].tolist()[args.i2i_rank:args.i2i_rank_max]
            else:
                range_target = i2i_ranks[i].tolist()[args.i2i_rank:]
        else:
            range_target = list(range(N))
            range_target.remove(i)
        j = random.sample(range_target, k)
        return j

    extend_triplets = []
    for i, name1 in tqdm(enumerate(relative_train_dataset.imagenames)):
        if args.refer and name1 not in refer2target:
            continue
        idx_list = get_diff_id(i, N, args.k)
        for idx in idx_list:
            name2 = relative_train_dataset.imagenames[idx]
            caption1 = name2caption[name1]
            caption2 = name2caption[name2]
            caption = get_captions(caption1, caption2)
            # caption = "{} instead of {}".format(caption2, caption1)
            triplet = {
                "target": name2,
                "candidate": name1,
                "captions": caption,
                "caption1": caption1,
                "caption2": caption2,
            }
            extend_triplets.append(triplet)
    if args.K > 0:
        extend_triplets = random.sample(extend_triplets, args.K)
    N = len(extend_triplets)
    if args.use_llm:
        for triplet in tqdm(extend_triplets):
            llm_caption = generate_modified_text(triplet['caption1'], triplet['caption2'], mod_type=1, data='fiq',
                                                 llm_type=args.use_llm)
            triplet['llm_caption'] = llm_caption
    print(len(extend_triplets))
    llm_cap = "_llm" if args.use_llm else ""
    with open(
            f"fashionIQ_dataset/captions/cap.extend_{args.model}{llm_cap}.train.json",
            'w') as f:
        f.write(json.dumps(extend_triplets))


def get_cirr():
    if args.i2i_rank >= 0:
        sims_path = 'mm_data/cirr_dataset/sims.pth'
        sims_cross_i2t, sims_cross_t2i, sims_intra_i2i, sims_intra_t2t = torch.load(sims_path)
        i2i_ranks = torch.argsort(sims_intra_i2i, descending=True)
    json_file = f'mm_data/cirr/cirr_it_{args.mllm}_{args.word_num}.json'
    with open(json_file) as f:
        it_list = json.loads(f.read())
    name2caption = dict()
    for it in it_list:
        name2caption[it['image_id']] = it['caption']
    preprocess = targetpad_transform(1.25, 288)
    relative_train_dataset = CIRDataset('cirr', 'train', 'relative', preprocess, "cirr_dataset")
    refer2target = dict()
    N = len(relative_train_dataset.imagenames)
    print(N)
    print(len(relative_train_dataset.triplets))
    for triplet in relative_train_dataset.triplets:
        refer_name = triplet['reference_name']
        target_name = triplet['target_name']
        if refer_name not in relative_train_dataset.imagename2id or target_name not in relative_train_dataset.imagename2id:
            continue
        if refer_name not in refer2target:
            refer2target[refer_name] = dict()
        refer2target[refer_name][target_name] = 1

    def get_diff_id(i, N, k):
        if args.i2i_rank >= 0:
            if args.i2i_rank_max > args.i2i_rank:
                range_target = i2i_ranks[i].tolist()[args.i2i_rank:args.i2i_rank_max]
            else:
                range_target = i2i_ranks[i].tolist()[args.i2i_rank:]
        else:
            range_target = list(range(N))
            range_target.remove(i)
        j = random.sample(range_target, k)
        return j

    extend_triplets = []
    for i, name1 in tqdm(enumerate(relative_train_dataset.imagenames)):
        # if name1 in refer2target:
        if args.refer and name1 not in refer2target:
            continue
        idx_list = get_diff_id(i, N, args.k)
        for idx in idx_list:
            name2 = relative_train_dataset.imagenames[idx]
            caption1 = name2caption[name1]
            caption2 = name2caption[name2]
            caption = get_captions(caption1, caption2)
            triplet = {
                "target_hard": name2,
                "reference": name1,
                "caption": caption,
                "pairid": 0,
                "img_set": {"members": ["xxx"]},
                "caption1": caption1,
                "caption2": caption2,
            }
            extend_triplets.append(triplet)
    if args.K > 0:
        extend_triplets = random.sample(extend_triplets, args.K)
    N = len(extend_triplets)
    if args.use_llm:
        for triplet in tqdm(extend_triplets):
            llm_caption = generate_modified_text(triplet['caption1'], triplet['caption2'], data='cirr', mod_type=0,
                                                 llm_type=args.use_llm)
            triplet['llm_caption'] = llm_caption
    print(len(extend_triplets))
    llm_cap = "_llm" if args.use_llm else ""
    with open(
            f"cirr_dataset/cirr/captions/cap.rc2.train.extend_{args.model}{llm_cap}.json",
            'w') as f:
        f.write(json.dumps(extend_triplets))


def get_cc():
    if 'fiq' in args.data:
        id_ls = [
            0, 64, 128, 192,
        ]
    else:
        id_ls = [
            0, 64, 128, 192,
            32, 96, 160,
        ]
    it_list = []
    json_file = f'mm_data/zs/cc_it_0000_{args.mllm}_{args.word_num}.json'
    for cc_id in id_ls:
        with open(json_file.replace("0000", str(cc_id))) as f:
            it_list.extend(json.loads(f.read()))
    N = len(it_list)
    print("image num:", N)

    def get_diff_id(i, N, k) -> list:
        range_target = list(range(N))
        range_target.remove(i)
        j = random.sample(range_target, k)
        return j

    triplets = []
    for i, it in tqdm(enumerate(it_list)):
        id_list = get_diff_id(i, N, args.k)
        for j in id_list:
            triplets.append({
                "target": it_list[j]['image_path'],
                "reference": it_list[i]['image_path'],
                "reference_name": it_list[i]['image_id'],
                "target_name": it_list[j]['image_id'],
                "captions": get_captions(it_list[i]['caption'], it_list[j]['caption']),
            })
    flag_2 = "2" if "2" in args.data else ""
    if 'fiq' in args.data:
        file_path = f"/root_path/fashionIQ_dataset/captions/cap.cc{flag_2}.train.json"
    else:
        file_path = f"/root_path/cirr_dataset/cirr/captions/cap.rc2.train.cc{flag_2}.json"

    with open(file_path,
              'w') as f:
        f.write(json.dumps(triplets))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default='fiq')
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--refer", action='store_true')
    parser.add_argument("--model", default='clip',
                        choices=['clip', 'blip', 'blip2', 'zs', "tgcir", "amc", "cirplant", "zs2"])
    parser.add_argument("--k", default=1, type=int)
    parser.add_argument("--K", default=-1, type=int)
    parser.add_argument("--i2i_rank", default=-1, type=int)
    parser.add_argument("--i2i_rank_max", default=-1, type=int)
    parser.add_argument("--use_llm", default=0, type=int)
    parser.add_argument("--p_list", default='0,1')
    parser.add_argument("--mllm", default='llava', choices=['blip', 'blip2', 'llava'])
    parser.add_argument("--word_num", default=10, type=int)
    args = parser.parse_args()
    if args.use_llm:
        from llama_generate import generate_modified_text
    random.seed(args.seed)
    prompt_ids = list(map(int, args.p_list.split(",")))
    if args.data == 'fiq':
        get_fiq()
    elif args.data == 'cirr':
        get_cirr()
    else:
        get_cc()
