import json
import shutil
from argparse import ArgumentParser

from tqdm import tqdm
import os
from data_utils import CIRDataset


def get_fiq_case():
    model_name1 = 'base'
    model_name2 = 'full'
    casedata_path_1 = f'cases/fiq/{model_name1}.json'
    casedata_path_2 = f'cases/fiq/{model_name2}.json'
    with open(casedata_path_1) as f:
        casedata_ls_1 = json.loads(f.read())
    with open(casedata_path_2) as f:
        casedata_ls_2 = json.loads(f.read())
    case_path = f'cases/fiq/base-full'
    os.makedirs(case_path, exist_ok=True)
    cnt = 0
    for i in tqdm(range(len(casedata_ls_1))):
        triplet = cirDataset.triplets[i]
        refer_path = triplet['reference']
        target_path = triplet['target']
        caption = f"#1:{triplet['captions'][0]}\n\n#2:{triplet['captions'][1]}"
        output_path = os.path.join(case_path, f"{i}")
        k_1 = casedata_ls_1[i]["k"]
        k_2 = casedata_ls_2[i]["k"]
        if k_1 >= 10 and k_2 == 0:
            cnt += 1
            # continue
            os.makedirs(output_path, exist_ok=True)
            shutil.copy(refer_path, os.path.join(output_path, "refer.png"))
            shutil.copy(target_path, os.path.join(output_path, "target.png"))
            with open(os.path.join(output_path, "caption.txt"), 'w') as f:
                caption += f"rank1:{k_1} rank2:{k_2}"
                f.write(caption)
            top_k_names_1 = casedata_ls_1[i]["top_k_names"]
            top_k_names_2 = casedata_ls_2[i]["top_k_names"]
            for i, name in enumerate(top_k_names_1):
                image_path = os.path.join(cirDataset.image_path, f"{name}.png")
                shutil.copy(image_path, os.path.join(output_path, f"{model_name1}_{i}.png"))
            for i, name in enumerate(top_k_names_2):
                image_path = os.path.join(cirDataset.image_path, f"{name}.png")
                shutil.copy(image_path, os.path.join(output_path, f"{model_name2}_{i}.png"))
    print("case num:", cnt)


def get_cirr_case():
    model_name1 = 'base'
    model_name2 = 'full'
    casedata_path_1 = f'cases/cirr/{model_name1}.json'
    casedata_path_2 = f'cases/cirr/{model_name2}.json'
    with open(casedata_path_1) as f:
        casedata_ls_1 = json.loads(f.read())
    with open(casedata_path_2) as f:
        casedata_ls_2 = json.loads(f.read())
    case_path = f'cases/cirr/base-full'
    os.makedirs(case_path, exist_ok=True)
    cnt = 0
    for i in tqdm(range(len(casedata_ls_1))):
        triplet = cirDataset.triplets[i]
        refer_path = triplet['reference']
        target_path = triplet['target']
        caption = f"#1:{triplet['captions'][0]}"
        output_path = os.path.join(case_path, f"{i}")
        k_1 = casedata_ls_1[i]["k"]
        k_2 = casedata_ls_2[i]["k"]
        if k_1 >= 5 and k_2 == 0:
            cnt += 1
            os.makedirs(output_path, exist_ok=True)
            shutil.copy(refer_path, os.path.join(output_path, "refer.png"))
            shutil.copy(target_path, os.path.join(output_path, "target.png"))
            with open(os.path.join(output_path, "caption.txt"), 'w') as f:
                caption += f"\n\nrank1:{k_1} rank2:{k_2}"
                f.write(caption)
            top_k_names_1 = casedata_ls_1[i]["top_k_names"]
            top_k_names_2 = casedata_ls_2[i]["top_k_names"]
            for i, name in enumerate(top_k_names_1):
                image_path = os.path.join(cirDataset.image_path,
                                          cirDataset.name_to_relpath[name])
                shutil.copy(image_path, os.path.join(output_path, f"{model_name1}_{i}.png"))
            for i, name in enumerate(top_k_names_2):
                image_path = os.path.join(cirDataset.image_path,
                                          cirDataset.name_to_relpath[name])
                shutil.copy(image_path, os.path.join(output_path, f"{model_name2}_{i}.png"))
    print("case num:", cnt)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, help="should be either 'cirr' or 'fiq'")
    args = parser.parse_args()
    if args.dataset == 'fiq':
        data_path = 'fashionIQ_dataset'
    else:
        data_path = 'cirr_dataset'
    cirDataset = CIRDataset(args.dataset, 'val', 'relative', None, data_path)
    func = eval(f"get_{args.dataset}_case")
    func()
