import os

os.environ['HF_ENDPOINT'] = "https://hf-mirror.com"
import argparse
import json

from PIL import Image
from tqdm import tqdm

from data_process import get_fiq_it, get_cirr_it, get_cc_it
import torch
from lavis.models import load_model_and_preprocess

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def generate_caption(image_path, prompt):
    # preprocess the image
    # vis_processors stores image transforms for "train" and "eval" (validation / testing / inference)
    raw_image = Image.open(image_path).convert('RGB')
    image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)
    # generate caption
    res = model.generate({"image": image, "prompt": prompt})[0]
    # ['a large fountain spewing water into the air']
    return res


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="blip", choices=['blip', 'blip2'])
    parser.add_argument("--dress_type", default='dress,shirt,toptee')
    parser.add_argument("--cir_data", default="fiq")
    parser.add_argument("--cc_id", type=int, default=0)
    args = parser.parse_args()
    if args.model_name == 'blip':
        model, vis_processors, _ = load_model_and_preprocess(name="blip_caption", model_type="base_coco", is_eval=True,
                                                             device="cuda")
    else:
        model, vis_processors, _ = load_model_and_preprocess(
            name="blip2_opt", model_type="pretrain_opt2.7b", is_eval=True, device="cuda"
        )
    if args.cir_data == 'fiq':
        type2itlist = get_fiq_it()
        dress_types = args.dress_type.split(',')
        all_it_list = []
        for dress_type in dress_types:
            it_list = type2itlist[dress_type]
            prompt = f'please briefly describe the {dress_type} in 5 words'
            for it in tqdm(it_list):
                image_path = it['image_path']
                it['caption'] = generate_caption(image_path, prompt)
                # print(it['caption'])
            # with open(f"/root_path/cirdata/fashioniq_{dress_type}_it.json", 'w', encoding='utf-8') as f:
            #     f.write(json.dumps(it_list, ensure_ascii=False))
            all_it_list.extend(it_list)
        with open(f"mm_data/fiq/fashioniq_it_{args.model_name}.json", 'w', encoding='utf-8') as f:
            f.write(json.dumps(all_it_list, ensure_ascii=False))
    elif args.cir_data == 'cirr':
        it_list = get_cirr_it()
        prompt = f'please briefly describe the image in 10 words'
        for it in tqdm(it_list):
            image_path = it['image_path']
            it['caption'] = generate_caption(image_path, prompt)
        with open(f"mm_data/cirr/cirr_it_{args.model_name}.json", 'w', encoding='utf-8') as f:
            f.write(json.dumps(it_list, ensure_ascii=False))
    elif args.cir_data == 'cc':
        it_list = get_cc_it(args.cc_id)
        prompt = f'please briefly describe the image in 10 words'
        for it in tqdm(it_list):
            image_path = it['image_path']
            it['caption'] = generate_caption(image_path, prompt)
        with open(f"mm_data/zs/cc_it_{args.cc_id}_{args.model_name}.json", 'w', encoding='utf-8') as f:
            f.write(json.dumps(it_list, ensure_ascii=False))
