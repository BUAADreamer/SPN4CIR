import argparse
import json

import torch
from tqdm import tqdm

from data_process import get_fiq_it, get_cirr_it, get_cc_it

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria

from PIL import Image

import requests
from PIL import Image
from io import BytesIO
from transformers import TextStreamer


def load_image(image_file):
    if image_file.startswith('http') or image_file.startswith('https'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    return image


def generate_caption(args, prompt, image=None):
    image = load_image(args.image_file) if image is None else image
    image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'].half().cuda()
    inp = f"user: {prompt}"
    # first message
    if model.config.mm_use_im_start_end:
        inp = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + inp
    else:
        inp = DEFAULT_IMAGE_TOKEN + '\n' + inp
    conv.append_message(conv.roles[0], inp)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=image_tensor,
            do_sample=True,
            temperature=0.01,
            top_p=0.01,
            max_new_tokens=1024,
            use_cache=True,
            stopping_criteria=[stopping_criteria])

    outputs = tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip().replace('</s>', '')
    return outputs


def main(args, image=None):
    image = load_image(args.image_file) if image is None else image
    image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'].half().cuda()

    while True:
        try:
            inp = input(f"{roles[0]}: ")
        except EOFError:
            inp = ""
        if not inp:
            print("exit...")
            break

        print(f"{roles[1]}: ", end="")

        if image is not None:
            # first message
            if model.config.mm_use_im_start_end:
                inp = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + inp
            else:
                inp = DEFAULT_IMAGE_TOKEN + '\n' + inp
            conv.append_message(conv.roles[0], inp)
            image = None
        else:
            # later messages
            conv.append_message(conv.roles[0], inp)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
        streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor,
                do_sample=True,
                temperature=0.01,
                top_p=0.01,
                max_new_tokens=1024,
                streamer=streamer,
                use_cache=True,
                stopping_criteria=[stopping_criteria])

        outputs = tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()
        conv.messages[-1][-1] = outputs

        if args.debug:
            print("\n", {"prompt": prompt, "outputs": outputs}, "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="/root_path/cirdata/llava-v1-0719")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-file", type=str)
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--load-8bit", action="store_true")
    parser.add_argument("--load-4bit", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--dress_type", default='dress,shirt,toptee')
    parser.add_argument("--cir_data", default="fiq")
    parser.add_argument("--cc_id", type=int, default=0)
    parser.add_argument("--k", type=int, default=10)
    args = parser.parse_args()

    # 读入图片
    # img1 = '/root_path/fashionIQ_dataset/images/B0083I6W08.png'
    # img2 = '/root_path/fashionIQ_dataset/images/B00BPD4N5E.png'
    # img = concat_2_images(img1, img2, type=1, save=True)
    image_id = 'B00BPD4N5E'  # B0083I6W08 B00BPD4N5E
    args.image_file = 'fashionIQ_dataset/images/{}.png'.format(image_id)
    # Model
    disable_torch_init()

    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(args.model_path, args.model_base, model_name,
                                                                           args.load_8bit, args.load_4bit)

    if 'llama-2' in model_name.lower():
        conv_mode = "llava_llama_2"
    elif "v1" in model_name.lower():
        conv_mode = "llava_v1"
    elif "mpt" in model_name.lower():
        conv_mode = "mpt"
    else:
        conv_mode = "llava_v0"

    if args.conv_mode is not None and conv_mode != args.conv_mode:
        print(
            '[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}'.format(conv_mode,
                                                                                                              args.conv_mode,
                                                                                                              args.conv_mode))
    else:
        args.conv_mode = conv_mode

    conv = conv_templates[args.conv_mode].copy()
    if "mpt" in model_name.lower():
        roles = ('user', 'assistant')
    else:
        roles = conv.roles
    # main(args)
    if args.cir_data == 'fiq':
        type2itlist = get_fiq_it()
        dress_types = args.dress_type.split(',')
        all_it_list = []
        for dress_type in dress_types:
            it_list = type2itlist[dress_type]
            prompt = f'please briefly describe the {dress_type} in {args.k} words'
            for it in tqdm(it_list):
                args.image_file = it['image_path']
                it['caption'] = generate_caption(args, prompt)
                # print(it['caption'])
                conv = conv_templates[args.conv_mode].copy()
            all_it_list.extend(it_list)
        with open(f"mm_data/fiq/fashioniq_it_llava_{args.k}.json", 'w', encoding='utf-8') as f:
            f.write(json.dumps(all_it_list, ensure_ascii=False))
    elif args.cir_data == 'cirr':
        it_list = get_cirr_it()
        prompt = f'please briefly describe the image in {args.k} words'
        for it in tqdm(it_list):
            args.image_file = it['image_path']
            it['caption'] = generate_caption(args, prompt)
            conv = conv_templates[args.conv_mode].copy()
        with open(f"mm_data/cirr/cirr_it_llava_{args.k}.json", 'w', encoding='utf-8') as f:
            f.write(json.dumps(it_list, ensure_ascii=False))
    elif args.cir_data == 'cc':
        it_list = get_cc_it(args.cc_id)
        prompt = f'please briefly describe the image in {args.k} words'
        for it in tqdm(it_list):
            args.image_file = it['image_path']
            it['caption'] = generate_caption(args, prompt)
            conv = conv_templates[args.conv_mode].copy()
        with open(f"mm_data/zs/cc_it_{args.cc_id}_llava_{args.k}.json", 'w', encoding='utf-8') as f:
            f.write(json.dumps(it_list, ensure_ascii=False))