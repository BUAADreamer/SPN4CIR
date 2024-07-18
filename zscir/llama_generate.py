import json
# import random
#
# caption_triplets = []
# with open("mm_data/human-written-prompts.json") as f:
#     data_ls = json.loads(f.read())
#     for data in data_ls:
#         caption_triplets.append(
#             {
#                 "caption1": data['input'],
#                 "caption2": data['output'],
#                 "mod": data['edit']
#             }
#         )
# caption_triplets = random.sample(caption_triplets, 10)
# context = ""
# for caption_triplet in caption_triplets:
#     context += f"first caption:{caption_triplet['caption1']}\n" \
#                f"second caption:{caption_triplet['caption2']}\n" \
#                f"modified text:{caption_triplet['mod']}\n"
from prompt import prompt_templates, get_prompt

import argparse

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

model: AutoModelForCausalLM = AutoModelForCausalLM.from_pretrained("/root_path/LLM/llama2/llama2-7b-chat",
                                                                   device_map="auto",
                                                                   torch_dtype=torch.float16)
tokenizer = AutoTokenizer.from_pretrained("/root_path/LLM/llama2/llama2-7b-chat")
tokenizer.use_default_system_prompt = False
pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    torch_dtype=torch.float16,
    device_map="auto",
)
print("model load successfully")
gpu_mem = torch.cuda.get_device_properties(0).total_memory
available_mem = gpu_mem - torch.cuda.memory_allocated(0)
print('Total memory:', gpu_mem, 'Available memory:', available_mem)


def post_process(output):
    res = output.strip()
    if ":" in output:
        res = res.split(":")[-1]
    res = res.split("\n")[0]
    return res


def generate(prompt, max_new_tokens=25, post=True):
    sequences = pipeline(
        prompt,
        do_sample=True,
        top_k=10,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
        max_new_tokens=max_new_tokens,
    )
    N = len(prompt)
    res = sequences[0]['generated_text'][N:]
    # print(res)
    if post:
        res = post_process(res)
    return res


def generate_modified_text(caption1="The toptee is a black, sleeveless, heart-shaped top.",
                           caption2="The toptee is a green, long-sleeved, and loose-fitting shirt.",
                           data='cirr',
                           mod_type=0,
                           llm_type=1):
    prompt_template = prompt_templates[data]
    if mod_type == 0:
        if caption1[-1] == '.':
            caption1 = caption1[:-1]
        if caption2[-1] == '.':
            caption2 = caption2[:-1]
        old_text = '{1} instead of {0}'.format(caption1, caption2)
    else:
        old_text = caption2
    if llm_type == 1:
        prompt = get_prompt(old_text, data)
    elif llm_type == 2:
        prompt = prompt_template.format(old_text)
    else:
        prompt = prompt_template.format(old_text)
    max_new_tokens = 25 if data == 'fiq' else 50
    text = generate(prompt, max_new_tokens=max_new_tokens)
    print(text, flush=True)
    return text


def get_triplets():
    generate_modified_text(args.data)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default='fiq')
    parser.add_argument("--seed", default=42, type=int)
    args = parser.parse_args()
    get_triplets()
