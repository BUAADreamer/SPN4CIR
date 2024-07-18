cirr_prompt = """Rewrite the sentence to maintain the original meaning while reducing grammatical errors and increasing the variety of expression
Remember only output the new sentence without other additional words.
sentence:{0}
new sentence:
"""

fiq_prompt = """Rewrite the sentence to maintain the original meaning while reducing grammatical errors and increasing the variety of expression
Remember only output the new sentence without other additional words.
sentence:{0}
new sentence:
"""

prompt_templates = {
    "fiq": fiq_prompt,
    "cirr": cirr_prompt,
}

fiq_caption_pairs = [
    {
        "source": "The dress is a sleeveless, black, fitted, and stylish dress",
        "target": "is solid black with no sleeves",
    },
    {
        "source": "Red, flowy, short, sequined, and elegant.",
        "target": "is red",
    },
    {
        "source": "Obama Mama shirt, black color.",
        "target": "has the words Obama Mama on front",
    },
]

system_prompt = """You are a researcher tasked with rewrite source sentence to mimic target sentence while trying to keep the original meaning.  
Please ensure that your responses are close to the style of the target sentences in the examples.
If you encounter harmful words, please change them to harmless content.
Remember only output the new sentence without other additional words.
Output answer in one string
"""


def get_fiq_prompt(caption):
    prompt = f"""<s>[INST] <<SYS>>\n{system_prompt}<</SYS>>\n\n"""
    fiq_caption_pairs = [
        {
            "source": "The dress is a sleeveless, black, fitted, and stylish dress",
            "target": "is solid black with no sleeves",
        },
        {
            "source": "Red, flowy, short, sequined, and elegant.",
            "target": "is red and flowy",
        },
        {
            "source": "Obama Mama shirt, black color.",
            "target": "has the words Obama Mama on front",
        },
        {
            "source": "Striped, black and white, sleeveless, fitted, and stylish.",
            "target": "has sleeveless black and white stripes",
        },
        {
            "source": "Colorful striped top with a v-neck.",
            "target": "Has stripes.",
        }
    ]
    for i, caption_pair in enumerate(fiq_caption_pairs):
        if i == 0:
            prompt += f"source caption: {caption_pair['source']}\ntarget caption: [/INST]{caption_pair['target']} </s>"
        else:
            prompt += f"<s>[INST]source caption: {caption_pair['source']}\ntarget caption: [/INST]{caption_pair['target']} </s>"
    prompt += f"<s>[INST]source caption: {caption}\n target caption: [/INST]"
    return prompt


def get_cirr_prompt(caption):
    prompt = f"""<s>[INST] <<SYS>>\n{system_prompt}<</SYS>>\n\n"""
    caption_pairs = [
        {
            "source": "A large, brown dog with a black nose is sitting on the grass, looking up instead of A cute baby panda is being held by a person in a zoo",
            "target": "Dog in grass instead of a panda.",
        },
        {
            "source": "A street with several blue buildings, including churches, and a park with trees and bushes instead of A large, old stone church with a tower and a wall, surrounded by a grassy field and a dirt road",
            "target": "instead of an old fortress with a rampart, an Orthodox church with a courtyard.",
        },
        {
            "source": "A colorful parrot is standing on a perch in a cage instead of Two parrots are sitting on a branch, sharing a piece of fruit",
            "target": "Remove one of the parrots.",
        },
        {
            "source": "Two colorful parrots are kissing on a branch instead of A colorful parrot is perched on a tree branch, looking at the camera.",
            "target": "two birds, facing each other.",
        },
        {
            "source": "A monkey is standing on a grassy field, looking at the camera instead of A group of monkeys is sitting on the ground, with some of them touching each other.",
            "target": "I want the pic to show just one monkey.",
        }
    ]
    for i, caption_pair in enumerate(caption_pairs):
        if i == 0:
            prompt += f"source caption: {caption_pair['source']}\ntarget caption: [/INST]{caption_pair['target']} </s>"
        else:
            prompt += f"<s>[INST]source caption: {caption_pair['source']}\ntarget caption: [/INST]{caption_pair['target']} </s>"
    prompt += f"<s>[INST]source caption: {caption}\n target caption: [/INST]"
    return prompt


test_prompt = get_cirr_prompt(
    "Two women, one with large breasts, pose for a selfie on a beach instead of A white dining table with chairs and a staircase in a room with artwork and a computer")


def get_prompt(caption, data='fiq'):
    if data == 'fiq':
        return get_fiq_prompt(caption)
    else:
        return get_cirr_prompt(caption)
