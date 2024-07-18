prompts_reference = [
    'remove {0}',
    '{0} is removed',
]

prompts_target = [
    '{1}',
    'apply {1}',
    'add {1}',
    'if it is {1}',
    '{1} is the new option',
    'choose {1} instead',
    '{1} is the updated version',
    'use {1} from now on',
    '{1} is the new choice',
    'opt for {1}',
    '{1} is the updated option',
    '{1} is the new selection',
    '{1} is the new option available',
    '{1} is the updated choice',
    '{1} is introduced as the new option after'
]

prompts_both = [
    'I want an image of {1} instead of an image of {0}',
    'change {0} to {1}',
    'Replace the image of {0} with {1} in the output.',
    'Generate an output image where {1} is depicted instead of {0}.',
    'Transform the input image to show {1} instead of {0}.',
    'Obtain an image with {1} replacing {0} from the input.',
    'Produce an output image featuring {1} rather than {0}.',
    'Modify the input image to display {1} instead of {0}.',
    'Create a new image by substituting {0} for {1} in the original.',
    'Request an image of {1} as a replacement for {0}.',
    'replace {0} with {1}',
    'substitute {1} for {0}',
    'exchange {0} with {1}',
    'alter {0} to {1}',
    'convert {0} to {1}',
    'transform {0} into {1}',
    'swap {0} for {1}',
    'replace {0} with {1}',
    'remodel {0} into {1}',
    'redesign {0} as {1}',
    'update {0} to {1}',
    'revamp {0} into {1}',
    'substitute {1} for {0}',
    'modify {0} to become {1}',
    'turn {0} into {1}',
    'alter {0} to match {1}',
    'customize {0} to become {1}',
    'adapt {0} to fit {1}',
    'upgrade {0} to {1}',
    'change {0} to match {1}',
    'tweak {0} to become {1}',
    'amend {0} to fit {1}',
    '{0} is replaced with {1}',
    '{0} is removed and {1} is added',
    '{1} is introduced after {0} is removed',
    '{0} is removed and {1} takes its place',
    '{1} is added after {0} is removed',
    '{0} is removed and {1} is introduced',
    '{1} is added in place of {0}',
    '{1} is introduced after {0} is retired',
    '{1} is added as a replacement for {0}'
]

print("prompts_reference len:", len(prompts_reference))
print("prompts_target len:", len(prompts_target))
print("prompts_both len:", len(prompts_both))
