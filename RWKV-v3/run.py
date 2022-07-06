########################################################################################################
# The RWKV Language Model - https://github.com/BlinkDL/RWKV-LM
########################################################################################################

import numpy as np
import types
import copy
import torch
from torch.nn import functional as F

from src.model import RWKV_RNN

np.set_printoptions(precision=4, suppress=True, linewidth=200)

# ---> Edit src/model.py to set MODEL_NAME and CPU / CUDA mode <---

TEMPERATURE = 1.0
TOP_P = 0.7

DEBUG_DEBUG = False
LENGTH_OF_EACH = 222
NUM_TRIALS = 100

context = '\nIn a shocking finding,'

##############################################################################################################

model = RWKV_RNN()
print('\n--> Currently the first run takes a while if your prompt is long, as we are using RNN to process the prompt. Use GPT to build the hidden state for better speed. <--\n')

def sample_logits(out, temperature=1.0, top_p=None):
    probs = F.softmax(torch.tensor(out), dim=-1)
    sorted_probs, _ = torch.sort(probs, descending=True)

    cumulative_probs = torch.cumsum(sorted_probs, dim=-1).numpy()
    cutoff = float(sorted_probs[np.argmax(cumulative_probs > top_p)])
    probs[probs < cutoff] = 0

    if temperature != 1.0:
        probs = probs.pow(1.0 / temperature)

    return torch.multinomial(probs, num_samples=1)[0]


for TRIAL in range(1 if DEBUG_DEBUG else NUM_TRIALS):
    ctx = [model.tokenizer.encode(context)][0]
    src_len = len(ctx)
    print(context, end='')

    model.clear()
    if TRIAL == 0: # build the RNN hidden state?
        init_state = types.SimpleNamespace()
        for i in range(src_len if DEBUG_DEBUG else src_len):
            x = ctx[:i+1]
            if i == src_len - 1:
                init_state.out = model.run(x)
            else:
                model.run(x)
        model.save(init_state)
    else:
        model.load(init_state)

    if DEBUG_DEBUG:
        out = init_state.out
        print('\n', np.array(x), '==>', np.array(
            out), np.max(out), np.min(out))

    for i in range(src_len, src_len + (0 if DEBUG_DEBUG else LENGTH_OF_EACH)):
        x = ctx[:i+1]
        x = x[-model.ctx_len:]

        if i == src_len:
            out = copy.deepcopy(init_state.out) # load the RNN hidden state
        else:
            out = model.run(x) # run the RNN

        out[0] = -999999999  # disable <|endoftext|>

        char = sample_logits(out, temperature=TEMPERATURE, top_p=TOP_P)
        char = char.item()
        print(model.tokenizer.decode(char), end='', flush=True)

        ctx += [char]
    print('\n' + '-' * 70, end='')
