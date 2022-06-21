########################################################################################################
# The RWKV v2-RNN Language Model - https://github.com/BlinkDL/RWKV-LM
########################################################################################################

# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "4"

# this is for verifying the results of different models and make sure they agree with each other

import numpy as np
np.set_printoptions(precision=4, suppress=True, linewidth=200)

import torch
from src.model import RWKV_RNN, RWKV_GPT
from src.model_train import GPT, GPTConfig

from transformers import PreTrainedTokenizerFast
VOCAB_NAME = '20B_tokenizer.json'
tokenizer = PreTrainedTokenizerFast(tokenizer_file=VOCAB_NAME)

ctx_len = 768
n_layer = 24
n_embd = 1024
vocab_size = 50277
model_type = 'RWKV'

model_name = 'all-10803'

########################################################################################################

model_train = GPT(GPTConfig(vocab_size, ctx_len, model_type=model_type, n_layer=n_layer, n_embd=n_embd)).cuda()
print('loading ' + model_name)
m2 = torch.load(model_name + '.pth')
model_train.load_state_dict(m2)

model_rnn = RWKV_RNN(model_name)
model_gpt = RWKV_GPT(model_name).cuda()

########################################################################################################

context = '\nIn a shocking finding, scientist discovered a herd of dragons living in a remote, previously unexplored valley, in Tibet. Even more surprising to the researchers was the fact that the dragons spoke perfect Chinese.'
ctx = tokenizer.encode(context)
print(f'input len {len(ctx)} data {ctx}')

print('\nRWKV-GPT output')
out = model_gpt.forward(torch.tensor(ctx).unsqueeze(0).cuda())[0].detach().cpu().numpy()
print(out)

print('\nRWKV-RNN output')
model_rnn.clear()
src_len = len(ctx)
for i in range(src_len):
    x = ctx[:i+1]
    out = model_rnn.run(x)
    if i < 3 or i >= src_len - 3:
        print(np.array(out))
    if i == 2:
        print('...')

print('\nRWKV-train output')
ctx += [0] * (ctx_len - src_len) # pad to ctx_len
ctx = [ctx] * 8 # batch 8
out = model_train.forward(torch.tensor(ctx).cuda())[0][0][:src_len].detach().cpu().numpy()
print(out, '\n')
