########################################################################################################
# The RWKV Language Model - https://github.com/BlinkDL/RWKV-LM
########################################################################################################

import numpy as np

from transformers import PreTrainedTokenizerFast
tokenizer = PreTrainedTokenizerFast(tokenizer_file='20B_tokenizer.json')

input_file = 'train.txt'
output_file = 'train.npy'

TASK = 'tokenize' # tokenize verify

if TASK == 'tokenize':

    print(f'Tokenizing {input_file} (VERY slow. please wait)')

    data_raw = open(input_file, encoding="utf-8").read()
    print(f'Raw length = {len(data_raw)}')

    data_code = tokenizer.encode(data_raw)
    print(f'Tokenized length = {len(data_code)}')

    out = np.array(data_code, dtype='uint16')
    np.save(output_file, out, allow_pickle=False)

elif TASK == 'verify':

    test = np.load(output_file)
    print(test)
    print('\n\n')
    print(tokenizer.decode(test[:100]))
    print('\n\n')
    print(tokenizer.decode(test[-100:]))
