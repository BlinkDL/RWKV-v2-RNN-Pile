########################################################################################################
# The RWKV Language Model - https://github.com/BlinkDL/RWKV-LM
########################################################################################################

import numpy as np
from torch import le

from transformers import PreTrainedTokenizerFast
tokenizer = PreTrainedTokenizerFast(tokenizer_file='20B_tokenizer.json')

input_file = 'big.txt'
output_file = 'train2.npy'

TASK = 'tokenize' # tokenize verify

if TASK == 'tokenize':

    print(f'Tokenizing {input_file} (VERY slow. please wait)')

    data_raw = open(input_file).read()
    print(f'Raw length = {len(data_raw)}')

    # if len(data_raw) > 1000000 split into chunks -- Lets you tokenize a large file without running out of memory
    chunks = []
    if len(data_raw) > 1000000:
        print('Splitting into chunks')
        chunks = [data_raw[i:i+1000000] for i in range(0, len(data_raw), 1000000)]
    else:
        print('No need to split')
        chunks = [data_raw]
    

    data_code = []
    print(f'Total chunks = {len(chunks)}')
    for chunk in chunks:
        print(f'Tokenizing chunk {len(chunks)-chunks.index(chunk)}')
        data_code = data_code + tokenizer.encode(chunk)
        print(len(data_code))
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
