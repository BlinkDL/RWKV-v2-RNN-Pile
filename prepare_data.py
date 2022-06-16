########################################################################################################
# The RWKV Language Model - https://github.com/BlinkDL/RWKV-LM
########################################################################################################

import numpy as np
from torch import le

import os

from transformers import PreTrainedTokenizerFast
from threading import Thread
tokenizer = PreTrainedTokenizerFast(tokenizer_file='20B_tokenizer.json')

input_dir = './training/'
chunksDir = "./training_chunks/"

output_file = 'train.npy'

threadCount = 1 # number of threads to use -- unfortunately, 1 seems to be the fastest on my setup. Maybe it's bottlenecked by the SSD, or the CPU? none of these threads seem to be maxing usage, so give it a shot.
reuse_old_chunks = True
chunkSize = 1000000
keep_buffer = True # is supposed to buffer load chunks from the SSD to memory, but I'm not quite sure it's properly implemented yet. I think I'm facing a bottleneck, so I can't really test if this helps or not.

TASK = 'tokenize' # tokenize verify

os.makedirs(input_dir, exist_ok=True)
os.makedirs(chunksDir, exist_ok=True)
if reuse_old_chunks:
    # check if we have any chunks, if not, toggle the flag back to false
    chunks = os.listdir(chunksDir)
    if len(chunks) == 0:
        reuse_old_chunks = False

data_code = []
if TASK == 'tokenize':
    rawChunks = []
    def parseChunk(chunk,buf,i):
        global rawChunks
        global data_code
        if(not buf):
            chunk = open(chunksDir + chunk).read()
        print(f'{i}:Tokenizing chunk {len(rawChunks)}/{len(os.listdir(chunksDir))}')
        data_code = data_code + tokenizer.encode(chunk)
        print(len(data_code))
        print(f'{i}:Tokenized length = {len(data_code)}')
    if reuse_old_chunks:
        print("Loading old chunk data...")
    else:
        print("Loading raw text data for chunking(MEMORY INTENSIVE)...")
        files = os.listdir(input_dir)
        files = [f for f in files if f.endswith('.txt')]
        
        # clean up old chunks
        for chunk in os.listdir(chunksDir):
            os.remove(chunksDir + chunk)    

        for input_file in files:
            print(f'Loading {input_file}...')
            chunk = open(input_dir + input_file).read()
            if len(rawChunks) == 0:
                print("First chunk")
                rawChunks.append(chunk)
            if (len(rawChunks[-1]) + len(chunk)) < chunkSize:
                rawChunks[-1] = f"{rawChunks[-1]}\n{chunk}"
                # print(len(rawChunks[-1]))
            else:
                print(f"{len(rawChunks)} chunks")
                rawChunks.append(chunk)
                with open(chunksDir + str(len(rawChunks)) + '.txt', 'w') as f:
                    f.write(chunk)
    rawChunks = os.listdir(chunksDir) # clearing the actual chunks from ram and instead rereads from disk to save on memory usage during tokenization -- TODO: make this more efficient and earlier in the process so the memory usage doesn't spike during text chunking
    print(f'Total chunks = {len(rawChunks)}')
    buffer = []
    if keep_buffer:
        bufferSize = threadCount * 2
        while len(buffer) < bufferSize:
            raw_chunk = rawChunks.pop()
            chunk_data = open(chunksDir + raw_chunk).read()
            buffer.append(chunk_data)

    while len(rawChunks) > 0:
        threads = []
        for i in range(min(len(rawChunks), threadCount)):
            if len(rawChunks) == 0:
                break
            print(f'Starting tokenizing thread {i}')
            if keep_buffer:
                BUF = buffer.pop()
                thread = Thread(target=parseChunk, args=(BUF,True,i))
            else:
                CHNK = rawChunks.pop()
                thread = Thread(target=parseChunk, args=(CHNK,False,i))
            thread.start()
            threads.append(thread)
        for thread in threads:
            thread.join()
            if keep_buffer:
                raw_chunk = rawChunks.pop()
                chunk_data = open(chunksDir + raw_chunk).read()
                buffer.append(chunk_data)


elif TASK == 'verify':

    test = np.load(output_file)
    print(test)
    print('\n\n')
    print(tokenizer.decode(test[:100]))
    print('\n\n')
    print(tokenizer.decode(test[-100:]))
        
out = np.array(data_code, dtype='uint16')
np.save(output_file, out, allow_pickle=False)
