########################################################################################################
# The RWKV v2-RNN Language Model - https://github.com/BlinkDL/RWKV-LM
########################################################################################################

# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "3"

import logging
import datetime
import json
from src.model_train import GPT, GPTConfig
from src.trainer import Trainer, TrainerConfig
import torch
from torch.utils.data import Dataset
import numpy as np
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_tf32 = True

ctx_len = 768
n_layer = 24
n_embd = 1024
vocab_size = 50277

model_type = 'RWKV'
model_name = 'all-10803'

datafile = 'train.npy' # use 'prepare-data.py' to tokenize .txt into .npy

########################################################################################################

batch_size = 8 
# The batch_size must be divisible by B_GROUP_FORWARD and B_GROUP_BACKWARD in src/model_train.py.
# you can reduce B_GROUP_FORWARD and B_GROUP_BACKWARD to make it easier to find a good batch_size for your GPU.
# just remember B_GROUP_FORWARD=8 and B_GROUP_BACKWARD=2 is the fastest.

lr_init = 1e-5
lr_final = 1e-5

n_epoch = 10000 # the mini-epoch is very short and of fixed length (ctx_len * epoch_length_fixed tokens)
epoch_length_fixed = 10000

epoch_save_frequency = 5 # 0 = never, 1 = every mini-epoch, 2 = every two mini-epochs, etc.
epoch_save_path = 'trained-'

########################################################################################################

grad_norm_clip = 1.0
warmup_tokens = 0

betas = (0.9, 0.99)
eps = 1e-8

num_workers = 0

########################################################################################################
# Load data
########################################################################################################

class Dataset(Dataset):
    def __init__(self, data, vocab_size, ctx_len, epoch_length_fixed):
        data_size, vocab_size = len(data), vocab_size
        print('data has %d tokens, %d unique.' % (data_size, vocab_size))
        self.ctx_len = ctx_len
        self.epoch_length_fixed = epoch_length_fixed
        self.vocab_size = vocab_size
        self.data = data

    def __len__(self):
        return self.epoch_length_fixed

    def __getitem__(self, idx):
        # cheat: pick a random spot in dataset
        i = np.random.randint(0, len(self.data) - (self.ctx_len + 1))
        dix = self.data[i:i+self.ctx_len+1]
        x = torch.tensor(dix[:-1], dtype=torch.long,
                         device=torch.device('cuda'))
        y = torch.tensor(dix[1:], dtype=torch.long,
                         device=torch.device('cuda'))
        return x, y

print('loading data... ' + datafile)
train_dataset = Dataset(np.load(datafile).astype('int'), vocab_size, ctx_len, epoch_length_fixed)

########################################################################################################
# Train model
########################################################################################################

np.set_printoptions(precision=4, suppress=True, linewidth=200)
logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO,)

if __name__ == '__main__':

    model = GPT(GPTConfig(train_dataset.vocab_size, train_dataset.ctx_len, model_type=model_type,
                          n_layer=n_layer, n_embd=n_embd)).cuda()

    print('loading ' + model_name)
    m2 = torch.load(model_name + '.pth')
    model.load_state_dict(m2)

    print('model', model_type, 'epoch', n_epoch, 'batchsz', batch_size, 'betas',
          betas, 'eps', eps, 'ctx', ctx_len, 'layer', n_layer, 'embd', n_embd, )
    tconf = TrainerConfig(model_type=model_type, max_epochs=n_epoch, batch_size=batch_size,
                          learning_rate=lr_init, lr_decay=True, lr_final=lr_final, betas=betas, eps=eps, grad_norm_clip=grad_norm_clip,
                          warmup_tokens=warmup_tokens, final_tokens=n_epoch*len(train_dataset)*ctx_len, num_workers=num_workers, epoch_save_frequency=epoch_save_frequency, epoch_save_path=epoch_save_path)
    trainer = Trainer(model, train_dataset, None, tconf)

    trainer.train()

    torch.save(model.state_dict(), 'trained-' + str(n_epoch) + '-' + trainer.get_run_name() +
               '-' + datetime.datetime.today().strftime('%Y-%m-%d-%H-%M-%S') + '.pth')
