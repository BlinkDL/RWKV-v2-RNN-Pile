########################################################################################################
# The RWKV Language Model - https://github.com/BlinkDL/RWKV-LM
########################################################################################################

from torch.utils.cpp_extension import load
import math
import numpy as np
import logging
import torch
import torch.nn as nn
from torch.nn import functional as F
logger = logging.getLogger(__name__)

print('\n\n\n' + '*' * 120)
print("WARNING: The model_train.py here does not have RWKV_Init(), so it's only suitable for fine-tuning a trained model !!!")
print('*' * 120 + '\n\n\n')

########################################################################################################
# CUDA Kernel
########################################################################################################

T_MAX = 1024          # increase this if your ctx_len > 1024
B_GROUP_FORWARD = 8   # set to 8 for best performance
B_GROUP_BACKWARD = 2  # set to 2 for best performance

timex_cuda = load(name="timex", sources=["cuda/timex_op.cpp", "cuda/timex_cuda.cu"],
                  verbose=True, extra_cuda_cflags=['--use_fast_math', '--extra-device-vectorization', f'-DTmax={T_MAX}', f'-DBF={B_GROUP_FORWARD}', f'-DBB={B_GROUP_BACKWARD}'])


class TimeX(torch.autograd.Function):
    @staticmethod
    def forward(ctx, w, k, B, C, T, eps):
        ctx.B = B
        ctx.C = C
        ctx.T = T
        assert ctx.T % 4 == 0 and ctx.T <= T_MAX and ctx.B % B_GROUP_FORWARD == 0 and ctx.B % B_GROUP_BACKWARD == 0
        w = w.contiguous()
        k = k.contiguous()
        ctx.save_for_backward(w, k)
        wk = torch.empty((B, C, T), device='cuda',
                         memory_format=torch.contiguous_format)
        timex_cuda.forward(w, k, wk, eps, B, C, T)
        return wk

    @staticmethod
    def backward(ctx, gwk):
        assert ctx.T % 4 == 0 and ctx.T <= T_MAX and ctx.B % B_GROUP_FORWARD == 0 and ctx.B % B_GROUP_BACKWARD == 0
        w, k = ctx.saved_tensors
        gw = torch.empty((ctx.B, ctx.C, ctx.T), device='cuda',
                         memory_format=torch.contiguous_format)
        gk = torch.empty((ctx.B, ctx.C, ctx.T), device='cuda',
                         memory_format=torch.contiguous_format)
        timex_cuda.backward(w, k, gwk.contiguous(), gw,
                            gk, ctx.B, ctx.C, ctx.T)
        return (gw.sum(dim=0), gk, None, None, None, None)

########################################################################################################
# RWKV: RWKV Time-mix + RWKV Channel-mix
########################################################################################################

RWKV_K_CLAMP = 60  # e^60 = 1e26
RWKV_K_EPS = 1e-9
RWKV_HEAD_QK_DIM = 256

class RWKV_TimeMix(nn.Module):
    def __init__(self, config, layer_id):
        super().__init__()
        self.layer_id = layer_id
        self.ctx_len = config.ctx_len
        self.n_embd = config.n_embd

        attn_sz = config.n_embd

        self.time_decay = nn.Parameter(torch.ones(attn_sz, 1))
        self.time_curve = torch.tensor(
            [-(config.ctx_len - 2 - i) for i in range(config.ctx_len-1)]).unsqueeze(0)
        self.time_curve = self.time_curve.to('cuda')
        self.time_first = nn.Parameter(torch.ones(attn_sz, 1) * math.log(0.3))
        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
        with torch.no_grad():
            ww = torch.ones(1, 1, config.n_embd)
            for i in range(config.n_embd // 2):
                ww[0, 0, i] = 0
        self.time_mix_k = nn.Parameter(ww)
        self.time_mix_v = nn.Parameter(ww)
        self.time_mix_r = nn.Parameter(ww)

        self.key = nn.Linear(config.n_embd, attn_sz, bias=False)
        self.value = nn.Linear(config.n_embd, attn_sz, bias=False)
        self.receptance = nn.Linear(config.n_embd, attn_sz, bias=False)

        self.output = nn.Linear(attn_sz, config.n_embd, bias=False)

        self.key.scale_init = 0
        self.receptance.scale_init = 0
        self.output.scale_init = 0

    def forward(self, x):
        B, T, C = x.size()
        assert T == self.ctx_len

        xx = self.time_shift(x)
        xk = x * self.time_mix_k + xx * (1 - self.time_mix_k)
        xv = x * self.time_mix_v + xx * (1 - self.time_mix_v)
        xr = x * self.time_mix_r + xx * (1 - self.time_mix_r)

        k = self.key(xk).transpose(-1, -2)
        v = self.value(xv).transpose(-1, -2)
        r = self.receptance(xr)

        # RWKV_K_CLAMP can be removed if the CUDA kernel substracts the correct k_max for each k (I will do this later)
        k = torch.clamp(k, max=RWKV_K_CLAMP)
        k = torch.exp(k)
        kv = k * v

        self.time_w = torch.cat(
            [torch.exp(self.time_decay) * self.time_curve, self.time_first], dim=-1)
        w = torch.exp(self.time_w)

        wkv = TimeX.apply(w, kv, B, C, T, 0)
        # RWKV_K_EPS can be removed if the CUDA kernel sets 0/0 = 0 (I will do this later)
        wk = TimeX.apply(w, k, B, C, T, RWKV_K_EPS)

        rwkv = torch.sigmoid(r) * (wkv / wk).transpose(-1, -2)
        rwkv = self.output(rwkv)
        return rwkv


class RWKV_ChannelMix(nn.Module):
    def __init__(self, config, layer_id):
        super().__init__()
        self.layer_id = layer_id

        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))

        with torch.no_grad():  # init to "shift half of the channels"
            x = torch.ones(1, 1, config.n_embd)
            for i in range(config.n_embd // 2):
                x[0, 0, i] = 0
        self.time_mix_k = nn.Parameter(x)
        self.time_mix_r = nn.Parameter(x)

        hidden_sz = 4 * config.n_embd
        self.key = nn.Linear(config.n_embd, hidden_sz, bias=False)
        self.receptance = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.value = nn.Linear(hidden_sz, config.n_embd, bias=False)

        self.value.scale_init = 0
        self.receptance.scale_init = 0

    def forward(self, x):
        xx = self.time_shift(x)
        xk = x * self.time_mix_k + xx * (1 - self.time_mix_k)
        xr = x * self.time_mix_r + xx * (1 - self.time_mix_r)

        k = self.key(xk)
        k = torch.square(torch.relu(k))
        kv = self.value(k)
        
        rkv = torch.sigmoid(self.receptance(xr)) * kv
        return rkv

########################################################################################################
# The GPT Model with our blocks
########################################################################################################


class GPTConfig:
    def __init__(self, vocab_size, ctx_len, **kwargs):
        self.vocab_size = vocab_size
        self.ctx_len = ctx_len
        for k, v in kwargs.items():
            setattr(self, k, v)


class Block(nn.Module):
    def __init__(self, config, layer_id):
        super().__init__()
        self.config = config
        self.layer_id = layer_id

        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
        if self.layer_id == 0:
            self.ln0 = nn.LayerNorm(config.n_embd)

        self.att = RWKV_TimeMix(config, layer_id)
        self.ffn = RWKV_ChannelMix(config, layer_id)

    def forward(self, x):
        
        if self.layer_id == 0:
            x = self.ln0(x)
        x = x + self.att(self.ln1(x))
        x = x + self.ffn(self.ln2(x))

        return x


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.step = 0
        self.config = config

        self.emb = nn.Embedding(config.vocab_size, config.n_embd)

        self.blocks = nn.Sequential(*[Block(config, i)
                                    for i in range(config.n_layer)])

        self.ln_out = nn.LayerNorm(config.n_embd)
        self.head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        self.ctx_len = config.ctx_len

        # RWKV_Init(self, config)

        logger.info("number of parameters: %e", sum(p.numel()
                    for p in self.parameters()))

    def get_ctx_len(self):
        return self.ctx_len

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear)):
            module.weight.data.normal_(mean=0.0, std=0.01)
        if isinstance(module, (nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=1e-5)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def configure_optimizers(self, train_config):
        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()

        for mn, m in self.named_modules():  # here we disable weight_decay
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn  # full param name
                no_decay.add(fpn)

        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(
            inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
        assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
            % (str(param_dict.keys() - union_params), )

        optim_groups = [
            {"params": [param_dict[pn]
                        for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]

        optimizer = torch.optim.Adam(
            optim_groups, lr=train_config.learning_rate, betas=train_config.betas, eps=train_config.eps)

        return optimizer

    def forward(self, idx, targets=None):
        self.step += 1
        B, T = idx.size()
        assert T <= self.ctx_len, "Cannot forward, because len(input) > model ctx_len."
        x = self.emb(idx)
        x = self.blocks(x)
        x = self.ln_out(x)
        x = self.head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(x.view(-1, x.size(-1)), targets.view(-1))

        return x, loss
