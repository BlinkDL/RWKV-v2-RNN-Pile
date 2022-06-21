import numpy as np
import math, os, datetime
os.environ["CUDA_VISIBLE_DEVICES"] = '7' # CHANGE ME!
import torch
from torch.nn import functional as F

RWKV_PAD = [] # USE THIS TO VERIFY THE MATCH WITH MY 430M RESULTS
# RWKV_PAD = [0] # <|endoftext|> USE THIS FOR BETTER RESULTS
# RWKV_PAD = [187] # \n
# RWKV_PAD = [187, 187] # \n\n

RUN_TABLE = [492] # part of model file name

eval_tasks=['lambada']
# eval_tasks=['hellaswag']
# eval_tasks=['piqa'] # 'storycloze_2016'

TEST_MODEL = 'rwkv' # 'rwkv' 'neo'
USE_CUDA = True # True False
RUN_DEVICE = 'cuda' if USE_CUDA else 'cpu' # cpu cuda
# if TEST_MODEL != 'rwkv':
#     USE_CUDA = True

RWKV_SLOW_MODE = False # True False

from tqdm import tqdm
import torch
import torch.nn.functional as F

from lm_eval.base import CacheHook
from lm_eval.models.gpt2 import GPT2LM
from lm_eval import tasks, evaluator, utils

class TokenizerWrapper:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.eos_token_id = 0

    def encode(self, string: str, add_special_tokens=False):
        return self.tokenizer.encode(string).ids

    def decode(self, tokens):
        return self.tokenizer.decode(tokens)

class EvalHarnessAdapter(GPT2LM):
    def __init__(self):
        if TEST_MODEL == 'rwkv':
            self.tokenizer = TokenizerWrapper(tokenizers.Tokenizer.from_file('20B_tokenizer.json'))
        elif TEST_MODEL == 'neo':
            self.tokenizer = gpt.tokenizer 
        else:
            self.tokenizer = gpt2_tokenizer

    def greedy_until(self, requests):
        raise NotImplementedError()

    def _loglikelihood_tokens(self, requests, disable_tqdm=False):
        res = []
        sum_logit = 0
        nCorrect = 0

        for COUNTER in range(len(requests)):
            n = COUNTER

            raw_src = requests[n][0][0] + requests[n][0][1]

            src = requests[n][1] + requests[n][2]
            if TEST_MODEL == 'rwkv':
                raw_src = '\n' + raw_src
                src = RWKV_PAD + src

            sss = str(src)
            correct = True
            if sss in logitBuf:
                logit = logitBuf[sss]
                correct = correctBuf[sss]
            else:
                q_len = len(requests[n][1])
                if TEST_MODEL == 'rwkv':
                    q_len += len(RWKV_PAD)
                logit = 0
                
                with torch.no_grad():
                    if RWKV_SLOW_MODE:
                        rwkv_rnn.clear()
                        for i in range(1, len(src)):
                            x = src[:i]
                            out = rwkv_rnn.run(x)
                            if i >= q_len:
                                oo = torch.tensor(out)
                                sorted_probs, s_index = torch.sort(oo, descending=True)
                                pred = s_index[0].item()
                                if pred != src[i]:
                                    correct = False
                                # print(x, '=>', src[i], 'pred', pred)
                                logit += math.log(F.softmax(oo, dim=-1)[src[i]])
                    else:
                        if TEST_MODEL == 'neo':
                            gpt_inputs['input_ids'] = torch.tensor([src], device=RUN_DEVICE)
                            gpt_inputs['attention_mask'] = torch.ones_like(gpt_inputs['input_ids'], device=RUN_DEVICE)
                            outputs = gpt.model(**gpt_inputs).logits[0]
                        else:
                            outputs = rwkv_gpt.forward(torch.tensor([src], device=RUN_DEVICE))[0]

                        for i in range(q_len-1, len(src)-1):
                            oo = outputs[i]
                            dst = src[i+1]
                            logit += math.log(F.softmax(oo, dim=-1)[dst])
                            sorted_probs, s_index = torch.sort(oo, descending=True)
                            pred = s_index[0].item()
                            # print('pred', pred, 'dst', dst)
                            if pred != dst:
                                correct = False
                logitBuf[sss] = logit
                correctBuf[sss] = correct
            
            if correct:
                nCorrect += 1
            res += [(logit, correct)]
            sum_logit += logit
            mean = sum_logit / (COUNTER+1)
            acc = nCorrect / (COUNTER+1) * 100

            if n % 100 == 0:
                print(f'{n//100}/{len(requests)//100}', end = ' ', flush=True)
        return res

    @torch.no_grad()
    def run_eval(self, eval_tasks=None, num_fewshot=0, bootstrap_iters=2):
        results = evaluator.evaluate(
            lm=self,
            task_dict=tasks.get_task_dict(eval_tasks),
            provide_description=False,
            num_fewshot=num_fewshot,
            limit=None,
            bootstrap_iters=bootstrap_iters,
        )
        return results

RWKV_ID = ''
for RUN_NUM in RUN_TABLE:
    RWKV_ID = RUN_NUM
    logitBuf = {}
    correctBuf = {}

    RWKV_FILENAME = '/home/mchorse/all-' + str(RUN_NUM)

    if TEST_MODEL == 'neo':
        from transformers import pipeline
        gpt = pipeline('text-generation', model='EleutherAI/gpt-neo-125M') # , device=0
        # gpt = pipeline('text-generation', model='xhyi/PT_GPTNEO350_ATG') # , device=0
        # gpt = pipeline('text-generation', model='facebook/opt-125m') # , device=0
        # gpt = pipeline('text-generation', model='facebook/opt-350m') # , evice=0
        gpt.tokenizer.pad_token = gpt.tokenizer.eos_token
        gpt.model.eval()
        gpt_inputs = gpt.tokenizer("This is a test", return_tensors="pt")
        if USE_CUDA:
            gpt.model.cuda()
    else:
        if RWKV_SLOW_MODE:
            from src.model_run import RWKV_RNN
            rwkv_rnn = RWKV_RNN()
        else:
            from src.model_run import RWKV_GPT
            rwkv_gpt = RWKV_GPT(RWKV_FILENAME)
            if USE_CUDA:
                rwkv_gpt.cuda()
        import tokenizers

    try:
        print("Running evaluation harness...")
        adapter = EvalHarnessAdapter()
        results = adapter.run_eval(
            eval_tasks=eval_tasks,
            bootstrap_iters=10000,
        )
        print(results)
    except:
        pass