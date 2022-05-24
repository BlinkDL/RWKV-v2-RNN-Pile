# RWKV-v2-RNN-Pile

RWKV-v2-RNN trained on the full Pile (no dev/val/test split).

Training log: https://wandb.ai/blinkdl/RWKV-v2-RNN-Pile

See https://github.com/BlinkDL/RWKV-LM for details.

You can use the "GPT" mode to quickly build the hidden state for the "RNN" mode. (I am not doing it in the run.py here so the initial generation is slower than usual).

===================================================

Model 20220524-4006 (see Releases):

This is a preview of a L24-D1024 RWKV-v2-RNN trained on the Pile for only 123B tokens.

It is NOT indicative of the final performance (which requires 300B tokens).

Performance of the preview model:

LAMBADA ppl 15.88 acc 42.36% (computed using https://github.com/EleutherAI/lm-evaluation-harness)

The Pile loss 2.383

===================================================

Model 20220515-1853 (see Releases):

This is a preview of a L24-D1024 RWKV-v2-RNN trained on the Pile for only 57B tokens.

It is NOT indicative of the final performance (which requires 300B tokens).

Performance of the preview model:

LAMBADA ppl 18.63 acc 39.61% (computed using https://github.com/EleutherAI/lm-evaluation-harness)

The Pile loss 2.417

===================================================

Model 20220501-6548 (see Releases):

This is a preview of a L12-D768 RWKV-v2-RNN trained on the Pile for only 50B tokens.

It is NOT indicative of the final performance (which requires 300B tokens).

Performance of the preview model:

LAMBADA ppl 52.45 acc 26.66% (computed using https://github.com/EleutherAI/lm-evaluation-harness)

The Pile loss 2.728
