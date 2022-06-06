# RWKV-v2-RNN-Pile

RWKV-v2-RNN trained on the full Pile (no dev/val/test split).

Training log: https://wandb.ai/blinkdl/RWKV-v2-RNN-Pile

See https://github.com/BlinkDL/RWKV-LM for details.

You can use the "GPT" mode to quickly build the hidden state for the "RNN" mode. (I am not doing it in the run.py here so the initial generation is slower than usual).

## Fine-tuning

Use prepare_data.py to tokenize your .txt into .npy, then run finetune.py to fine-tune the Pile model.

Reduce batch_sz if you see CUDA OOM (and change B_GROUP_FORWARD and B_GROUP_BACKWARD in src/model_train.py to make sure they can divide batch_sz).

===================================================

Model 20220605-7663 (see Releases):

This is a preview of a L24-D1024 RWKV-v2-RNN trained on the Pile for 235B tokens.

It is NOT indicative of the final performance (which requires 300B tokens).

Performance of the preview model:

LAMBADA ppl 15.3 acc 42.62% (computed using https://github.com/EleutherAI/lm-evaluation-harness)

The Pile loss 2.361

===================================================

Model 20220524-4006 (see Releases):

This is a preview of a L24-D1024 RWKV-v2-RNN trained on the Pile for only 123B tokens.

It is NOT indicative of the final performance (which requires 300B tokens).

Performance of the preview model:

LAMBADA ppl 15.88 acc 42.36% (computed using https://github.com/EleutherAI/lm-evaluation-harness)

The Pile loss 2.383

===================================================

Model 20220501-6548 (see Releases):

This is a preview of a L12-D768 RWKV-v2-RNN trained on the Pile for only 50B tokens.

It is NOT indicative of the final performance (which requires 300B tokens).

Performance of the preview model:

LAMBADA ppl 52.45 acc 26.66% (computed using https://github.com/EleutherAI/lm-evaluation-harness)

The Pile loss 2.728
