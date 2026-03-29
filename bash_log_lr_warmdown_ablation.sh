cd /v/ai/nobackup/xma/openai/parameter-golf

mkdir -p logs/logs_ablation_lr_warmdown_exp3

# LR / warmdown ablation around exp3.
# Goal: separate "lower effective LR" from "late-stage warmdown" under a fixed-step budget.
# Keep attention init and momentum fixed to the exp3 settings that looked strongest.

# Exp A: exp3 backbone, no warmdown
CUDA_VISIBLE_DEVICES=0 \
RUN_ID=ablate_a_no_warmdown \
DATA_PATH=./data/datasets/fineweb10B_sp1024/ \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
ITERATIONS=3000 \
MAX_WALLCLOCK_SECONDS=0 \
TIED_EMBED_LR=0.05 \
MATRIX_LR=0.04 \
SCALAR_LR=0.04 \
QK_GAIN_INIT=1.0 \
MUON_MOMENTUM=0.97 \
WARMDOWN_ITERS=0 \
python -u train_gpt_xma.py 2>&1 | tee logs/logs_ablation_lr_warmdown_exp3/ablate_a_no_warmdown.txt

# Exp B: exp3 backbone, short warmdown that should only affect the end of training
CUDA_VISIBLE_DEVICES=0 \
RUN_ID=ablate_b_short_warmdown \
DATA_PATH=./data/datasets/fineweb10B_sp1024/ \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
ITERATIONS=3000 \
MAX_WALLCLOCK_SECONDS=0 \
TIED_EMBED_LR=0.05 \
MATRIX_LR=0.04 \
SCALAR_LR=0.04 \
QK_GAIN_INIT=1.0 \
MUON_MOMENTUM=0.97 \
WARMDOWN_ITERS=200 \
python -u train_gpt_xma.py 2>&1 | tee logs/logs_ablation_lr_warmdown_exp3/ablate_b_short_warmdown.txt

# Exp C: exp3 backbone, medium warmdown
CUDA_VISIBLE_DEVICES=0 \
RUN_ID=ablate_c_mid_warmdown \
DATA_PATH=./data/datasets/fineweb10B_sp1024/ \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
ITERATIONS=3000 \
MAX_WALLCLOCK_SECONDS=0 \
TIED_EMBED_LR=0.05 \
MATRIX_LR=0.04 \
SCALAR_LR=0.04 \
QK_GAIN_INIT=1.0 \
MUON_MOMENTUM=0.97 \
WARMDOWN_ITERS=400 \
python -u train_gpt_xma.py 2>&1 | tee logs/logs_ablation_lr_warmdown_exp3/ablate_c_mid_warmdown.txt

# Exp D: lower LR, no warmdown
CUDA_VISIBLE_DEVICES=0 \
RUN_ID=ablate_d_low_lr_no_warmdown \
DATA_PATH=./data/datasets/fineweb10B_sp1024/ \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
ITERATIONS=3000 \
MAX_WALLCLOCK_SECONDS=0 \
TIED_EMBED_LR=0.03 \
MATRIX_LR=0.035 \
SCALAR_LR=0.03 \
QK_GAIN_INIT=1.0 \
MUON_MOMENTUM=0.97 \
WARMDOWN_ITERS=0 \
python -u train_gpt_xma.py 2>&1 | tee logs/logs_ablation_lr_warmdown_exp3/ablate_d_low_lr_no_warmdown.txt

# Exp E: lower LR, short warmdown
CUDA_VISIBLE_DEVICES=0 \
RUN_ID=ablate_e_low_lr_short_warmdown \
DATA_PATH=./data/datasets/fineweb10B_sp1024/ \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
ITERATIONS=3000 \
MAX_WALLCLOCK_SECONDS=0 \
TIED_EMBED_LR=0.03 \
MATRIX_LR=0.035 \
SCALAR_LR=0.03 \
QK_GAIN_INIT=1.0 \
MUON_MOMENTUM=0.97 \
WARMDOWN_ITERS=200 \
python -u train_gpt_xma.py 2>&1 | tee logs/logs_ablation_lr_warmdown_exp3/ablate_e_low_lr_short_warmdown.txt
