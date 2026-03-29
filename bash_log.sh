cd /v/ai/nobackup/xma/openai/parameter-golf

# Smoke test
CUDA_VISIBLE_DEVICES=0 \
RUN_ID=torch_smoke_baseline \
DATA_PATH=./data/datasets/fineweb10B_sp1024/ \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
# ITERATIONS=200 \
ITERATIONS=700 \
TRAIN_BATCH_TOKENS=8192 \
VAL_LOSS_EVERY=0 \
VAL_BATCH_SIZE=8192 \
python -u train_gpt_xma.py 2>&1 | tee logs/torch_smoke_direct_700.txt

# Exp 1: lower LR, longer warmdown
CUDA_VISIBLE_DEVICES=0 \
RUN_ID=exp1_low_lr_long_warmdown \
DATA_PATH=./data/datasets/fineweb10B_sp1024/ \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
TIED_EMBED_LR=0.03 \
MATRIX_LR=0.035 \
SCALAR_LR=0.03 \
QK_GAIN_INIT=1.25 \
WARMDOWN_ITERS=1600 \
python -u train_gpt_xma.py 2>&1 | tee logs/exp1_low_lr_long_warmdown.txt

# Exp 2: slightly more aggressive updates
CUDA_VISIBLE_DEVICES=0 \
RUN_ID=exp2_high_lr \
DATA_PATH=./data/datasets/fineweb10B_sp1024/ \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
TIED_EMBED_LR=0.07 \
MATRIX_LR=0.045 \
SCALAR_LR=0.05 \
QK_GAIN_INIT=1.75 \
WARMDOWN_ITERS=1200 \
python -u train_gpt_xma.py 2>&1 | tee logs/exp2_high_lr.txt

# Exp 3: conservative attention, stronger finish
CUDA_VISIBLE_DEVICES=0 \
RUN_ID=exp3_long_warmdown_momentum \
DATA_PATH=./data/datasets/fineweb10B_sp1024/ \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
TIED_EMBED_LR=0.05 \
MATRIX_LR=0.04 \
SCALAR_LR=0.04 \
QK_GAIN_INIT=1.0 \
WARMDOWN_ITERS=2200 \
MUON_MOMENTUM=0.97 \
python -u train_gpt_xma.py 2>&1 | tee logs/exp3_long_warmdown_momentum.txt

# Exp 4: stronger embedding learning
CUDA_VISIBLE_DEVICES=0 \
RUN_ID=exp4_embed_heavier \
DATA_PATH=./data/datasets/fineweb10B_sp1024/ \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
TIED_EMBED_LR=0.08 \
MATRIX_LR=0.035 \
SCALAR_LR=0.035 \
QK_GAIN_INIT=1.5 \
TIED_EMBED_INIT_STD=0.007 \
python -u train_gpt_xma.py 2>&1 | tee logs/exp4_embed_heavier.txt

# Exp 5: deeper and slightly narrower at similar budget
CUDA_VISIBLE_DEVICES=0 \
RUN_ID=exp5_deeper_narrower \
DATA_PATH=./data/datasets/fineweb10B_sp1024/ \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
NUM_LAYERS=10 \
MODEL_DIM=480 \
NUM_HEADS=8 \
NUM_KV_HEADS=4 \
MLP_MULT=2 \
TIED_EMBED_LR=0.05 \
MATRIX_LR=0.04 \
SCALAR_LR=0.04 \
QK_GAIN_INIT=1.5 \
python -u train_gpt_xma.py 2>&1 | tee logs/exp5_deeper_narrower.txt
