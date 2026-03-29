cd /v/ai/nobackup/xma/openai/parameter-golf

mkdir -p logs/logs_slidingwindow_eval

# sliding window eval ablations based on Exp C: exp3 backbone, medium warmdown

# sliding window eval off
CUDA_VISIBLE_DEVICES=0 \
EVAL_STRIDE=0 \
RUN_ID=slidingwindow_eval \
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
python -u train_gpt_xma.py 2>&1 | tee logs/logs_slidingwindow_eval/slidingwindow_eval_off.txt

# sliding window eval on
CUDA_VISIBLE_DEVICES=0 \
EVAL_STRIDE=64 \
EVAL_BATCH_SEQS=256 \
RUN_ID=slidingwindow_eval \
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
python -u train_gpt_xma.py 2>&1 | tee logs/logs_slidingwindow_eval/slidingwindow_eval_on.txt

