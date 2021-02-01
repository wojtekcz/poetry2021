# home_dir=/root/poetry2021
home_dir=/workspace/poetry2021.gt

PYTHONIOENCODING=UTF-8 python3 run_generation.py \
    --model_type=gpt2 \
    --length=300 \
    --num_return_sequences=5 \
    --model_name_or_path=${home_dir}/runs/pan_tadeusz5/run_5/checkpoint-2000 \
    --tokenizer_name ${home_dir}/data/pan_tadeusz5/tokenizer
