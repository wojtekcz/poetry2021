# home_dir=/root/poetry2021
home_dir=/workspace/poetry2021.gt

PYTHONIOENCODING=UTF-8 python3 run_generation.py \
    --model_type=gpt2 \
    --length=200 \
    --num_return_sequences=5 \
    --model_name_or_path=${home_dir}/runs/pan_tadeusz5/run_3/checkpoint-4000
