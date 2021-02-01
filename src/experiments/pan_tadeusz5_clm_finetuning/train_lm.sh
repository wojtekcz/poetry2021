home_dir=/root/poetry2021
# home_dir=/workspace/poetry2021.gt
run=5

PYTHONIOENCODING=UTF-8 python3 run_clm.py \
    --model_type gpt2 \
    --do_train \
    --train_file ${home_dir}/data/pan_tadeusz5/dataset/pan_tadeusz.sampled1.txt \
    --output_dir ${home_dir}/runs/pan_tadeusz5/run_${run} \
    --logging_dir ${home_dir}/runs/pan_tadeusz5/run_${run}_logs \
    --overwrite_output_dir \
    --config_name ${home_dir}/data/pan_tadeusz5/model_config2 \
    --tokenizer_name ${home_dir}/data/pan_tadeusz5/tokenizer \
    --fp16 \
    --seed 42 \
    --save_total_limit 2 \
    --save_steps 1000 \
    --logging_steps 10 \
    --num_train_epochs 1000 \
    --do_eval \
    --validation_file ${home_dir}/data/pan_tadeusz5/dataset/pan_tadeusz.sampled1.txt \
    --evaluation_strategy 'steps' \
    --eval_steps 1000 \
    --per_device_train_batch_size 6 \
    --per_device_eval_batch_size 3

    # --model_name_or_path gpt2 \
    # --model_name_or_path ${home_dir}/runs/pan_tadeusz5/run_1/checkpoint-6000 \
    # --disable_tqdm False
    # --model_type roberta \
    # --max_seq_length 128 \
    # --learning_rate 5e-4 \
    # --line_by_line \
