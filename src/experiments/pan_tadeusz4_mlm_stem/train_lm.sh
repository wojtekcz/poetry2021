home_dir=/root/poetry2021
# home_dir=/workspace/poetry2021.gt
run=4
PYTHONIOENCODING=UTF-8 python3 run_mlm.py \
    --output_dir ${home_dir}/runs/pan_tadeusz4/run_${run} \
    --logging_dir ${home_dir}/runs/pan_tadeusz4/run_${run}_logs \
    --overwrite_output_dir \
    --model_name_or_path "${home_dir}/runs/pan_tadeusz4/run_3 copy/checkpoint-72000" \
    --model_type roberta \
    --tokenizer_name ${home_dir}/data/pan_tadeusz4/tokenizer \
    --max_seq_length 128 \
    --train_file ${home_dir}/data/pan_tadeusz4/dataset/pan_tadeusz.syl1.x100.txt \
    --do_train \
    --fp16 \
    --learning_rate 1e-4 \
    --per_device_train_batch_size 800 \
    --num_train_epochs 650 \
    --seed 42 \
    --save_total_limit 2 \
    --save_steps 1000 \
    --logging_steps 10 \
    --validation_file ${home_dir}/data/pan_tadeusz4/dataset/pan_tadeusz.syl1.txt \
    --do_eval \
    --evaluation_strategy 'steps' \
    --eval_steps 1000 \
    --per_device_eval_batch_size 800

    # --disable_tqdm False
    # --model_name_or_path ${home_dir}/runs/pan_tadeusz4/run_1/checkpoint-6000 \
    # --line_by_line \
    # --config_name ${home_dir}/data/pan_tadeusz4/model_config3 \
