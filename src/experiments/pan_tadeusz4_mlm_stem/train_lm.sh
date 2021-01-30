home_dir=/root/poetry2021
# home_dir=/workspace/poetry2021.gt
run=3
PYTHONIOENCODING=UTF-8 python3 run_mlm.py \
    --output_dir ${home_dir}/runs/pan_tadeusz4/run_${run} \
    --logging_dir ${home_dir}/runs/pan_tadeusz4/run_${run}_logs \
    --overwrite_output_dir \
    --model_type roberta \
    --config_name ${home_dir}/data/pan_tadeusz4/model_config3 \
    --tokenizer_name ${home_dir}/data/pan_tadeusz4/tokenizer \
    --max_seq_length 128 \
    --line_by_line \
    --train_file ${home_dir}/data/pan_tadeusz4/dataset/pan_tadeusz.syl1.x100.txt \
    --do_train \
    --fp16 \
    --learning_rate 1e-3 \
    --per_device_train_batch_size 1000 \
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
