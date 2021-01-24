home_dir=/root/poetry2021
run=1
python3 run_mlm.py \
    --output_dir ${home_dir}/runs/pan_tadeusz2/run_${run} \
    --logging_dir ${home_dir}/runs/pan_tadeusz2/run_${run}_logs \
    --overwrite_output_dir \
    --model_type roberta \
    --config_name ${home_dir}/data/pan_tadeusz2/model_config \
    --tokenizer_name ${home_dir}/data/pan_tadeusz2/tokenizer \
    --line_by_line \
    --max_seq_length 128 \
    --train_file ${home_dir}/data/pan_tadeusz2/dataset/pan_tadeusz.x100.txt \
    --do_train \
    --learning_rate 1e-3 \
    --per_device_train_batch_size 4000 \
    --num_train_epochs 400 \
    --seed 42 \
    --save_total_limit 2 \
    --save_steps 1000 \
    --logging_steps 10 \
    --fp16 \
    --disable_tqdm False
