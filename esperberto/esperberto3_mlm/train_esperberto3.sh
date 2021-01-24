run=4

python3 run_mlm.py \
    --output_dir ./runs/esperberto3/run_${run} \
    --logging_dir ./runs/esperberto3/run_${run}_logs \
    --overwrite_output_dir \
    --model_type roberta \
    --config_name ./data/esperberto3/model_config \
    --tokenizer_name ./data/esperberto3/tokenizer \
    --line_by_line \
    --max_seq_length 128 \
    --train_file ./data/esperberto3/dataset/oscar.eo.1000x10.txt \
    --do_train \
    --fp16 \
    --learning_rate 1e-3 \
    --per_device_train_batch_size 1300 \
    --num_train_epochs 4000 \
    --seed 42 \
    --save_total_limit 2 \
    --save_steps 1000 \
    --logging_steps 10 \
    --disable_tqdm False
