python3 run_mlm.py \
    --output_dir ./runs/esperberto3/run_1 \
    --model_type roberta \
    --config_name ./data/esperberto3/model_config \
    --tokenizer_name ./data/esperberto3/tokenizer \
    --line_by_line \
    --max_seq_length 128 \
    --train_file ./data/esperberto3/dataset/oscar.eo.1000x10.txt \
    --do_train \
    --disable_tqdm False

    # --fp16 \
    # --learning_rate 1e-4 \
    # --num_train_epochs 5 \
    # --save_total_limit 2 \
    # --save_steps 2000 \
    # --per_device_train_batch_size 16 \
    # --seed 42
