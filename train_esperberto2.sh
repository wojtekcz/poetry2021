python3 run_language_modeling.py \
    --output_dir ./run/esperberto2/run_1 \
    --model_type roberta \
    --mlm \
    --config_name ./data/esperberto2/model_config \
    --tokenizer_name roberta-base \
    --train_data_file ./data/esperberto2/dataset/oscar.eo.1000x10.txt \
    --do_train \
    --learning_rate 1e-4 \
    --num_train_epochs 5 \
    --save_total_limit 2 \
    --save_steps 2000 \
    --per_gpu_train_batch_size 16 \
    --seed 42

#    --tokenizer_name ./data/esperberto2/tokenizer \
#    --do_eval \
