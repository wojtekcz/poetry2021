run=4a

python3 run_mlm.py \
    --model_name_or_path ./runs/esperberto3/run_4/ \
    --output_dir ./runs/esperberto3/run_${run}_logs \
    --logging_dir ./runs/esperberto3/run_${run}_logs \
    --overwrite_output_dir \
    --model_type roberta \
    --tokenizer_name ./data/esperberto3/tokenizer \
    --line_by_line \
    --max_seq_length 128 \
    --train_file ./data/esperberto3/dataset/oscar.eo.1000x10.txt \
    --validation_file ./data/esperberto3/dataset/oscar.eo.1000.txt \
    --do_eval \
    --evaluation_strategy 'steps' \
    --seed 42 \
    --logging_steps 1 \
    --eval_steps 1
