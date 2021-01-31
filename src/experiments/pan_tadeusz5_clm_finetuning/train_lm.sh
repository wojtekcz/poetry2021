home_dir=/root/poetry2021
# home_dir=/workspace/poetry2021.gt
run=2

# TODO: + use small GPT2 model, + use pan_tadeusz dataset
PYTHONIOENCODING=UTF-8 python3 run_clm.py \
    --model_name_or_path gpt2 \
    --do_train \
    --train_file ${home_dir}/data/pan_tadeusz5/dataset/pan_tadeusz.txt \
    --output_dir ${home_dir}/runs/pan_tadeusz5/run_${run} \
    --logging_dir ${home_dir}/runs/pan_tadeusz5/run_${run}_logs \
    --overwrite_output_dir \
    --fp16 \
    --seed 42 \
    --save_total_limit 2 \
    --save_steps 1000 \
    --logging_steps 10 \
    --num_train_epochs 1000 \
    --per_device_train_batch_size 6 \
    --per_device_eval_batch_size 3

    # --do_eval \
    # --dataset_name wikitext \
    # --dataset_config_name wikitext-2-raw-v1 \
    # --disable_tqdm False
    # --model_name_or_path "${home_dir}/runs/pan_tadeusz5/run_3/" \
    # --model_type roberta \
    # --tokenizer_name ${home_dir}/data/pan_tadeusz5/tokenizer \
    # --max_seq_length 128 \
    # --train_file ${home_dir}/data/pan_tadeusz5/dataset/pan_tadeusz.syl1.x100.txt \
    # --learning_rate 5e-4 \
    # --validation_file ${home_dir}/data/pan_tadeusz5/dataset/pan_tadeusz.syl1.txt \
    # --evaluation_strategy 'steps' \
    # --eval_steps 1000 \
    # --model_name_or_path ${home_dir}/runs/pan_tadeusz5/run_1/checkpoint-6000 \
    # --line_by_line \
    # --config_name ${home_dir}/data/pan_tadeusz5/model_config3 \
