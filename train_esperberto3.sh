python3 run_mlm.py \
    --model_name_or_path roberta-base \
    --dataset_name wikitext \
    --dataset_config_name wikitext-2-raw-v1 \
    --do_train \
    --do_eval \
    --disable_tqdm False \
    --output_dir /content/tmp/test-mlm

    # --fp16 \
