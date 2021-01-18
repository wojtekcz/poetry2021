from pathlib import Path
import torch
import os
import warnings
from preprocessing.text_tokenizer import TextTokenizer
from preprocessing.evaluator import Evaluator
# https://github.com/huggingface/transformers/issues/7234#issuecomment-720092292
from transformers import PreTrainedTokenizerFast
from transformers import RobertaConfig, RobertaForMaskedLM
from transformers import LineByLineTextDataset, DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
from typing import Dict, List, Optional
from torch.utils.data.dataset import Dataset

warnings.simplefilter(action='ignore', category=FutureWarning)

dataset_path = Path('data') / 'pan_tadeusz'
fn_corpus = dataset_path / 'pan_tadeusz.sampled1.mini.txt'
fn_corpus_eval = dataset_path / 'pan_tadeusz.sampled1.eval.txt'
run_path = Path('runs') / 'run_5'
model_path = run_path / 'model'

# Check that PyTorch sees it
USE_GPU = torch.cuda.is_available()
# USE_GPU = False
print(f'USE_GPU={USE_GPU}')

# Roberta LM colator with stemmed text

# 2. Create a tokenizer
# load our tokenizer
text_tokenizer = TextTokenizer(dataset_path)
text_tokenizer.load_vocab(dataset_path / 'vocab.json')

# Create transformers compatible tokenizer
tokenizer2 = PreTrainedTokenizerFast.from_pretrained(
    dataset_path / 'my-pretrained-tokenizer-fast2',
    max_len=128
)

# 3. Train a language model
config = RobertaConfig(
    vocab_size=tokenizer2._tokenizer.get_vocab_size(),
    hidden_size=240,
    intermediate_size=2048,
    max_position_embeddings=514,
    num_attention_heads=12,
    num_hidden_layers=6,
    type_vocab_size=1,
    bos_token_id=tokenizer2._tokenizer.token_to_id("<s>"),
    eos_token_id=tokenizer2._tokenizer.token_to_id("</s>"),
    pad_token_id=tokenizer2._tokenizer.token_to_id("<pad>"),
    # attention_probs_dropout_prob=0.0,
    # hidden_dropout_prob=0.0,
)
print(config)

# model = RobertaForMaskedLM(config=config)
# !tar xzvf "PanTadeuszRoBERTa.tgz"
model = RobertaForMaskedLM.from_pretrained('runs/run_4/model')
print(f'model.num_parameters(): {model.num_parameters()}')

# Build training/eval Datasets
print(fn_corpus)
dataset = LineByLineTextDataset(
    tokenizer=tokenizer2,
    file_path=fn_corpus,
    block_size=128,
)

eval_dataset = LineByLineTextDataset(
    tokenizer=tokenizer2,
    file_path=fn_corpus_eval,
    block_size=128,
)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer2, mlm=True, mlm_probability=0.15
)

# setting parameters per gpu
cpu_1 = {
    'per_device_train_batch_size': 2,
    'learning_rate': 5e-4,
    'max_length': 40,
    'fp16': False,
}

gpu_T4_1 = {
    'per_device_train_batch_size': 460,
    'learning_rate': 5e-4,
    'max_length': 100,
    'fp16': True,
}

dev_params = gpu_T4_1

training_args = TrainingArguments(
    output_dir=str(run_path),
    logging_dir=str(run_path),
    overwrite_output_dir=True,
    num_train_epochs=1000,
    per_device_train_batch_size=dev_params['per_device_train_batch_size'],
    logging_steps=10,
    save_steps=100,
    save_total_limit=1,
    evaluation_strategy='steps',
    # prediction_loss_only=True,
    learning_rate=dev_params['learning_rate'],
    fp16=dev_params['fp16'],
    # fp16_opt_level="O1",
    # fp16_backend="amp"
)
print(training_args)

max_length = dev_params['max_length']
eval = Evaluator(text_tokenizer, tokenizer2)


class MyTrainer(Trainer):

    def evaluate(
        self,
        eval_dataset: Optional[Dataset] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> Dict[str, float]:
        print("evaluate()")
        eval.print_eval(eval.evaluate(model, 'Ruszyli szczwacze zwolna,', max_length=max_length, greedy=True))
        rslt = super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)
        # TODO: save generation & bad_words to logger
        return rslt


trainer = MyTrainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset,
    eval_dataset=dataset,
)

trainer.train()

# Save final model to disk
trainer.save_model(str(model_path))

# killing checkpoints before tgz-ting model
# check_path = (model_path/'checkpoint-2000')
# [x.unlink() for x in check_path.iterdir()]
# check_path.rmdir()

os.system(f'tar czvf {model_path}.tgz {model_path}/')
