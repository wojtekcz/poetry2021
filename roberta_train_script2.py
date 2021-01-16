from pathlib import Path
import torch
import os
from preprocessing.text_tokenizer import TextTokenizer
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import CharDelimiterSplit
from tokenizers.processors import BertProcessing
# https://github.com/huggingface/transformers/issues/7234#issuecomment-720092292
from transformers import RobertaTokenizerFast, RobertaConfig, RobertaForMaskedLM
from transformers import LineByLineTextDataset, DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments


dataset_path = Path('data')/'pan_tadeusz'
fn_corpus_sampled = dataset_path/'pan_tadeusz.sampled1.txt'
run_path = Path('runs')/'run_3'
model_path = run_path/'model'

# Check that PyTorch sees it
USE_GPU = torch.cuda.is_available()
# USE_GPU = False
print(f'USE_GPU={USE_GPU}')

# Roberta LM colator with stemmed text

# 2. Create a tokenizer
# load our tokenizer
text_tokenizer = TextTokenizer(dataset_path)
text_tokenizer.load_vocab(dataset_path/'vocab.json')

# Create transformers compatible tokenizer
tokenizer2 = RobertaTokenizerFast.from_pretrained(dataset_path/"my-roberta2", max_len=128)

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

model = RobertaForMaskedLM(config=config)
# !tar xzvf "PanTadeuszRoBERTa.tgz"
# model = RobertaForMaskedLM.from_pretrained("PanTadeuszRoBERTa")
print(f'model.num_parameters(): {model.num_parameters()}')

# Now let's build our training Dataset
print(fn_corpus_sampled)
dataset = LineByLineTextDataset(
    tokenizer=tokenizer2,
    file_path=fn_corpus_sampled,
    block_size=128,  # 512
)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer2, mlm=True, mlm_probability=0.15
)

# TODO: pack setup parameters per gpu

training_args = TrainingArguments(
    output_dir=str(run_path),
    logging_dir=str(run_path),
    overwrite_output_dir=True,
    num_train_epochs=10,
    per_device_train_batch_size=320,  # 64
    logging_steps=10,
    save_steps=100,
    save_total_limit=1,
    # prediction_loss_only=True,
    learning_rate=2e-4,  # 5e-05,
    fp16=True,
    # fp16_opt_level="O1",
    # fp16_backend="amp"
)
print(training_args)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset,
)

trainer.train()

# 🎉 Save final model (+ tokenizer + config) to disk
trainer.save_model(str(model_path))

# killing checkpoints before tgz-ting model
# check_path = (model_path/'checkpoint-2000')
# [x.unlink() for x in check_path.iterdir()]
# check_path.rmdir()

os.system(f'tar czvf {model_path}.tgz {model_path}/')
