#!/usr/bin/env python
# coding: utf-8
import os
import torch
from pathlib import Path
from tokenizers import ByteLevelBPETokenizer
from tokenizers.processors import BertProcessing
from transformers import RobertaConfig
from transformers import RobertaTokenizerFast
from transformers import RobertaForMaskedLM
from transformers import LineByLineTextDataset
from transformers import DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
from transformers import pipeline

data_path = Path('data/esperberto')
dataset_path = data_path / 'dataset'
tokenizer_path = data_path / 'tokenizer'
run_dir = Path('runs/esperberto') / 'run_1'

# ## 1. Find a dataset
# os.system('wget -c https://cdn-datasets.huggingface.co/EsperBERTo/data/oscar.eo.txt')

# ## 2. Train a tokenizer
# paths = [str(x) for x in dataset_path.glob("**/*.txt")]
paths = [str(dataset_path / 'oscar.eo.mini.txt')]

tokenizer = ByteLevelBPETokenizer()
tokenizer.train(files=paths, vocab_size=52_000, min_frequency=2, special_tokens=[
    "<s>",
    "<pad>",
    "</s>",
    "<unk>",
    "<mask>",
])
tokenizer_path.mkdir(parents=True, exist_ok=True)
tokenizer.save_model(str(tokenizer_path))
# tokenizer.save_model(str(tokenizer_path / 'tokenizer.json'))

# tokenizer = ByteLevelBPETokenizer(
#     tokenizer_path / "vocab.json",
#     tokenizer_path / "merges.txt",
# )

# tokenizer._tokenizer.post_processor = BertProcessing(
#     ("</s>", tokenizer.token_to_id("</s>")),
#     ("<s>", tokenizer.token_to_id("<s>")),
# )
# tokenizer.enable_truncation(max_length=512)

# print(tokenizer.encode("Mi estas Julien."))
# print(tokenizer.encode("Mi estas Julien.").tokens)

# ## 3. Train a language model from scratch
print(f'cuda: {torch.cuda.is_available()}')

config = RobertaConfig(
    vocab_size=52_000,
    max_position_embeddings=514,
    num_attention_heads=12,
    num_hidden_layers=6,
    type_vocab_size=1,
)

# Now let's re-create our tokenizer in transformers
tokenizer = RobertaTokenizerFast.from_pretrained(tokenizer_path, max_len=512)
model = RobertaForMaskedLM(config=config)
print(model.num_parameters())

dataset = LineByLineTextDataset(
    tokenizer=tokenizer,
    file_path=dataset_path / "oscar.eo.mini.txt",
    block_size=128,
)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=0.15
)

training_args = TrainingArguments(
    output_dir=run_dir,
    overwrite_output_dir=True,
    num_train_epochs=1,
    per_gpu_train_batch_size=64,
    save_steps=10_000,
    save_total_limit=2,
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset
)

# ### Start training
trainer.train()

# #### 🎉 Save final model (+ tokenizer + config) to disk
trainer.save_model(run_dir / "model")

# ## 4. Check that the LM actually trained
# fill_mask = pipeline(
#     "fill-mask",
#     model=str(run_dir / "model"),
#     tokenizer=str(tokenizer_path)
# )

# # The sun <mask>.
# print(fill_mask("La suno <mask>."))

# # Ok, simple syntax/grammar works. Let’s try a slightly more interesting prompt:
# print(fill_mask("Jen la komenco de bela <mask>."))

# This is the beginning of a beautiful <mask>.

# ## 5. Generate

prime_str = 'Ala ma kota'
max_length = 100

ids = tokenizer.encode(prime_str, return_tensors="pt")[:, :-1]
preds = model.generate(ids.to(model.device), max_length=max_length)

print(f'preds[0]: {preds[0]}')
print(tokenizer.decode(preds[0]))
