#! pip install tokenizers

from pathlib import Path
from tokenizers import ByteLevelBPETokenizer

data_path = Path('data/esperberto2')
dataset_path = data_path / 'dataset'
tokenizer_path = data_path / 'tokenizer2'

paths = [str(dataset_path / 'oscar.eo.1000.txt')]

# Initialize a tokenizer
tokenizer = ByteLevelBPETokenizer()

# Customize training
tokenizer.train(files=paths, vocab_size=10_000, min_frequency=2, special_tokens=[
    "<s>",
    "<pad>",
    "</s>",
    "<unk>",
    "<mask>",
])

# Save files to disk
tokenizer_path.mkdir(parents=True, exist_ok=True)
tokenizer.save_model(str(tokenizer_path))
