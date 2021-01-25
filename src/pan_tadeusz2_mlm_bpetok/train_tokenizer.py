from pathlib import Path
from tokenizers import ByteLevelBPETokenizer
from tokenizers import Tokenizer
from transformers import RobertaTokenizerFast

data_path = Path('/workspace/poetry2021.gt/data/pan_tadeusz2')
dataset_path = data_path / 'dataset'
tokenizer_tmp_path = data_path / 'tokenizer_tmp'
tokenizer_path = data_path / 'tokenizer'

def save_tmp_tokenizer():
    paths = [str(dataset_path / 'pan_tadeusz.txt')]

    # Initialize a tokenizer
    tokenizer_tmp = ByteLevelBPETokenizer()

    # Customize training
    tokenizer_tmp.train(files=paths, vocab_size=10_000, min_frequency=2, special_tokens=[
        "<s>",
        "<pad>",
        "</s>",
        "<unk>",
        "<mask>",
    ])

    # Save files to disk
    tokenizer_tmp_path.mkdir(parents=True, exist_ok=True)
    tokenizer_tmp.save_model(str(tokenizer_tmp_path))

save_tmp_tokenizer()

tokenizer = RobertaTokenizerFast(
    tokenizer_tmp_path / 'vocab.json', 
    tokenizer_tmp_path / 'merges.txt'
)
tokenizer.save_pretrained(tokenizer_path)

# from transformers import AutoTokenizer, RobertaConfig
# tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, config=RobertaConfig())
