from pathlib import Path
from preprocessing.text_tokenizer import TextTokenizer
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import CharDelimiterSplit
from tokenizers.processors import BertProcessing
from transformers import PreTrainedTokenizerFast
from transformers import AutoTokenizer
from typing import Optional, Tuple, Union
import os
import json


data_path = Path('/workspace/poetry2021.gt/data/pan_tadeusz6')
dataset_path = data_path / 'dataset'
vocab_path = data_path / 'vocab.json'
tokenizer_tmp_path = data_path / 'tokenizer_tmp'
tokenizer_path = data_path / 'tokenizer'

text_tokenizer = TextTokenizer(dataset_path)
text_tokenizer.load_vocab(vocab_path)

vocab = text_tokenizer.vocab
vocab_count = len(vocab.keys())
vocab.update({'<|endoftext|>': vocab_count})

tokenizer_tmp = Tokenizer(WordLevel(text_tokenizer.vocab))
tokenizer_tmp.pre_tokenizer = CharDelimiterSplit(' ')

tokenizer_tmp.post_processor = BertProcessing(
    ("<|endoftext|>", tokenizer_tmp.token_to_id("<|endoftext|>")),
    ("<|endoftext|>", tokenizer_tmp.token_to_id("<|endoftext|>")),
)

tokenizer_tmp_path.mkdir(parents=True, exist_ok=True)
tokenizer_tmp.save(str(tokenizer_tmp_path / "tokenizer.json"))


# Re-create as GPT2 compatible tokenizer

class GPT2CompatibleTokenizer(PreTrainedTokenizerFast):
    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        file = str(tokenizer_path / "tokenizer.json")
        tokenizer.backend_tokenizer.save(file)
        files = [file]
        return tuple(files)

    def save_pretrained(
        self,
        save_directory: Union[str, os.PathLike],
        legacy_format: bool = True,
        filename_prefix: Optional[str] = None,
    ) -> Tuple[str]:
        a_tuple = super().save_pretrained(save_directory, legacy_format, filename_prefix)
        config_path: Path = save_directory / 'config.json'
        config_path.write_text(json.dumps({"model_type": "gpt2"}))
        return a_tuple


tokenizer = GPT2CompatibleTokenizer(tokenizer_file=str(tokenizer_tmp_path / "tokenizer.json"), model_max_length=1024, add_prefix_space=True)
tokenizer.backend_tokenizer.enable_truncation(max_length=1024)
tokenizer.bos_token = "<|endoftext|>"
tokenizer.eos_token = "<|endoftext|>"
tokenizer.unk_token = "<|endoftext|>"

tokenizer.model_max_length = 1024

# save tokenizer
tokenizer.save_pretrained(tokenizer_path)

# load tokenizer
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
print(tokenizer)
