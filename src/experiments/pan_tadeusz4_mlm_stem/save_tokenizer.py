from pathlib import Path
from preprocessing.text_tokenizer import TextTokenizer
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import CharDelimiterSplit
from tokenizers.processors import BertProcessing
from transformers import PreTrainedTokenizerFast
from transformers import AutoTokenizer, RobertaConfig
from typing import Optional, Tuple


data_path = Path('/workspace/poetry2021.gt/data/pan_tadeusz4')
dataset_path = data_path / 'dataset'
vocab_path = data_path / 'vocab.json'
tokenizer_tmp_path = data_path / 'tokenizer_tmp'
tokenizer_path = data_path / 'tokenizer'

text_tokenizer = TextTokenizer(dataset_path)
text_tokenizer.load_vocab(vocab_path)

tokenizer_tmp = Tokenizer(WordLevel(text_tokenizer.vocab))
tokenizer_tmp.pre_tokenizer = CharDelimiterSplit(' ')

tokenizer_tmp.post_processor = BertProcessing(
    ("</s>", tokenizer_tmp.token_to_id("</s>")),
    ("<s>", tokenizer_tmp.token_to_id("<s>")),
)

tokenizer_tmp_path.mkdir(parents=True, exist_ok=True)
tokenizer_tmp.save(str(tokenizer_tmp_path / "tokenizer.json"))


# Re-create as roberta compatible tokenizer

class RobertaCompatibleTokenizer(PreTrainedTokenizerFast):
    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        file = str(tokenizer_path / "tokenizer.json")
        tokenizer.backend_tokenizer.save(file)
        files = [file]
        return tuple(files)


tokenizer = RobertaCompatibleTokenizer(tokenizer_file=str(tokenizer_tmp_path / "tokenizer.json"))
tokenizer.backend_tokenizer.enable_truncation(max_length=128)
tokenizer.mask_token = "<mask>"
tokenizer.pad_token = "<pad>"
# TODO: add other special tokens
# TODO: what about tokens in tokenizer_config.json?

# save tokenizer
tokenizer.save_pretrained(tokenizer_path)

# load tokenizer
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, config=RobertaConfig())
print(tokenizer)
