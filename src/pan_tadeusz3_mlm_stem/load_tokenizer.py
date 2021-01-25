from pathlib import Path

data_path = Path('/workspace/poetry2021.gt/data/pan_tadeusz3')
tokenizer_path = data_path / 'my-pretrained-tokenizer-fast3'


# from transformers import PreTrainedTokenizerFast
# tokenizer = PreTrainedTokenizerFast.from_pretrained(
#     tokenizer_path,
#     max_len=128
# )

from transformers import AutoTokenizer, RobertaConfig
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, config=RobertaConfig())

print(tokenizer)
