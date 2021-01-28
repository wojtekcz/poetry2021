from pathlib import Path
# from transformers import PreTrainedTokenizerFast
from transformers import AutoTokenizer, RobertaConfig

data_path = Path('/workspace/poetry2021.gt/data/pan_tadeusz3')
tokenizer_path = data_path / 'my-pretrained-tokenizer-fast3'


# tokenizer = PreTrainedTokenizerFast.from_pretrained(
#     tokenizer_path,
#     max_len=128
# )

tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, config=RobertaConfig())

print(tokenizer)
