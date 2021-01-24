from torch.utils.data import Dataset
import torch
# from tokenizers import ByteLevelBPETokenizer
# from tokenizers.processors import BertProcessing
from pathlib import Path
from transformers import RobertaTokenizerFast


class EsperantoDataset(Dataset):

    @staticmethod
    def get_tokenizer(max_len=128):
        # /workspace/poetry2021.gt/data
        data_path = Path('/workspace/poetry2021.gt/data/esperberto2')
        tokenizer_path = data_path / 'tokenizer'

        # tokenizer = ByteLevelBPETokenizer(
        #     str(tokenizer_path / "vocab.json"),
        #     str(tokenizer_path / "merges.txt"),
        # )
        # tokenizer._tokenizer.post_processor = BertProcessing(
        #     ("</s>", tokenizer.token_to_id("</s>")),
        #     ("<s>", tokenizer.token_to_id("<s>")),
        # )
        # tokenizer.enable_truncation(max_length=512)
        # or use the RobertaTokenizer from `transformers` directly.
        tokenizer = RobertaTokenizerFast.from_pretrained(tokenizer_path, max_len=max_len)
        return tokenizer

    def __init__(self, max_len=128, evaluate: bool = False):
        data_path = Path('/workspace/poetry2021.gt/data/esperberto2')
        dataset_path = data_path / 'dataset'
        tokenizer = __class__.get_tokenizer(max_len)

        self.examples = []

        # src_files = Path("./data/").glob("*-eval.txt") if evaluate else Path("./data/").glob("*-train.txt")
        src_files = [dataset_path / 'oscar.eo.1000x10.txt']

        for src_file in src_files:
            print("🔥", src_file)
            lines = src_file.read_text(encoding="utf-8").splitlines()
            # self.examples += [x.ids for x in tokenizer.encode_batch(lines)]
            lines = [x[:max_len] for x in lines]  # truncate long lines
            self.examples += [x for x in tokenizer.batch_encode_plus(lines).input_ids]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        # We’ll pad at the batch level.
        return torch.tensor(self.examples[i])


# tokenizer = EsperantoDataset.get_tokenizer()
# dataset = EsperantoDataset()
