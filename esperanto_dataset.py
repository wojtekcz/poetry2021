from torch.utils.data import Dataset
import torch
from transformers import ByteLevelBPETokenizer, BertProcessing
from pathlib import Path


class EsperantoDataset(Dataset):
    def __init__(self, evaluate: bool = False):

        data_path = Path('data/esperberto')
        dataset_path = data_path / 'dataset'
        tokenizer_path = data_path / 'tokenizer2'

        tokenizer = ByteLevelBPETokenizer(
            tokenizer_path / "vocab.json",
            tokenizer_path / "merges.txt",
        )
        tokenizer._tokenizer.post_processor = BertProcessing(
            ("</s>", tokenizer.token_to_id("</s>")),
            ("<s>", tokenizer.token_to_id("<s>")),
        )
        tokenizer.enable_truncation(max_length=512)
        # or use the RobertaTokenizer from `transformers` directly.

        self.examples = []

        # src_files = Path("./data/").glob("*-eval.txt") if evaluate else Path("./data/").glob("*-train.txt")
        src_files = [str(dataset_path / 'oscar.eo.1000.txt')]

        for src_file in src_files:
            print("🔥", src_file)
            lines = src_file.read_text(encoding="utf-8").splitlines()
            self.examples += [x.ids for x in tokenizer.encode_batch(lines)]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        # We’ll pad at the batch level.
        return torch.tensor(self.examples[i])
