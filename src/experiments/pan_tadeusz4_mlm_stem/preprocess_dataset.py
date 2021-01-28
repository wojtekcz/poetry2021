#!/usr/bin/env python
# coding: utf-8

from pathlib import Path
from preprocessing.stemmer import Stemmer
from preprocessing.line_chunker import LineChunker
from preprocessing.text_processor import TextProcessor
from preprocessing.text_tokenizer import TextTokenizer


data_path = Path('/workspace/poetry2021.gt/data') / 'pan_tadeusz4'
dataset_path = data_path / 'dataset'

fn_corpus_char = dataset_path / 'pan_tadeusz.txt'
fn_corpus_caps = dataset_path / 'pan_tadeusz.caps1.txt'
fn_corpus_syl = dataset_path / 'pan_tadeusz.syl1.txt'
fn_corpus_syl2 = dataset_path / 'pan_tadeusz.syl2.txt'
fn_corpus_sampled = dataset_path / 'pan_tadeusz.sampled1.txt'

vocab_path = data_path / 'vocab.json'


def print_head(file_path, n_lines=10):
    print('\n'.join(file_path.read_text().split('\n')[:n_lines]))


class DatasetPreprocessor:
    def __init__(self, dataset_path: Path):
        self.tokenizer = TextTokenizer(dataset_path)
        self.processor = TextProcessor(dataset_path, self.tokenizer)

    def tokenize_caps(self, fn_corpus_char: Path, fn_corpus_caps: Path):
        print(f'\nTokenizacja wielkich liter: {fn_corpus_caps.name}')
        self.processor.do_caps_file(fn_corpus_char, fn_corpus_caps)
        print_head(fn_corpus_caps)

    def stem_corpus(self, fn_corpus_caps: Path, fn_corpus_syl: Path, stem_delim: str):
        print(f'\nPodział korpusu na sylaby "{stem_delim}"')
        Stemmer.stem_file(fn_corpus_caps, fn_corpus_syl, stem_delim=stem_delim)
        print_head(fn_corpus_syl)

    def load_and_create_vocab(self, fn_corpus_syl: Path, vocab_path: Path) -> [str]:
        # Załadowanie do pamięci i tokenizacja
        file_tok = self.processor.load_and_tokenize_file(fn_corpus_syl, repl_unk=False)

        # create & save vocab
        self.tokenizer.create_vocab(file_tok)
        self.tokenizer.save_vocab(vocab_path)
        return file_tok

    def create_sampled_file(self, fn_corpus_sampled: Path, min_n_samples: int, chunk_len: int):
        print(f"\nLet's make dataset with more than minimum {min_n_samples} samples")
        line_chunker = LineChunker(file_tok=file_tok, chunk_len=chunk_len)
        n_samples = len(file_tok) // chunk_len
        print(f'n_samples: {n_samples}')
        n_samples = max(min_n_samples, n_samples)
        print(f'chunk_len: {chunk_len}')
        print(f'n_samples: {n_samples}')

        sampled_chunks = [" ".join(line_chunker.random_chunk()) for _ in range(n_samples)]
        fn_corpus_sampled.write_text("\n".join(sampled_chunks))
        print(fn_corpus_sampled)


preprocessor = DatasetPreprocessor(data_path)

print(f'Corpus: {fn_corpus_char}')
print_head(fn_corpus_char)

preprocessor.tokenize_caps(fn_corpus_char, fn_corpus_caps)
preprocessor.stem_corpus(fn_corpus_caps, fn_corpus_syl, stem_delim='++ --')
preprocessor.stem_corpus(fn_corpus_caps, fn_corpus_syl2, stem_delim=' ##')

file_tok = preprocessor.load_and_create_vocab(fn_corpus_syl, vocab_path)
tokenizer = preprocessor.tokenizer

text = 'LITWO! Ojczyzno moja!\nTy jesteś jak zdrowie.\nIle cię trzeba cenić ble ble '
print(f'\nTesting tokenizer: {text}')
text_tok = tokenizer.str2syl2tok(text)
print(text_tok)

print(tokenizer.syl2str(text_tok))
text_decoded = tokenizer.decode_caps(tokenizer.syl2str(text_tok, delim=''))[:300]
print(text_decoded)
e_str = tokenizer.fix_punctuation(text_decoded)[:400]
print(e_str)
print(tokenizer.format_html(e_str))

min_n_samples = 1000  # 50000
chunk_len = 100  # 400
preprocessor.create_sampled_file(fn_corpus_sampled, min_n_samples, chunk_len)
