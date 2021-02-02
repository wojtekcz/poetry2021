#!/usr/bin/env python
# coding: utf-8

from pathlib import Path
from preprocessing.stemmer import Stemmer
from preprocessing.line_chunker import LineChunker, flatten
from preprocessing.text_processor import TextProcessor
from preprocessing.text_tokenizer import TextTokenizer
from typing import List
from pprint import pprint
from tqdm import tqdm


data_path = Path('/workspace/poetry2021.gt/data') / 'pan_tadeusz6'
dataset_path = data_path / 'dataset'

# corpus_path = dataset_path / 'poezja stara - Mickiewicz'
# corpus_path = data_path / 'dataset_fixes'
corpus_path = data_path / 'dataset'
# fn_corpus_char = corpus_path / 'pan_tadeusz.txt'

# fn_corpus_caps = processed_path / 'pan_tadeusz.caps1.txt'
# fn_corpus_syl = processed_path / 'pan_tadeusz.syl1.txt'
# fn_corpus_sampled = processed_path / 'pan_tadeusz.sampled1.txt'

vocab_path = data_path / 'vocab.json'
stem_delim = '++ --'
# stem_delim = ' ##'


def print_head(file_path, n_lines=10):
    print('\n'.join(file_path.read_text().split('\n')[:n_lines]))


class DatasetPreprocessor:
    def __init__(self, dataset_path: Path):
        self.tokenizer = TextTokenizer(dataset_path)
        self.processor = TextProcessor(dataset_path, self.tokenizer)

    def tokenize_caps(self, fn_corpus_char: Path, fn_corpus_caps: Path, verbose: True):
        if verbose:
            print(f'\nTokenizacja wielkich liter: {fn_corpus_caps.name}')
        self.processor.do_caps_file(fn_corpus_char, fn_corpus_caps)
        if verbose:
            print_head(fn_corpus_caps)

    def stem_corpus(self, fn_corpus_caps: Path, fn_corpus_syl: Path, stem_delim: str, verbose: True):
        if verbose:
            print(f'\nPodział korpusu na sylaby "{stem_delim}"')
        Stemmer.stem_file(fn_corpus_caps, fn_corpus_syl, stem_delim=stem_delim)
        if verbose:
            print_head(fn_corpus_syl)

    def load_and_create_vocab(self, fn_corpus_syl: Path, vocab_path: Path) -> List[str]:
        # Załadowanie do pamięci i tokenizacja
        if fn_corpus_syl.is_dir():
            file_tok = flatten([self.processor.load_and_tokenize_file(x, repl_unk=False) for x in fn_corpus_syl.glob('*.txt')])
        else:
            file_tok = self.processor.load_and_tokenize_file(fn_corpus_syl, repl_unk=False)

        # create & save vocab
        self.tokenizer.create_vocab(file_tok)
        self.tokenizer.save_vocab(vocab_path)
        return file_tok

    def create_sampled_file(self, file_tok: List[str], fn_corpus_sampled: Path, min_n_samples: int, max_n_samples=None, chunk_len=100):
        print(f"\nLet's make dataset with more than minimum {min_n_samples} samples")
        line_chunker = LineChunker(file_tok=file_tok, chunk_len=chunk_len)
        n_samples = len(file_tok) // chunk_len
        print(f'n_samples: {n_samples}')
        n_samples = max(min_n_samples, n_samples)
        if max_n_samples is not None:
            n_samples = min(max_n_samples, n_samples)
        print(f'chunk_len: {chunk_len}')
        print(f'n_samples: {n_samples}')

        sampled_chunks = [" ".join(line_chunker.random_chunk()) for _ in tqdm(range(n_samples))]
        fn_corpus_sampled.write_text("\n".join(sampled_chunks))
        print(fn_corpus_sampled)


processed_path = data_path / 'processed_dataset'
caps_path = processed_path / 'caps'
syl_path = processed_path / 'syl'
sampled_path = processed_path / 'sampled'

caps_path.mkdir(parents=True, exist_ok=True)
syl_path.mkdir(parents=True, exist_ok=True)
sampled_path.mkdir(parents=True, exist_ok=True)

print('Files to preprocess:')
paths = [x for x in corpus_path.glob("**/*.txt")]
# paths = [x for x in dataset_path.glob("**/*.txt")]
pprint(paths)

preprocessor = DatasetPreprocessor(dataset_path)

# print(f'Corpus: {fn_corpus_char}')
# print_head(fn_corpus_char)

if False:
    for char_path in paths:
        print(f'tokenizing caps and stemming: {char_path.name}')
        corpus_caps_path = caps_path / f'{char_path.stem}.caps1.txt'
        corpus_syl_path = syl_path / f'{char_path.stem}.syl1.txt'

        preprocessor.tokenize_caps(char_path, corpus_caps_path, verbose=False)
        preprocessor.stem_corpus(corpus_caps_path, corpus_syl_path, stem_delim=stem_delim, verbose=False)

print(f'creating vocab: {vocab_path}')
file_tok = preprocessor.load_and_create_vocab(syl_path, vocab_path)
tokenizer = preprocessor.tokenizer

text = 'LITWO! Ojczyzno moja!\nTy jesteś jak zdrowie.\nIle cię trzeba cenić ble ble '
print(f'\nTesting tokenizer: {text}')
text_tok = tokenizer.str2syl2tok(text, stem_delim=stem_delim)
print(text_tok)

print(tokenizer.syl2str(text_tok, stem_delim=stem_delim))
text_decoded = tokenizer.decode_caps(tokenizer.syl2str(text_tok, delim='', stem_delim=stem_delim))[:300]
print(text_decoded)
e_str = tokenizer.fix_punctuation(text_decoded)[:400]
print(e_str)
print(tokenizer.format_html(e_str))

min_n_samples = 10000  # 50000
max_n_samples = 10000
chunk_len = 100  # 400
fn_corpus_sampled = sampled_path / f'dataset.sampled1.{max_n_samples}.txt'
preprocessor.create_sampled_file(file_tok, fn_corpus_sampled, min_n_samples=min_n_samples, max_n_samples=max_n_samples, chunk_len=chunk_len)
