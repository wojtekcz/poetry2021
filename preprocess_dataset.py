#!/usr/bin/env python
# coding: utf-8

# Załadowanie bibliotek
from pathlib import Path
from preprocessing import *


# Preprocessing korpusu

def print_head(file_path, n_lines=10):
    print('\n'.join(file_path.read_text().split('\n')[:n_lines]))

dataset_path =   Path('data')/'pan_tadeusz'

fn_corpus_char = dataset_path/'pan_tadeusz.txt'
fn_corpus_caps = dataset_path/'pan_tadeusz.caps1.txt'
fn_corpus_syl = dataset_path/'pan_tadeusz.syl1.txt'
fn_corpus_sampled = dataset_path/'pan_tadeusz.sampled1.txt'

print(f'Corpus: {fn_corpus_char}')
print_head(fn_corpus_char)

# Plik wejściowy (korpus) to duży plik tekstowy. 
tokenizer = TextTokenizer(dataset_path)
processor = TextProcessor(dataset_path, tokenizer)

print(f'\nTokenizacja wielkich liter: {fn_corpus_caps.name}')
processor.do_caps_file(fn_corpus_char, fn_corpus_caps)
print_head(fn_corpus_caps)

print('\nPodział korpusu na sylaby')
Stemmer.stem_file(fn_corpus_caps, fn_corpus_syl)
print_head(fn_corpus_syl)

# Załadowanie do pamięci i tokenizacja
file_tok = processor.load_and_tokenize_file(fn_corpus_syl, repl_unk=False)

# create & save vocab
tokenizer.create_vocab(file_tok)
tokenizer.save_vocab(dataset_path/'all_tokens.json')

text = 'LITWO! Ojczyzno moja!\nTy jesteś jak zdrowie.\nIle cię trzeba cenić ble ble '
print(f'\nTesting tokenizer: {text}')
text_tok = tokenizer.str2syl2tok(text); print(text_tok)

print(tokenizer.syl2str(text_tok))
text_decoded = tokenizer.decode_caps(tokenizer.syl2str(text_tok, delim=''))[:300]
print(text_decoded)
e_str = tokenizer.fix_punctuation(text_decoded)[:400]
print(e_str)
print(tokenizer.format_html(e_str))

min_n_samples=50000
print(f"\nLet's make dataset with more than minimum {min_n_samples} samples")
chunk_len = 100 #400
line_chunker = LineChunker(file_tok=file_tok, chunk_len=chunk_len)
n_samples = len(file_tok) // chunk_len
n_samples = max(min_n_samples, n_samples)
print(f'chunk_len: {chunk_len}')
print(f'n_samples: {n_samples}')

sampled_chunks = [" ".join(line_chunker.random_chunk()) for _ in range(n_samples)]
fn_corpus_sampled.write_text("\n".join(sampled_chunks))
print(fn_corpus_sampled)

print('ala ma kota')
