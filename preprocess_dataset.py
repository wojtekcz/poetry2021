#!/usr/bin/env python
# coding: utf-8

# Załadowanie bibliotek
from pathlib import Path
import platform
import string
import random
import re
import os

# Preprocessing korpusu
dataset_path =   Path('data')/'pan_tadeusz'

fn_corpus_char = dataset_path/'pan_tadeusz.txt'
fn_corpus_caps = dataset_path/'pan_tadeusz.caps1.txt'
fn_corpus_syl = dataset_path/'pan_tadeusz.syl1.txt'
fn_corpus_sampled = dataset_path/'pan_tadeusz.sampled1.txt'

def print_head(file_path, n_lines=21):
    print('\n'.join(file_path.read_text().split('\n')[:n_lines]))

print_head(fn_corpus_char)


class Stemmer:
    # Dzielimy korpus na sylaby programem `stemmer`.
    @staticmethod
    def stem_file(fn_corpus_caps: Path, fn_corpus_syl: Path, s_opts=7683):
        platform_suffixes = {'Linux': 'linux', 'Darwin': 'macos'}
        platform_suffix = platform_suffixes[platform.system()]
        stemmer_bin = f'LD_PRELOAD="" bin/stemmer.{platform_suffix}'
        os.system(f'{stemmer_bin} -s {s_opts} -d bin/stemmer2.dic -i {fn_corpus_caps} -o {fn_corpus_syl}')


class TextProcessor:

    def __init__(self, dataset_path: Path):
        # taken from fastai/text.py
        # remove +,- chars from punctuation set to keep syllables e.g.'--PO++' intact
        # remove _ char to keep tokens intact
        # remove <,> chars to keep tokens intact
        punctuation=re.sub('[_\\+-<>]', '', string.punctuation)
        self.re_tok = re.compile(f'([{punctuation}“”¨«»®´·º½¾¿¡§£₤‘’])')

        self.dataset_path = dataset_path
        self.tmp_path = dataset_path / 'tmp'
        self.tmp_path.mkdir(parents=True, exist_ok=True)

    # Zamieniamy duże litery na małe dodając tokeny `_up_` (dla wyrazów pisanych wielkimi literami) lub `_cap_` (dla wyrazów pisanych z wielkiej litery).
    @staticmethod
    def do_caps(a_str: str) -> str:
        TOK_UP, TOK_CAP = ' _up_ ', ' _cap_ '
        res = []
        # re_word = re.compile('\w')
        for s in re.findall(r'\w+|\W+', a_str):
            res += ([TOK_UP,s.lower()] if (s.isupper() and (len(s)>2))
                    else [TOK_CAP,s.lower()] if s.istitle()
                    else [s.lower()])
        return ''.join(res)

    @staticmethod
    def do_caps_file(fn_corpus_char: Path, fn_corpus_caps: Path):
        corpus_tmp = fn_corpus_char.open('r').read()
        corpus_tmp = __class__.do_caps(corpus_tmp)
        # trim lines
        corpus_lines = [x.strip() for x in corpus_tmp.split('\n')]
        corpus_tmp = '\n'.join(corpus_lines)
        fn_corpus_caps.open('w').write(corpus_tmp)

    def tokenize(self, a_str: str, repl_unk=True) -> [str]: 
        strings = self.re_tok.sub(r' \1 ', a_str).replace('\n', ' _eol_ ').split()
        if repl_unk:
            strings = [self.str2tok(s) for s in strings]
        return strings

    # Ładujemy korpus do pamięci i tokenizujemy.
    def load_and_tokenize_file(self, fn_corpus_syl: Path, create_vocab=True):
        """
        outputs:
            file_tok, file_tok_len
        """
        a_file = open(fn_corpus_syl).read()
        repl_unk = not create_vocab
        self.file_tok = self.tokenize(a_file, repl_unk=repl_unk)
        self.file_tok_len = len(self.file_tok)

        if create_vocab:
            self.create_vocab(self.file_tok)
            self.save_vocab(self.dataset_path/'all_tokens.txt')

    # Tworzymy też listę wszystkich tokenów `all_tokens`. Mamy już specjalne tokeny `_cap_` i `_up_`, zamieniamy znaki końca lini na token `_eol_` i dodajemy token `_unk_` na wypadek, gdybyśmy użyli sylaby (tokena), który nie wystąpił wcześniej w korpusie.
    def create_vocab(self, file_tok: [str]):
        """
        outputs:
            all_tokens, tok2idx_dict
        """
        spec_tokens = ['<unk>', '<pad>', '<mask>', '<s>', '</s>', '_eol_', '_cap_', '_up_']
        all_tokens = []; all_tokens.extend(spec_tokens)
        all_tokens.extend(sorted(list(set([x for x in file_tok if not x in spec_tokens]))))
        self.all_tokens = all_tokens
        self.tok2idx_dict = {tok: idx for (idx, tok) in enumerate(all_tokens)}

    def save_vocab(self, vocab_path: Path):
        vocab_path.write_text(' '.join(self.all_tokens))

    def load_vocab(self, vocab_path: Path):
        self.all_tokens = vocab_path.read_text().split(' ')
        self.tok2idx_dict = {tok: idx for (idx, tok) in enumerate(self.all_tokens)}

    def str2tok(self, a_str: str) -> str:
        return a_str if self.tok2idx_dict.get(a_str, 0) else '<unk>'

    def tok2idx(self, tok: str) -> int:
        return self.tok2idx_dict.get(tok, 0)

    # Przyda nam się funkcja do zakodowania dowolnego tekstu na listę zsylabizowanych tokenów:
    def str2syl2tok(self, text: str) -> [str]:  
        fn_tmp_text_caps = self.tmp_path / 'tmp_text_caps1.txt'
        fn_tmp_text_syl = self.tmp_path / 'tmp_text_syl1.txt'

        text = self.do_caps(text)
        fn_tmp_text_caps.open('w').write(text)
        Stemmer.stem_file(fn_tmp_text_caps, fn_tmp_text_syl)
        text_syl = fn_tmp_text_syl.open('r').read()

        # kill last \n eol char possibly added by stemmer
        if text_syl[-1] == '\n':
            text_syl = text_syl[:-1]

        text_tok = self.tokenize(text_syl, repl_unk=True)
        return text_tok

    # Funkcje pomocnicze do zdekodowania listy tokenów na tekst:
    @staticmethod
    def syl2str(a_list: [str], delim='/') -> str:
        s = ' '.join(a_list)

        repl_list = [
            ('++ --', delim), 
        ]
        for repl in repl_list:
            s = s.replace(repl[0], repl[1])

        return s

    @staticmethod
    def do_uncaps_tokens(e_str: str) -> str:
        # decode _eol_, _cap_ and _up_
        # leave <unk> token alone
        # kill <s> and </s>
        e_syl = e_str.split(' ')
        e_syl2 = []

        cap = False; up = False

        for syl in e_syl:
            if syl == '_eol_': syl = '\n'

            if syl not in ['_cap_', '_up_', '<s>', '</s>']:
                if cap == True: syl = syl.title(); cap = False
                if up == True: syl = syl.upper(); up = False        
                e_syl2.append(syl)

            if syl == '_cap_': cap = True
            if syl == '_up_': up = True

        return ' '.join(e_syl2)

    @staticmethod
    def fix_punctuation(s: str) -> str: 
        repl_list = [
            ('\n ', '\n'), 
            (' ,', ','),
            (' .', '.'),
            (' !', '!'),
            (' ?', '?'),
            (' ;', ';'),
            ('( ', '('),
            (' )', ')'),
            (' «', '«'),
            ('» ', '»'),
            (' :', ':')
        ]
        
        for repl in repl_list:
            s = s.replace(repl[0], repl[1])
        
        return s

    # Sformatujmy zdekodowany tekst w HTML i zaznaczmy na czerwono sylaby, z których nie dało się skleić słów.
    class X(str):
        def rpl(self, p, c='lightgray'):
            return TextProcessor.X(self.replace(p, f'<font color="{c}">{p}</font>'))
        def rpl2(self, p, p2):
            return TextProcessor.X(self.replace(p, p2))

    @staticmethod
    def format_html(e_str):
        return TextProcessor.X(e_str).rpl('/').rpl('--', c='red').rpl('++', c='red').rpl2('\n', '\n<br/>')

# Plik wejściowy (korpus) to duży plik tekstowy. 
processor = TextProcessor(dataset_path)

# Tokenizacja wielkich liter
processor.do_caps_file(fn_corpus_char, fn_corpus_caps)
print_head(fn_corpus_caps)

# Podział korpusu na sylaby
Stemmer.stem_file(fn_corpus_caps, fn_corpus_syl)
print_head(fn_corpus_syl)

# Załadowanie do pamięci i tokenizacja
# TODO: decouple tokenization, also save all tokens
processor.load_and_tokenize_file(fn_corpus_syl)

tekst = 'LITWO! Ojczyzno moja!\nTy jesteś jak zdrowie.\nIle cię trzeba cenić ble ble '
tekst_tok = processor.str2syl2tok(tekst); print(tekst_tok)

print(processor.syl2str(tekst_tok))
print(processor.do_uncaps_tokens(processor.syl2str(tekst_tok, delim=''))[:300])
print(processor.fix_punctuation(processor.do_uncaps_tokens(processor.syl2str(tekst_tok, delim='')))[:300])

e_str = processor.fix_punctuation(processor.do_uncaps_tokens(processor.syl2str(tekst_tok, delim='')))[:400]
e_html = processor.format_html(e_str); print(e_html)

# Sample chunk_len token-sized chunks to a file
chunk_len = 100 #400

# def random_chunk():
#     start_index = random.randint(0, file_tok_len - chunk_len -1)
#     end_index = start_index + chunk_len + 1
#     return file_tok[start_index:end_index]

flatten = lambda t: [item for sublist in t for item in sublist]

class LineChunker:
    def __init__(self, file_tok: [str], chunk_len: int):
        file_str = " ".join(file_tok)
        file_lines = [(x.strip() + ' _eol_').strip() for x in file_str.split('_eol_')]
        self.file_lines_tok = [x.split() for x in file_lines]
        self.chunk_len = chunk_len
        self.last_num_lines = self.count_tok_lines(self.file_lines_tok[::-1], chunk_len=self.chunk_len)
        self.last_line_index = len(self.file_lines_tok) - self.last_num_lines

    @staticmethod
    def count_tok_lines(file_lines_tok: [[str]], chunk_len: int):
        # count how many last lines (almost) add up to chunk_len
        sum_tok = 0
        idx = 0
        while True:
            n = len(file_lines_tok[idx])
            if sum_tok+n >= chunk_len or idx+1 >= len(file_lines_tok):
                break
            sum_tok += n
            idx += 1

        return idx

    def random_chunk(self):
        start_index = random.randint(0, self.last_line_index)
        num_lines = self.count_tok_lines(self.file_lines_tok[start_index:], chunk_len=self.chunk_len)
        end_index = start_index + num_lines
        return flatten(self.file_lines_tok[start_index:end_index])


# Let's make dataset with more than minimum n_samples
line_chunker = LineChunker(file_tok=processor.file_tok, chunk_len=chunk_len)

n_samples = processor.file_tok_len // chunk_len
# print(n_samples, processor.file_tok_len)
n_samples = max(50000, n_samples); n_samples

sampled_chunks = [" ".join(line_chunker.random_chunk()) for _ in range(n_samples)]
print(sampled_chunks[0])

fn_corpus_sampled.write_text("\n".join(sampled_chunks))
print(fn_corpus_sampled)
