from pathlib import Path
from .text_tokenizer import TextTokenizer


class TextProcessor:

    def __init__(self, dataset_path: Path, tokenizer: TextTokenizer):
        self.dataset_path = dataset_path
        self.tokenizer = tokenizer

    def do_caps_file(self, fn_corpus_char: Path, fn_corpus_caps: Path):
        corpus_tmp = fn_corpus_char.open('r').read()
        corpus_tmp = self.tokenizer.do_caps(corpus_tmp)
        corpus_tmp = self.tokenizer.separate_punctuation(corpus_tmp)
        # trim lines
        corpus_lines = [x.strip() for x in corpus_tmp.split('\n')]
        corpus_tmp = '\n'.join(corpus_lines)
        fn_corpus_caps.open('w').write(corpus_tmp)

    # Ładujemy korpus do pamięci i tokenizujemy.
    def load_and_tokenize_file(self, fn_corpus_syl: Path, repl_unk=False) -> [str]:
        a_file = fn_corpus_syl.read_text()
        return self.tokenizer.tokenize(a_file, repl_unk=repl_unk)
