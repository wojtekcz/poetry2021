from pathlib import Path
import string
import re
from .stemmer import Stemmer


class TextTokenizer:

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

    def tokenize(self, a_str: str, repl_unk=True) -> [str]: 
        strings = self.re_tok.sub(r' \1 ', a_str).replace('\n', ' _eol_ ').split()
        if repl_unk:
            strings = [self.str2tok(s) for s in strings]
        return strings

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
            return TextTokenizer.X(self.replace(p, f'<font color="{c}">{p}</font>'))
        def rpl2(self, p, p2):
            return TextTokenizer.X(self.replace(p, p2))

    @staticmethod
    def format_html(e_str):
        return TextTokenizer.X(e_str).rpl('/').rpl('--', c='red').rpl('++', c='red').rpl2('\n', '\n<br/>')
