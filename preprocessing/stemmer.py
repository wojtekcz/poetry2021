from pathlib import Path
import platform
import os


class Stemmer:
    # Dzielimy korpus na sylaby programem `stemmer`.
    @staticmethod
    def stem_file(fn_corpus_caps: Path, fn_corpus_syl: Path, s_opts=7683):
        platform_suffixes = {'Linux': 'linux', 'Darwin': 'macos'}
        platform_suffix = platform_suffixes[platform.system()]
        stemmer_bin = f'LD_PRELOAD="" bin/stemmer.{platform_suffix}'
        os.system(f'{stemmer_bin} -s {s_opts} -d bin/stemmer2.dic -i {fn_corpus_caps} -o {fn_corpus_syl}')
