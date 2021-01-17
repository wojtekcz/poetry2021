from pathlib import Path
import torch
from preprocessing.text_tokenizer import TextTokenizer
from transformers import PreTrainedTokenizerFast, RobertaForMaskedLM

# Check that PyTorch sees it
USE_GPU = torch.cuda.is_available()
# USE_GPU = False
print(f'USE_GPU={USE_GPU}')

run_path = Path('runs')/'run_4'
model_path = run_path/'model'

dataset_path = Path('data')/'pan_tadeusz'
text_tokenizer = TextTokenizer(dataset_path)
text_tokenizer.load_vocab(dataset_path/'vocab.json')

tokenizer2 = PreTrainedTokenizerFast.from_pretrained(
    dataset_path/'my-pretrained-tokenizer-fast2',
    max_len=128
)


# 4. Check that the LM actually trained

def to_gpu(x, *args, **kwargs):
    return x.cuda(*args, **kwargs) if USE_GPU else x


# load trained model

# os.system('tar xzvf PanTadeuszRoBERTa.tgz')

model = RobertaForMaskedLM.from_pretrained(str(model_path))
model = to_gpu(model)
model.device

# ## generate
model.eval()


# Wskaźnik liczby sylab, z których nie dało się skleić słów:
def bad_words(e_str): e_syl = e_str.split(' '); return (e_str.count('++') + e_str.count('--')) / len(e_syl)
# def bad_words(e_syl): e_str = syl2str(e_syl); return (e_str.count('++') + e_str.count('--')) / len(e_syl)


def print_eval(generated):
    print(f'bad_words: {bad_words(generated)}')
    e_syl = generated.split(' ')
    decoded = text_tokenizer.decode_caps(text_tokenizer.syl2str(e_syl, delim=''))
    print(text_tokenizer.fix_punctuation(decoded))
    # display(HTML(text_tokenizer.format_html(text_tokenizer.fix_punctuation(decoded))))


def evaluate(prime_str, max_length=100, temperature=0.8):
    prime_tok = text_tokenizer.str2syl2tok(prime_str)
    prime_tok_str = " ".join(prime_tok)
    ids = tokenizer2.encode(prime_tok_str, return_tensors="pt")[:,:-1]
    preds = model.generate(ids.to(model.device), max_length=max_length, 
                           temperature=temperature, 
                           num_beams=10, early_stopping=True,
                           no_repeat_ngram_size=1,
                           do_sample=True,
                           top_k=50,
                           top_p=0.92
                           )
    print(f'preds[0]: {preds[0]}')
    return tokenizer2.decode(preds[0])


max_length = 100
# gen1 = evaluate('chwycił na taśmie przypięty', max_length=max_length, temperature=1.0)
# print_eval(gen1)
# gen1

# print_eval(evaluate('Litwo! Ojczyzno', max_length=max_length, temperature=1.0))
# print_eval(evaluate('Litwo! Ojczyzno', max_length=max_length, temperature=0.8))
# print_eval(evaluate('Litwo! Ojczyzno', max_length=max_length, temperature=1.5))
# print_eval(evaluate('Tadeusz', max_length=max_length, temperature=0.8))
# print_eval(evaluate('Moskale', max_length=max_length, temperature=0.8))

# print_eval(evaluate('Ruszyli szczwacze zwolna,', max_length=max_length, temperature=1.0))

print_eval(evaluate('Ruszyli szczwacze zwolna, jeden tuż za drugim, _eol_ Ale za bramą rzędem rozbiegli się długim; _eol_ ', max_length=max_length, temperature=1.0))
