from pathlib import Path
import torch
from preprocessing.text_tokenizer import TextTokenizer
from transformers import PreTrainedTokenizerFast, RobertaForMaskedLM
from preprocessing.evaluator import Evaluator


# Check that PyTorch sees it
USE_GPU = torch.cuda.is_available()
# USE_GPU = False
print(f'USE_GPU={USE_GPU}')

run_path = Path('runs') / 'run_4'
model_path = run_path / 'model'

dataset_path = Path('data') / 'pan_tadeusz'
text_tokenizer = TextTokenizer(dataset_path)
text_tokenizer.load_vocab(dataset_path / 'vocab.json')

tokenizer2 = PreTrainedTokenizerFast.from_pretrained(
    dataset_path / 'my-pretrained-tokenizer-fast2',
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

max_length = 100

eval = Evaluator(text_tokenizer, tokenizer2)

# gen1 = evaluate('chwycił na taśmie przypięty', max_length=max_length, temperature=1.0)
# print_eval(gen1)
# gen1

# print_eval(evaluate('Litwo! Ojczyzno', max_length=max_length, temperature=1.0))
# print_eval(evaluate('Litwo! Ojczyzno', max_length=max_length, temperature=0.8))
# print_eval(evaluate('Litwo! Ojczyzno', max_length=max_length, temperature=1.5))
# print_eval(evaluate('Tadeusz', max_length=max_length, temperature=0.8))
# print_eval(evaluate('Moskale', max_length=max_length, temperature=0.8))

eval.print_eval(eval.evaluate(model, 'Ruszyli szczwacze zwolna,', max_length=max_length, greedy=True))

# eval.print_eval(eval.evaluate('Ruszyli szczwacze zwolna, jeden tuż za drugim, _eol_ Ale za bramą rzędem rozbiegli się długim; _eol_ ', max_length=max_length, temperature=1.0))
