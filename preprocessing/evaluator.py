from .text_tokenizer import TextTokenizer


class Evaluator:

    def __init__(self, text_tokenizer: TextTokenizer, tokenizer2):
        self.text_tokenizer = text_tokenizer
        self.tokenizer2 = tokenizer2

    @staticmethod
    # Wskaźnik liczby sylab, z których nie dało się skleić słów:
    def bad_words(e_str):
        e_syl = e_str.split(' ')
        return (e_str.count('++') + e_str.count('--')) / len(e_syl)
    # def bad_words(e_syl): e_str = syl2str(e_syl); return (e_str.count('++') + e_str.count('--')) / len(e_syl)

    def print_eval(self, generated):
        print(f'bad_words: {self.bad_words(generated)}')
        e_syl = generated.split(' ')
        decoded = self.text_tokenizer.decode_caps(self.text_tokenizer.syl2str(e_syl, delim=''))
        print(self.text_tokenizer.fix_punctuation(decoded))
        # display(HTML(self.text_tokenizer.format_html(self.text_tokenizer.fix_punctuation(decoded))))

    def evaluate(self, model, prime_str, max_length=100, temperature=1.0, greedy=False):
        prime_tok = self.text_tokenizer.str2syl2tok(prime_str)
        prime_tok_str = " ".join(prime_tok)
        ids = self.tokenizer2.encode(prime_tok_str, return_tensors="pt")[:, :-1]
        preds = None
        if greedy:
            preds = model.generate(
                ids.to(model.device), max_length=max_length,
                temperature=temperature
            )
        else:
            preds = model.generate(
                ids.to(model.device), max_length=max_length,
                temperature=temperature,
                num_beams=10, early_stopping=True,
                no_repeat_ngram_size=1,
                do_sample=True,
                top_k=50,
                top_p=0.92
            )
        # print(f'preds[0]: {preds[0]}')
        return self.tokenizer2.decode(preds[0])
