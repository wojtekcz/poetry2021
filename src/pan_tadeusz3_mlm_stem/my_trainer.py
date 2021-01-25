from transformers import Trainer, TrainingArguments
from typing import Dict, List, Optional
from torch.utils.data.dataset import Dataset



class MyTrainer(Trainer):

    def evaluate(
        self,
        eval_dataset: Optional[Dataset] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> Dict[str, float]:
        print("\nevaluate()")
        prime_str = '<s> _cap_ lit++ --wo ! _cap_ oj++ --czyz++ --no mo++ --ja'
        max_length = 100
        ids = self.tokenizer.encode(prime_str, return_tensors="pt")[:, :-1]
        preds = self.model.generate(
            ids.to(self.model.device), 
            max_length=max_length,
            temperature=1.0,
            # num_beams=10, early_stopping=True,
            # no_repeat_ngram_size=1,
            # do_sample=True,
            # top_k=50,
            # top_p=0.92
        )
        print(preds[0])
        print(self.tokenizer.decode(preds[0]))
        return super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)
