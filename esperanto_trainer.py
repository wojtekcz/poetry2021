from transformers import Trainer, TrainingArguments
from typing import Dict, List, Optional
from torch.utils.data.dataset import Dataset



class EsperantoTrainer(Trainer):

    def evaluate(
        self,
        eval_dataset: Optional[Dataset] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> Dict[str, float]:
        print("\nevaluate()")
        prime_str = 'Eichkorn en tri kajeroj,'
        max_length = 100
        ids = self.tokenizer.encode(prime_str, return_tensors="pt")[:, :-1]
        preds = self.model.generate(ids.to(self.model.device), max_length=max_length)
        print(self.tokenizer.decode(preds[0]))
        return super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)
