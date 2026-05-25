from dataclasses import dataclass
from typing import Any, Dict, List

import torch


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        input_features = [f["input_features"] for f in features]
        labels = [f["labels"] for f in features]

        batch = {
            "input_features": torch.tensor(input_features, dtype=torch.float32)
        }

        labels_batch = self.processor.tokenizer.pad(
            {"input_ids": labels},
            return_tensors="pt"
        )

        labels = labels_batch["input_ids"]

        # pad token -> -100
        labels[labels == self.processor.tokenizer.pad_token_id] = -100

        # bos 제거
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch
