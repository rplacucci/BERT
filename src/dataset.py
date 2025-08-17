import random
import torch
from torch.utils.data import Dataset

class WikipediaDataset(Dataset):
    def __init__(self, dataset, tokenizer, max_len):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        example = self.dataset[index]
        
        sent_a = example["sentence"]
        is_last = example["is_last"]

        if random.random() < 0.5 and not is_last and (index < len(self.dataset) - 1):
            sent_b = self.dataset[index + 1]["sentence"]
            is_next = 1
        else:
            sent_b = random.choice(self.dataset)["sentence"]
            is_next = 0

        # Tokenize sentences
        encoded = self.tokenizer(
            sent_a,
            sent_b,
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_overflowing_tokens=False,
            return_special_tokens_mask=True,
            return_token_type_ids=True,
            return_attention_mask=True
        )

        # Convert to tensors
        input_ids = torch.tensor(encoded['input_ids'], dtype=torch.long)
        segment_ids = torch.tensor(encoded['token_type_ids'], dtype=torch.long)
        attention_mask = torch.tensor(encoded['attention_mask'], dtype=torch.long)
        special_tokens_mask = torch.tensor(encoded['special_tokens_mask'], dtype=torch.bool)

        # Apply masking
        masked_ids, mlm_labels = self.mask_tokens(input_ids, special_tokens_mask)

        # Create next-sentence prediction label
        nsp_label = torch.tensor(is_next, dtype=torch.long)

        return {
            "input_ids": masked_ids,
            "segment_ids": segment_ids,
            "attention_mask": attention_mask,
            "mlm_labels": mlm_labels,
            "nsp_label": nsp_label
        }

    def mask_tokens(self, input_ids, special_tokens_mask):
        masked_ids = []
        labels = []
        for token_id, is_special in zip(input_ids.tolist(), special_tokens_mask.tolist()):
            # Skip masking for special tokens or by probability
            if is_special or random.random() > 0.15:
                masked_ids.append(token_id)
                labels.append(-100)
            else:
                prob = random.random()
                if prob < 0.8:
                    masked_ids.append(self.tokenizer.mask_token_id)
                elif prob < 0.9:
                    masked_ids.append(random.randrange(self.tokenizer.vocab_size))
                else:
                    masked_ids.append(token_id)
                labels.append(token_id)

        return torch.tensor(masked_ids, dtype=torch.long), torch.tensor(labels, dtype=torch.long)

class GLUEDataset(Dataset):
    def __init__(
        self,
        dataset,
        tokenizer,
        max_len,
        task_name,
    ):
        super().__init__()
        assert task_name in ["ax", "mnli", "qqp", "qnli", "sst2", "cola", "stsb", "mrpc", "rte", "wnli"], "task_name not a GLUE task"

        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.task_name = task_name

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        example = self.dataset[index]
        idx = example["idx"]
        label = example["label"]

        if self.task_name in ("ax", "mnli"):
            sent_a, sent_b = example["premise"], example["hypothesis"]
        elif self.task_name == "qqp":
            sent_a, sent_b = example["question1"], example["question2"]
        elif self.task_name == "qnli":
            sent_a, sent_b = example["question"], example["sentence"]
        elif self.task_name in ("sst2", "cola"):
            sent_a, sent_b = example["sentence"], None
        else:
            sent_a, sent_b = example["sentence1"], example["sentence2"]

        if sent_b is None:
            encoded = self.tokenizer(
                sent_a,
                padding="max_length",
                truncation=True,
                max_length=self.max_len,
                return_token_type_ids=True,
                return_attention_mask=True
            )
        else:
            encoded = self.tokenizer(
                sent_a,
                sent_b,
                padding="max_length",
                truncation=True,
                max_length=self.max_len,
                return_token_type_ids=True,
                return_attention_mask=True
            )

        input_ids = torch.tensor(encoded["input_ids"], dtype=torch.long)
        segment_ids = torch.tensor(encoded['token_type_ids'], dtype=torch.long)
        attention_mask = torch.tensor(encoded['attention_mask'], dtype=torch.long)
        label = torch.tensor(label, dtype=torch.float) if self.task_name == "stsb" else torch.tensor(label, dtype=torch.long)

        return {
            "idx": idx,
            "input_ids": input_ids,
            "segment_ids": segment_ids,
            "attention_mask": attention_mask,
            "label": label,
        }