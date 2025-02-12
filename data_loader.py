from datasets import load_dataset
from config import TOKENIZER
import torch
from torch.utils.data import DataLoader

def _tokenize_function(examples):
    tokenized = TOKENIZER(examples['text'], padding="max_length", truncation=True)
    
    # 특정 필드만 float로 변환 (예: 'input_ids')
    if 'input_ids' in tokenized:
        tokenized['input_ids'] = [torch.tensor(ids, dtype=torch.float) for ids in tokenized['input_ids']]
    
    return tokenized


dataset = load_dataset("imdb")
train_dataset = dataset['train']
test_dataset = dataset['test']



tokenized_datasets = dataset.map(_tokenize_function, batched=True)

# 데이터 준비
train_dataset = tokenized_datasets["train"].remove_columns(["text"])
train_dataset = train_dataset.rename_column("label", "labels")
train_dataset.set_format("torch")

test_dataset = tokenized_datasets["test"].remove_columns(["text"])
test_dataset = test_dataset.rename_column("label", "labels")
test_dataset.set_format("torch")

train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=8)
test_dataloader = DataLoader(test_dataset, batch_size=8)