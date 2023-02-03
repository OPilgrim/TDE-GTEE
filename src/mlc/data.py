import os 
import json 

import torch 
from torch.utils.data import DataLoader, Dataset

def my_collate(batch):

    doc_keys = [ex['doc_key'] for ex in batch]
    sentence = [ex['sentence'] for ex in batch]
    evt_types = [ex['evt_type'] for ex in batch]
    input_ids = torch.stack([torch.LongTensor(ex["input_ids"]) for ex in batch])
    attention_mask = torch.stack([torch.BoolTensor(ex["attention_mask"]) for ex in batch])
    label = torch.FloatTensor([ex["label"] for ex in batch])

    return {
        'doc_key': doc_keys,
        'sentence': sentence,
        'evt_type': evt_types,
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'label': label
    }


class IEDataset(Dataset):
    def __init__(self, input_file):
        super().__init__()
        self.examples = []
        with open(input_file, 'r', encoding='utf-8') as f:
            for line in f:
                ex = json.loads(line.strip())
                self.examples.append(ex)
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        return self.examples[idx]
    

