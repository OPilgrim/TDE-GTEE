import argparse
import json
import string
import os
import re
from collections import defaultdict
import sys
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
import random
from .data import IEDataset, my_collate
from .utils import load_ontology
from .const import MAX_SRC_LENGTH


class DataModule(pl.LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.hparams.update(vars(args))
        self.tokenizer = AutoTokenizer.from_pretrained(args.backbone_model)
    
    def create_golden(self, source, target, split, extend):
        w = 'a' if extend else 'w'
        max_length = 512 if self.hparams.NSP else MAX_SRC_LENGTH[self.hparams.dataset]
        ontology_dict = load_ontology(dataset=self.hparams.dataset)
        ontology = "[SEP]".join(list(ontology_dict.keys()))
        
        assert self.hparams.num_class == len(ontology_dict), "except the classes of events is {}, but recived {}".format(len(ontology_dict), self.hparams.num_class)
                        
        self.event2id = dict()
        for idx, event in enumerate(ontology_dict.keys()):
            self.event2id[event] = idx

        with open(source, 'r', encoding='utf-8') as reader, open(target, w, encoding='utf-8') as writer:
            for lidx, line in enumerate(reader):
                ex = json.loads(line.strip())

                label = [0] * self.hparams.num_class
                event_types = [ex["event_mentions"][i]["event_type"] for i in range(len(ex["event_mentions"]))]
                if len(event_types):
                    for event_type in event_types:
                        label[self.event2id[event_type]] = 1
                
                sentence = ' '.join(ex['tokens']) if self.hparams.dataset == 'ERE' else ex['sentence']
                if self.hparams.NSP:
                    sentence = "{}[SEP]{}".format(ontology, sentence)

                tokenized = self.tokenizer(sentence, 
                                           padding= "max_length",
                                           max_length=max_length,
                                           truncation=True)

                input_ids = tokenized['input_ids']
                attention_mask = tokenized['attention_mask']

                processed_ex = {
                    "doc_key": ex["sent_id"],
                    "sentence": ' '.join(ex['tokens']) if self.hparams.dataset == 'ERE' else ex['sentence'],
                    "evt_type": event_types,
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "label": label
                }
                writer.write(json.dumps(processed_ex, ensure_ascii=False) + '\n')

    def prepare_data(self):
        if self.hparams.tmp_dir:
            data_dir = self.hparams.tmp_dir
        else:
            data_dir = 'preprocessed_{}'.format(self.hparams.dataset)

        if not os.path.exists(data_dir) or self.hparams.refresh:
            print('creating tmp dir ....')
            os.makedirs(data_dir, exist_ok=True)
            
            for split, f in [('train', self.hparams.train_file), ('val', self.hparams.val_file), ('test', self.hparams.test_file)]:
                if (split in ['train', 'val']) and not f:
                    continue
                
                self.create_golden(f, os.path.join(data_dir, '{}.jsonl'.format(split)), split, False)


    def train_dataloader(self):
        if self.hparams.tmp_dir:
            data_dir = self.hparams.tmp_dir
        else:
            data_dir = 'preprocessed_{}'.format(self.hparams.dataset)

        dataset = IEDataset(os.path.join(data_dir, 'train.jsonl'))

        dataloader = DataLoader(
            dataset,
            pin_memory=True, 
            num_workers=1,
            collate_fn=my_collate,
            batch_size=self.hparams.train_batch_size,
            shuffle=True
        )
        return dataloader

    def val_dataloader(self):
        if self.hparams.tmp_dir:
            data_dir = self.hparams.tmp_dir
        else:
            data_dir = 'preprocessed_{}'.format(self.hparams.dataset)

        dataset = IEDataset(os.path.join(data_dir, 'val.jsonl'))

        dataloader = DataLoader(
            dataset, 
            pin_memory=True,
            num_workers=1,
            collate_fn=my_collate,
            batch_size=self.hparams.eval_batch_size,
            shuffle=False
        )
        return dataloader

    def test_dataloader(self):
        if self.hparams.tmp_dir:
            data_dir = self.hparams.tmp_dir
        else:
            data_dir = 'preprocessed_{}'.format(self.hparams.dataset)

        dataset = IEDataset(os.path.join(data_dir, 'test.jsonl'))

        dataloader = DataLoader(
            dataset,
            pin_memory=True,
            num_workers=1,
            collate_fn=my_collate,
            batch_size=self.hparams.eval_batch_size,
            shuffle=False
        )

        return dataloader


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--train-file',
        type=str,
        default="data/kensho/train.oneie.json"
    )
    parser.add_argument(
        '--val-file',
        type=str,
        default="data/kensho/dev.oneie.json"
    )
    parser.add_argument(
        '--test-file',
        type=str,
        default="data/kensho/test.oneie.json"
    )
    parser.add_argument(
        '--tmp_dir',
        default='tmp'
    )
    parser.add_argument(
        '--train_batch_size',
        type=int,
        default=2
    )
    parser.add_argument(
        '--eval_batch_size',
        type=int,
        default=4
    )
    parser.add_argument(
        '--mark-trigger',
        action='store_true',
        default=True
    )
    parser.add_argument(
        '--dataset',
        type=str,
        default='kensho'
    )
    parser.add_argument(
        '--refresh',
        action='store_true',
        default=True
    )
    parser.add_argument(
        '--negative_sampling',
        action='store_true',
        default=True
    )
    parser.add_argument(
        '--negative_ratio',
        type=float,
        default=0.3
    )
    parser.add_argument(
        '--backbone_model',
        type=str,
        default=None
    )
    args = parser.parse_args()


    dm = DataModule(args=args)
    dm.prepare_data()

    # training dataloader
    dataloader = dm.test_dataloader()

    for idx, batch in enumerate(dataloader):
        # print(batch)

        batch_sz = batch["input_ids"].size(0)

        for _ in range(batch_sz):
            src_str = dm.tokenizer.decode(
                batch["input_ids"][_].tolist(),
                # skip_special_tokens=True,
                clean_up_tokenization_spaces=True
            )

            label = batch["label"]

            print("src text is:")
            print(src_str)
            print()
            print("label is:")
            print(label)
            print()