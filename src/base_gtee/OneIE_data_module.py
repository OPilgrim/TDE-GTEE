import argparse
import json
import os
import re
from collections import defaultdict
from json import load
import sys
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from transformers import BartTokenizer
import random
from .data import *
from .utils import *
from .const import *


class OneIEDataModule(pl.LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.hparams.update(vars(args))
        self.tokenizer = BartTokenizer.from_pretrained(args.backbone_model)
        self.tokenizer.add_tokens(special_tokens, special_tokens=True)

    def create_gold_gen(self, ex, evt_type, ontology_dict, idx_group):
        '''
        Input: <s> Template with special <arg> placeholders </s> </s> Passage </s>
        Output: <s> Template with arguments and <arg> when no argument is found </s> ...
        '''
        # The trigger is <tgr> <argument> <arg> did <arg> in <arg> place
        template = get_template_with_trigger_slot(
            ontology_dict[evt_type]['template']
        )
        input_template = re.sub(r'<arg\d>', argument_special_token.strip(), template)
        space_tokenized_input_template = input_template.split()

        tokenized_input_template = []
        for w in space_tokenized_input_template:
            tokenized_input_template.extend(
                self.tokenizer.tokenize(w, add_prefix_space=True)
            )
        
        context_words = ex['tokens']
        # The event type is Marry.
        content_prefix = get_template_prefix_with_event_type(
            evt_type
        )

        context = self.tokenizer.tokenize(
            content_prefix + ' ' + ' '.join(context_words),
            add_prefix_space=True
        )

        if self.hparams.inference:
            return tokenized_input_template, [], context

        tokenized_output_templates = []

        for index in idx_group:
            assert ex['event_mentions'][index]['event_type'] == evt_type

            output_template = get_template_with_trigger_slot(
                ontology_dict[evt_type]['template']
            )

            role2arg = defaultdict(list)
            for argument in ex['event_mentions'][index]['arguments']:
                role2arg[argument['role']].append(argument)

            role2arg = dict(role2arg)
            arg_idx2text = defaultdict(list)
            for role in role2arg.keys():
                if role not in ontology_dict[evt_type]:
                    # annotation error
                    continue
                for i, argument in enumerate(role2arg[role]):
                    arg_text = argument['text']
                    if i < len(ontology_dict[evt_type][role]):
                        # enough slots to fill in
                        arg_idx = ontology_dict[evt_type][role][i]
                    else:
                        # multiple participants for the same role
                        arg_idx = ontology_dict[evt_type][role][-1]
                    arg_idx2text[arg_idx].append(arg_text)

            for arg_idx, text_list in arg_idx2text.items():
                text = ' and '.join(text_list)
                output_template = re.sub('<{}>'.format(arg_idx), text, output_template)

            tgt_text = ex['event_mentions'][index]['trigger']['text']
            output_template = re.sub(
                '{}'.format(trigger_special_token.strip()),
                tgt_text,
                output_template
            )

            output_template = re.sub(r'<arg\d>', argument_special_token.strip(), output_template)
            space_tokenized_template = output_template.split()
            tokenized_output_template = []
            for w in space_tokenized_template:
                tokenized_output_template.extend(
                    self.tokenizer.tokenize(w, add_prefix_space=True)
                )
            tokenized_output_templates.append(tokenized_output_template)

        if len(tokenized_output_templates) == 0:
            not_an_event_template = get_not_an_event_template()
            space_tokenized_template = not_an_event_template.split()
            tokenized_output_template = []
            for w in space_tokenized_template:
                tokenized_output_template.extend(
                    self.tokenizer.tokenize(w, add_prefix_space=True)
                )
            tokenized_output_templates.append(tokenized_output_template)

        return tokenized_input_template, tokenized_output_templates, context

    def prepare_data(self):
        if self.hparams.tmp_dir:
            data_dir = self.hparams.tmp_dir
        else:
            data_dir = 'preprocessed_{}'.format(self.hparams.dataset)
        
        ontology_dict = load_ontology(dataset=self.hparams.dataset)

        if self.hparams.az_file:
            az_file_path = self.hparams.az_file
            
            id2event = dict()
            for idx, event in enumerate(ontology_dict.keys()):
                id2event[idx] = event
            doc_event = defaultdict(set)
            with open(az_file_path, 'r') as fin:
                for line in fin:
                    ss = json.loads(line)
                    es = [i for i,x in enumerate(ss['label']) if x==1]
                    for e in es:
                        doc_event[ss['doc_key']].add(id2event[e])

        if not os.path.exists(data_dir) or self.hparams.refresh:
            print('creating tmp dir ....')
            os.makedirs(data_dir, exist_ok=True)

            max_tokens = 0
            max_tgt = 0
            max_enty = 0

            for split, f in [('train', self.hparams.train_file), ('val', self.hparams.val_file), ('test', self.hparams.test_file)]:
                if (split in ['train', 'val']) and not f:
                    continue
                with open(f, 'r') as reader,  open(os.path.join(data_dir, '{}.jsonl'.format(split)), 'w') as writer:
                    for lidx, line in enumerate(reader):
                        ex = json.loads(line.strip())
                        
                        evt_type2instances = defaultdict(list)
                        for i in range(len(ex['event_mentions'])):
                            evt_type = ex['event_mentions'][i]['event_type']

                            # bad,  only use event_type which in ontology 
                            if evt_type not in ontology_dict:
                                print(f"{evt_type} not in ontology_dict...")
                                sys.exit(-1)

                            evt_type2instances[evt_type].append(i)
                        
                        for evt_type in ontology_dict.keys():
                            if split in ['train', 'val']:
                                if evt_type not in evt_type2instances:
                                    if not self.hparams.negative_sampling:
                                        continue
                                    # not adding negative samples in train and val only if allowed or sampled prob > threshold
                                    ratio = float(self.hparams.negative_ratio)/(len(ontology_dict.keys()))
                                    if random.random() > ratio:
                                        continue

                            if split in ['test']:
                                if self.hparams.az_file and not evt_type in doc_event[ex['sent_id']]:
                                    continue
                            
                            idx_group = evt_type2instances[evt_type] if not self.hparams.inference else None

                            # add positive samples normally
                            input_template, output_templates, context = self.create_gold_gen(
                                ex,
                                evt_type,
                                ontology_dict,
                                idx_group=idx_group
                            )

                            output = []
                            for idx, output_template in enumerate(output_templates):
                                if idx > 0:
                                    output.append(event_start_token)
                                output.extend(output_template)

                            max_tokens = max(
                                len(context) + len(input_template) + 4, 
                                max_tokens
                            )
                            max_tgt = max(len(output) + 2 , max_tgt)
                            
                            assert len(output) + 2 <= MAX_TGT_LENGTH[self.hparams.dataset], "{} : {}".format(len(output), output)
                            
                            input_tokens = self.tokenizer.encode_plus(
                                input_template, 
                                context,
                                add_special_tokens=True,
                                add_prefix_space=True,
                                max_length=MAX_SRC_LENGTH[self.hparams.dataset],
                                truncation='only_second',
                                padding='max_length'
                            )

                            if not self.hparams.inference:
                                tgt_tokens = self.tokenizer.encode_plus(
                                    output,
                                    add_special_tokens=True,
                                    add_prefix_space=True,
                                    max_length=MAX_TGT_LENGTH[self.hparams.dataset],
                                    truncation=True,
                                    padding='max_length'
                                )

                                processed_ex = {
                                    'doc_key': ex['sent_id'],
                                    'tokens': ex['tokens'],
                                    'evt_type': evt_type,
                                    'input_token_ids': input_tokens['input_ids'],
                                    'input_attn_mask': input_tokens['attention_mask'],
                                    'tgt_token_ids': tgt_tokens['input_ids'],
                                    'tgt_attn_mask': tgt_tokens['attention_mask']
                                }
                            else:
                                processed_ex = {
                                    'doc_key': ex['sent_id'],
                                    'tokens': ex['tokens'],
                                    'evt_type': evt_type,
                                    'input_token_ids': input_tokens['input_ids'],
                                    'input_attn_mask': input_tokens['attention_mask'],
                                    'tgt_token_ids': [],
                                    'tgt_attn_mask': []
                                }

                            writer.write(json.dumps(processed_ex) + '\n')

            print('longest context:{}'.format(max_tokens))
            print('longest target {}'.format(max_tgt))
            print('max entity number {}'.format(max_enty))

    def train_dataloader(self):
        if self.hparams.tmp_dir:
            data_dir = self.hparams.tmp_dir
        else:
            data_dir = 'preprocessed_{}'.format(self.hparams.dataset)

        dataset = IEDataset(os.path.join(data_dir, 'train.jsonl'))

        dataloader = DataLoader(
            dataset,
            pin_memory=True, 
            num_workers=8,
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
            num_workers=8,
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
            num_workers=8,
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
        default="data/ERE-EN/train.oneie.json"
    )
    parser.add_argument(
        '--val-file',
        type=str,
        default="data/ERE-EN/dev.oneie.json"
    )
    parser.add_argument(
        '--test-file',
        type=str,
        default="data/ERE-EN/test.oneie.json"
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
        default='ERE'
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
    args = parser.parse_args()

    dm = OneIEDataModule(args=args)
    dm.prepare_data()

    # training dataloader
    dataloader = dm.test_dataloader()

    for idx, batch in enumerate(dataloader):
        # print(batch)

        batch_sz = batch["input_token_ids"].size(0)

        for _ in range(batch_sz):
            src_str = dm.tokenizer.decode(
                batch["input_token_ids"][_].tolist(),
                # skip_special_tokens=True,
                clean_up_tokenization_spaces=True
            )

            tgt_str = dm.tokenizer.decode(
                batch["tgt_token_ids"][_].tolist(),
                # skip_special_tokens=True,
                clean_up_tokenization_spaces=True
            )

            if event_start_token in tgt_str:
                print("src text is:")
                print(src_str)
                print()
                print("tgt text is:")
                print(tgt_str)
                print()
