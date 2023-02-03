from collections import defaultdict
from copy import deepcopy
from pprint import pprint
import json
from .parser import parse_all_gold, parse_all_pred
from .utils import get_entity_span

'''modified from Text2Event https://github.com/luyaojie/text2event'''


class Metric:
    def __init__(self):
        self.tp = 0.
        self.cor_tp = 0.
        self.gold_num = 0.
        self.pred_num = 0.

    @staticmethod
    def safe_div(a, b):
        if b == 0.:
            return 0.
        else:
            return a / b

    def compute_f1(self, prefix=''):
        tp = self.tp
        cor_tp = self.cor_tp + self.tp
        pred_num = self.pred_num
        gold_num = self.gold_num
        p, r = self.safe_div(tp, pred_num), self.safe_div(tp, gold_num)
        cor_p, cor_r = self.safe_div(cor_tp, pred_num), self.safe_div(cor_tp, gold_num)
        return {
            prefix + 'tp': tp,
            prefix + 'gold': gold_num,
            prefix + 'pred': pred_num,
            prefix + 'P': p * 100,
            prefix + 'R': r * 100,
            prefix + 'F1': self.safe_div(2 * p * r, p + r) * 100,
            prefix + 'cor_P': cor_p * 100,
            prefix + 'cor_R': cor_r * 100,
            prefix + 'cor_F1': self.safe_div(2 * cor_p * cor_r, cor_p + cor_r) * 100
        }

    def count_instance(self, gold_list, pred_list, verbose=False, doc_key=None):
        if verbose:
            print("Gold:", gold_list)
            print("Pred:", pred_list)
        self.gold_num += len(gold_list)
        self.pred_num += len(pred_list)

        dup_gold_list = deepcopy(gold_list)
        for pred in pred_list:
            if pred in dup_gold_list:
                self.tp += 1
                dup_gold_list.remove(pred)

def remove_cls_info_from_trigger_list(trigger_list_cls):
    trigger_list_idn = []
    for trg in trigger_list_cls:
        # only add span [s, t]
        trigger_list_idn.append(trg[1])
    return trigger_list_idn

def remove_cls_info_from_argument_list(argument_list_cls):
    argument_list_idn = []
    for arg in argument_list_cls:
        # only add span [s, t]
        argument_list_idn.append(arg[2])
    return argument_list_idn

def evaluate_parsed_two_group(doc2ex_pred, doc2ex_gold, dataset, dataset_split_name, args):
    trigger_idn_metric = Metric()
    argument_idn_metric = Metric()
    trigger_cls_metric = Metric()
    argument_cls_metric = Metric()

        
    for doc_key, doc_instances_pred in doc2ex_pred.items():
        doc_instances_gold = doc2ex_gold[doc_key]

        #assert doc_instances_gold['text'] == doc_instances_pred['text'], print(doc_key, '\n', doc_instances_gold['text'], '\n', doc_instances_pred['text'])

        gold_trigger_cls_list = doc_instances_gold['trigger_list']
        pred_trigger_cls_list = doc_instances_pred['trigger_list']
        gold_trigger_idn_list = remove_cls_info_from_trigger_list(gold_trigger_cls_list)
        pred_trigger_idn_list = remove_cls_info_from_trigger_list(pred_trigger_cls_list)

        gold_argument_cls_list = doc_instances_gold['argument_list']
        pred_argument_cls_list = doc_instances_pred['argument_list']
        gold_argument_idn_list = remove_cls_info_from_argument_list(gold_argument_cls_list)
        pred_argument_idn_list = remove_cls_info_from_argument_list(pred_argument_cls_list)

        trigger_idn_metric.count_instance(
            gold_list=gold_trigger_idn_list,
            pred_list=pred_trigger_idn_list
        )
        trigger_cls_metric.count_instance(
            gold_list=gold_trigger_cls_list,
            pred_list=pred_trigger_cls_list
        )

        argument_idn_metric.count_instance(
            gold_list=gold_argument_idn_list,
            pred_list=pred_argument_idn_list
        )
        argument_cls_metric.count_instance(
            gold_list=gold_argument_cls_list,
            pred_list=pred_argument_cls_list
        )

    trigger_idn_result = trigger_idn_metric.compute_f1(
        prefix='{}-{}-trig-idn-'.format(dataset, dataset_split_name)
    )
    trigger_cls_result = trigger_cls_metric.compute_f1(
        prefix='{}-{}-trig-cls-'.format(dataset, dataset_split_name)
    )
    role_idn_result = argument_idn_metric.compute_f1(
        prefix='{}-{}-role-idn-'.format(dataset, dataset_split_name)
    )
    role_cls_result = argument_cls_metric.compute_f1(
        prefix='{}-{}-role-cls-'.format(dataset, dataset_split_name)
    )

    pprint(trigger_idn_result)
    pprint(trigger_cls_result)

    return trigger_idn_result, trigger_cls_result, role_idn_result, role_cls_result


def doc_positive_event_type(doc2ex_gold):
    doc2evt_tp = defaultdict(set)
    for doc, content in doc2ex_gold.items():
        for trigger in content['trigger_list']:
            doc2evt_tp[doc].add(trigger[0])
    return doc2evt_tp


def evaluate(doc_keys, evt_types, pred_strs, text_tokens, dataset, dataset_split_name, gold_file_path, args):
    # parse gold token-level event instances
    doc2ex_gold = parse_all_gold(
        gold_file_path,
        args,
    )
    # get gold event type filter for matching pos records first
    doc2evt_tp_gold = doc_positive_event_type(doc2ex_gold)

    # parse predicted token-level event instances
    # and filter unrelated event types
    doc2ex_pred = parse_all_pred(
        args,
        doc_keys,
        evt_types,
        pred_strs,
        text_tokens,
        dataset,
        doc2evt_tp_gold,
    )

    trigger_idn_result, trigger_cls_result, role_idn_result, role_cls_result = evaluate_parsed_two_group(
        doc2ex_pred, 
        doc2ex_gold, 
        dataset, 
        dataset_split_name,
        args,
    )

    return trigger_idn_result, trigger_cls_result, role_idn_result, role_cls_result



def inference(doc_keys, evt_types, pred_strs, text_tokens, dataset, args):
    
    doc2ex_pred = parse_all_pred(
        doc_keys,
        evt_types,
        pred_strs,
        text_tokens,
        dataset,
        only_text=False
    )

    recoders = []
    for doc_key, value in doc2ex_pred.items():
        recoder = {
            "data": {
                "text": value['text']
            },
            "predictions": [{
                "result": []
            }],
        }

        idx = 0
        for trigger in value['trigger_list']:
            t = {
                    "value": {
                        "start": trigger[1][0],
                        "end":trigger[1][1]+1,
                        "text": value['text'][trigger[1][0] : trigger[1][1]+1],
                        "labels": [f'TRG_{trigger[0]}'],
                    },
                    "id": idx,
                    "from_name": "label", 
                    "to_name": "text", 
                    "type": "labels"
                }
            recoder['predictions'][0]['result'].append(t)
            idx += 1

        for argument in value['argument_list']:
            a = {
                    "value": {
                        "start": argument[2][0],
                        "end":argument[2][1]+1,
                        "text": value['text'][argument[2][0] : argument[2][1]+1],
                        "labels": [f'ARG_{argument[0]}_{argument[1]}'],
                    },
                    "id": idx,
                    "from_name": "label",
                    "to_name": "text", 
                    "type": "labels"
                }
            recoder['predictions'][0]['result'].append(a)
            idx += 1

        recoders.append(recoder)
    return recoders
