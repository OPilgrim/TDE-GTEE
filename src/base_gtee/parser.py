import json
import re
from collections import defaultdict

from .const import argument_start_token, event_start_token, event_end_token, trigger_special_token
from .utils import *

import en_core_web_sm
nlp = en_core_web_sm.load()
nlp.tokenizer = WhitespaceTokenizer(nlp.vocab)

def clean_span(ex, span):
    tokens = ex['tokens']
    if tokens[span[0]].lower() in {'the', 'an', 'a'}:
        if span[0]!=span[1]:
            return (span[0]+1, span[1])
    return span

def generate_sentence_oneie(filename, args, type_format='subtype'):
    """
    modified from Text2Event https://github.com/luyaojie/text2event

    :return:
        dict:
            trigger_list -> [
                (evt_type1, (start1, end1)), 
                (evt_type2, (start2, end2)), 
                ...
            ]
            role_list -> [
                (evt_type1, role_type1, (start1, end1)), 
                (evt_type2, role_type2, (start2, end2)), 
                ...
            ]
    """

    for line in open(filename):
        instance = json.loads(line)
        doc_key = instance['doc_id'] if args.dataset == 'KAIRO' else instance['sent_id']
        entities = {entity['id']
            : entity for entity in instance['entity_mentions']}

        trigger_list = list()
        role_list = list()

        for event in instance['event_mentions']:
            suptype, subtype = event['event_type'].split(':')

            if type_format == 'subtype':
                event_type = subtype
            elif type_format == 'suptype':
                event_type = suptype
            else:
                event_type = suptype + type_format + subtype

            trigger_list += [(
                event_type,
                (event['trigger']['start'], event['trigger']['end'] - 1)
            )]

            for argument in event['arguments']:
                argument_entity = entities[argument['entity_id']]
                span = (argument_entity['start'], argument_entity['end'] - 1)

                role_list += [(
                    event_type,
                    argument['role'],
                    span,
                )]

        yield doc_key, instance['tokens'], trigger_list, role_list


def parse_all_gold(gold_file_path, args):
    doc2ex_gold = defaultdict(list)

    for doc_key, tokens, gold_trigger_list, gold_argument_list in generate_sentence_oneie(
        gold_file_path,
        args,
        type_format=":",
    ):
        doc2ex_gold[doc_key] = {
            'text': ' '.join(tokens),
            'trigger_list': gold_trigger_list,
            'argument_list': gold_argument_list
        }

    return doc2ex_gold


def extract_args_from_template(pred, evt_type, template, ontology_dict,):
    """
    modified from BARTGen https://github.com/raspberryice/gen-arg

    :return:
        dict:
            predicted_args -> [
                (evt_type, role_type, [token1, token2, ...]), ...
            ]
    """
    template = re.split(r'<(arg\d+)>', template)
    new_template = ''
    for tem in template:
        if "arg" == tem[:3] and tem[3:].isdigit():
            new_template += " <{}> ".format(tem)
        else:
            new_template += ''.join(tem)
    # extract argument text
    template_words = new_template.strip().split()
    predicted_words = pred.strip().split()
    
    # each argname may have multiple participants
    predicted_args = []
    t_ptr = 0
    p_ptr = 0
    while t_ptr < len(template_words) and p_ptr < len(predicted_words):
        if re.match(r'<(arg\d+)>', template_words[t_ptr]):
            m = re.match(r'<(arg\d+)>', template_words[t_ptr])
            arg_num = m.group(1)
            try:
                arg_name = ontology_dict[evt_type][arg_num]
            except KeyError:
                print(evt_type)
                exit()

            if predicted_words[p_ptr] == '<arg>':
                # missing argument
                p_ptr += 1
                t_ptr += 1
            else:
                arg_start = p_ptr
                nxt_arg_ptr = t_ptr+1
                while nxt_arg_ptr < len(template_words):
                    if re.match(r'<(arg\d+)>', template_words[nxt_arg_ptr]):
                        break
                    nxt_arg_ptr += 1
                while (p_ptr < len(predicted_words)):
                    if (t_ptr == len(template_words)-1) or (predicted_words[p_ptr] != template_words[t_ptr+1]):
                        p_ptr += 1
                    elif predicted_words[p_ptr] == template_words[t_ptr+1]:
                        t_spans = ' '.join(template_words[t_ptr+1 : nxt_arg_ptr])
                        p_spans = ' '.join(predicted_words[p_ptr : p_ptr+(nxt_arg_ptr-t_ptr-1)])
                        if t_spans != p_spans:
                            p_ptr += 1
                        else:
                            break
                arg_text = predicted_words[arg_start:p_ptr]
                predicted_args.append((evt_type, arg_name, arg_text))
                t_ptr += 1
                # aligned
        else:
            t_ptr += 1
            p_ptr += 1

    return predicted_args


def parse_once(content, evt_type, ontology_dict):
    '''
    content is like:
    Trigger antiwar  <argument>  <arg>  attacked  <arg>  hurting  <arg>  victims using  <arg>  instrument at  <arg>  place  <event>  Trigger war  <argument>  <arg>  attacked  <arg>  hurting  <arg>  victims using  <arg>  instrument at  <arg>  place

    :return:
        dict:
            once_instance -> [
                {
                    'evt_type': evt_type,
                    'trg': [token1, token2, ...],
                    'args': predicted_args from **extract_args_from_template**
                },
                ...
            ]
    '''
    once_instance = []

    events = []
    for e in re.split(f'{event_start_token}|{event_end_token}', content):
        if e != '' and e != ' ':
            events.append(e)
    # Trigger antiwar  <argument>  <arg>  attacked  <arg>  hurting  <arg>  victims using  <arg>  instrument at  <arg>  place
    for event in events:
        parts = event.strip().split(argument_start_token)

        # Trigger antiwar
        trigger_content = parts[0].strip()
        # ['antiwar']
        trigger_tokens = []
        for tk in trigger_content.split()[1:]:
            if tk != trigger_special_token.strip():
                trigger_tokens.append(tk)
        # skip if only <tgr>
        if len(trigger_tokens) == 0:
            continue
        # skip if not a completed event
        if len(parts) < 2:
            continue

        # <arg>  attacked  <arg>  hurting  <arg>  victims using  <arg>  instrument at  <arg>  place
        argument_content = parts[1].strip()
        predicted_args = extract_args_from_template(
            argument_content,
            evt_type,
            ontology_dict[evt_type]['template'],
            ontology_dict
        )

        once_instance.append({
            'evt_type': evt_type,
            'trg': trigger_tokens,
            'args': predicted_args
        })

    return once_instance


def match_sublist(the_list, to_match):
    """
    taken from Text2Event https://github.com/luyaojie/text2event

    :param the_list: [1, 2, 3, 4, 5, 6, 1, 2, 4, 5]
    :param to_match: [1, 2]
    :return:
        [(0, 1), (6, 7)]
    """
    len_to_match = len(to_match)
    matched_list = list()
    for index in range(len(the_list) - len_to_match + 1):
        if to_match == the_list[index:index + len_to_match]:
            matched_list += [(index, index + len_to_match - 1)]
    return matched_list


def record_to_offset(args, record_list, token_list):
    """
    modified from 
    1) Text2Event https://github.com/luyaojie/text2event
    2) BARTGen https://github.com/raspberryice/gen-arg

    Find Role's offset using closest matched with trigger work.
    Please note that this function is match-based without replacement, 
    so the order of the instances may affect the results.
    We will match pos event recoreds first.

    ***This is the final output of aligned triggers and arguments***
    :param instance:
    :return:
    """
    trigger_list = []
    role_list = []
    doc = None 
    if args.head_only:
        doc = nlp(' '.join(token_list))

    trigger_matched_set = set()
    for record in record_list:
        event_type = record['evt_type']
        trigger = record['trg']
        matched_list = match_sublist(token_list, trigger)

        trigger_offset = None
        for matched in matched_list:
            if matched not in trigger_matched_set:
                trigger_list += [(event_type, matched)]
                trigger_offset = matched
                trigger_matched_set.add(matched)
                break

        # No trigger word, skip the record
        if trigger_offset is None:
            continue
        
        for _, role_type, entity in record['args']:
            arg_span = find_arg_span(
                arg=entity,
                context_words=token_list,
                trigger_start=trigger_offset[0],
                trigger_end=trigger_offset[1],
                head_only=args.head_only, 
                doc=doc,
            )
            if arg_span:  # if None means hullucination
                role_list.append((
                    event_type,
                    role_type,
                    (arg_span[0], arg_span[1])
                ))
            else:
                # try to split by 'and'
                new_entity = []
                for w in entity:
                    if w == 'and' and len(new_entity) > 0:
                        # try new entity
                        arg_span = find_arg_span(
                            arg=new_entity,
                            context_words=token_list,
                            trigger_start=trigger_offset[0],
                            trigger_end=trigger_offset[1],
                            head_only=args.head_only, 
                            doc=doc,
                        )
                        if arg_span:
                            role_list.append((
                                event_type,
                                role_type,
                                (arg_span[0], arg_span[1])
                            ))
                        new_entity = []
                    else:
                        new_entity.append(w)

                if len(new_entity) > 0:  # last entity
                    arg_span = find_arg_span(
                        arg=new_entity,
                        context_words=token_list,
                        trigger_start=trigger_offset[0],
                        trigger_end=trigger_offset[1],
                        head_only=args.head_only, 
                        doc=doc,
                    )
                    if arg_span:
                        role_list.append((
                            event_type,
                            role_type,
                            (arg_span[0], arg_span[1])
                        ))
    return trigger_list, role_list


def parse_all_pred(args, doc_keys, evt_types, pred_strs, text_tokens, dataset, filter=None, only_text=False):
    ontology_dict = load_ontology(dataset=dataset)

    doc2ex_pos_pred = defaultdict(list)
    doc2ex_neg_pred = defaultdict(list)
    doc2tokens = {}
    doc_ket_set = set()
    for doc_key, evt_type, pred, tokens in zip(doc_keys, evt_types, pred_strs, text_tokens):

        # remove multiple spaces
        pred = clean_spaces(pred)

        once_instance = parse_once(
            pred,
            evt_type,
            ontology_dict
        )
       
        # store pos and neg separatelly
        # if not split them, the final trigger / argument match numbers 
        # (X-test-trig-tp and X-test-role-tp) will diff
        # when evaluating with less negative event types
        if filter and doc_key in filter:
            if evt_type not in filter[doc_key]:
                doc2ex_neg_pred[doc_key].extend(once_instance)
            else:
                doc2ex_pos_pred[doc_key].extend(once_instance)
        else:
            doc2ex_pos_pred[doc_key].extend(once_instance)

        # save all doc_key in case not completed list when iterating only pos or neg
        doc_ket_set.add(doc_key)
        if doc_key not in doc2tokens:
            doc2tokens[doc_key] = tokens

    if only_text:
        return doc2ex_pos_pred

    # output dict like **parse_all_gold**
    doc2ex_pred_ = {}
    for doc_key in doc_ket_set:
        # record_to_offset is match-based without replacement, so first match pos
        records = doc2ex_pos_pred[doc_key] + doc2ex_neg_pred[doc_key]
        tokens = doc2tokens[doc_key]
        trigger_list, role_list = record_to_offset(args, records, tokens)
        doc2ex_pred_[doc_key] = {
            'text': ' '.join(tokens),
            'trigger_list': trigger_list,
            'argument_list': role_list
        }

    return doc2ex_pred_
