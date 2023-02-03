import json
from typing import Iterable, List
import torch.nn as nn

from .const import *


def load_ontology(dataset, ontology_file=None):
    '''
    Read ontology file for event to argument mapping.
    '''
    ontology_dict = {}
    if not ontology_file:  # use the default file path
        if not dataset:
            raise ValueError
        with open('event_role_{}.json'.format(dataset), 'r', encoding='utf-8') as f:
            ontology_dict = json.load(f)
    else:
        with open(ontology_file, 'r', encoding='utf-8') as f:
            ontology_dict = json.load(f)

    for evt_name, evt_dict in ontology_dict.items():
        for i, argname in enumerate(evt_dict['roles']):
            evt_dict['arg{}'.format(i+1)] = argname
            # argname -> role is not a one-to-one mapping
            if argname in evt_dict:
                evt_dict[argname].append('arg{}'.format(i+1))
            else:
                evt_dict[argname] = ['arg{}'.format(i+1)]

    return ontology_dict


def freeze_params(model: nn.Module):
    """Set requires_grad=False for each of model.parameters()"""
    for par in model.parameters():
        par.requires_grad = False


def grad_status(model: nn.Module) -> Iterable:
    return (par.requires_grad for par in model.parameters())


def assert_all_frozen(model):
    model_grads: List[bool] = list(grad_status(model))
    n_require_grad = sum(lmap(int, model_grads))
    npars = len(model_grads)
    assert not any(model_grads), f"{n_require_grad/npars:.1%} of {npars} weights require grad"

