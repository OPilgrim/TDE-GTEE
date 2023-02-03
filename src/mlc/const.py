trigger_special_token = '<tgr>'
argument_special_token = '<arg>'

event_start_token = '<event>'
event_end_token = '</event>'

trigger_start_token = '<trigger>'
trigger_end_token = '</trigger>'

argument_start_token = '<argument>'
argument_end_token = '</argument>'

special_tokens = [
    argument_special_token,
    trigger_special_token,
    event_start_token,
    event_end_token,
    trigger_start_token,
    argument_start_token,
]


MAX_SRC_LENGTH = {
    'wikievents': 512,
    'ACE': 512,
    'ERE':512,
}
MAX_TGT_LENGTH = {
    'wikievents': 512,
    'ACE': 512,
    'ERE':512
}


NUM_BEAM = 1