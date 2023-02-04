# TDE-GTEE
This repository provides the source code of our paper "A Hybrid Detection and Generation Framework with Separate Encoders for Event Extraction". (Received by EACL)

Currently we provide a Baseline version of our implementation based on [GTEE-DYNPREF](https://aclanthology.org/2022.acl-long.358.pdf), and the code used in our paper will be coming soon after the paper is finalized for publication.

## How to run

### Preprocessing
All datasets can be placed in data or elsewhere.

#### ACE05-E: DyGIE++ to OneIE format
`prepreocessing/process_dygiepp.py` from OneIE v0.4.8

#### ACE05-E+: ACE2005 to OneIE format
`preprocessing/process_ace.py` from OneIE v0.4.8

#### ERE-EN: ERE to OneIE format
`prepreocessing/process_ere.py` from OneIE v0.4.8


## To Train

To train baseline on different datasets, specify the configurations located in scripts/*.sh

`bash scripts/train_base_generator.sh` and `bash scripts/train_base_IC.sh` can be executed in parallel, but `bash scripts/eval_base_IC.sh` needs to be executed first to get the file `az_file` used for `bash scripts/eval_base_generator.sh`