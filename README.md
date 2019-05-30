# Rank-aware Attention Network
Rank-aware Attention Network from 'The Pros and Cons: Rank-aware Temporal Attention for Skill Determination in Long Videos'.

## BEST Dataset
### Videos
The Bristol Everyday Skill Tasks (BEST) Dataset can be downloaded using `python utils/download_videos.py data/BEST/BEST.csv <download_dir> --trim`

The `--trim` flag extracts the part of the original video used in training/testing i.e. removing title sequences or unrelated tasks preceeding the start of the relevant task.

### Features
The extracted i3d features for the BEST tasks and the EPIC-Skills tasks can be downloaded using `bash utils/download_features.sh`

### Training/Test Splits
The split of training and test videos can be found under `data/BEST/splits/<task_name>/<train|test>_vid_list.txt`. The files containing the annotated pairs used for training and testing can be found at `data/BEST/splits/<task_name>/<train|test>.txt`. 

We also include the EPIC-Skills training and testing pair files in the same format under `data/EPIC-Skills/splits/<task_name>/<train|test>_split<split_num>.txt`


Code coming soon.
