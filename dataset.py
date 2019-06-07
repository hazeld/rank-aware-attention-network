import torch.utils.data as data

import os
import numpy as np

class FeatureRecord(object):
    def __init__(self, row):
        self._data = row

    @property
    def path_better(self):
        return self._data[0]

    @property
    def path_worse(self):
        return self._data[1]

class SkillDataSet(data.Dataset):
    def __init__(self, root_path, list_file, ftr_tmpl='{}_{}.npz'):

        self.root_path = root_path
        self.list_file = list_file
        self.ftr_tmpl = ftr_tmpl

        self._parse_list()

    def _load_features(self, vid):
        features = np.load(os.path.join(self.root_path, self.ftr_tmpl.format(vid,'rgb')))['arr_0'].astype(np.float32)
        return features

    def _parse_list(self):
        self.pair_list = [FeatureRecord(x.strip().split(' ')) for x in open(self.list_file)]

    def __getitem__(self, index):
        record = self.pair_list[index]
        return self.get(record)

    def get(self, record):
        vid1 = self._load_features(record.path_better)
        vid2 = self._load_features(record.path_worse)
        return vid1, vid2

    def __len__(self):
        return len(self.pair_list)


class FeatureRecordSingle(object):
    def __init__(self, row):
        self._data = row

    @property
    def path(self):
        return self._data[0]

class SkillDataSetSingle(SkillDataSet):
    def _parse_list(self):
        self.pair_list = [FeatureRecordSingle(x.strip().split(' ')) for x in open(self.list_file)]


    def get(self, record):
        vid = self._load_features(record.path)

        name = record.path
        return name, vid
