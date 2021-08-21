import torch
import numpy as np
import pandas as pd
import ast
import torch.nn as nn
import torch.nn.functional as F
from tqdm.notebook import tqdm


def ListMerge2d(list1, list2):
    assert len(list1) == len(list2), "Lists are not of equal length!"
    return [l1+ l2 for l1, l2 in zip(list1, list2)]


class LogsDataset(torch.utils.data.Dataset):
    def __init__(self, act_prefix, cts_prefix, act_next, cts_next):
        self.emb_prefix = [ListMerge2d(l1, l2) for l1, l2 in zip(act_prefix, cts_prefix)]
        self.act_next = act_next
        self.cts_next = cts_next

    def __len__(self):
        return len(self.act_next)
    
    def __getitem__(self, idx):
        sample = {
            'prefix': self.emb_prefix[idx],
            'act_next': self.act_next[idx],
            'cts_next': self.cts_next[idx]
        }
        return sample


def pad_collate(batch):
    prefix_batch = [torch.as_tensor(trace['prefix']) for trace in batch]
    activity_next = torch.tensor([trace['act_next'] for trace in batch])
    timestamp_next = torch.tensor([trace['cts_next'] for trace in batch])

    prefix_pad = nn.utils.rnn.pad_sequence(prefix_batch, batch_first=True, padding_value=0)

    result = {
        'prefix': prefix_pad,
        'act_next': activity_next,
        'cts_next': timestamp_next
    }
    return result


def date2ts(date: str):
    return pd.Timestamp(date).value // 1000000000


def ParseDatelist(datelist: str, applyer=date2ts):
    return list(map(applyer, ast.literal_eval(datelist)))


def ParseActivitylist(activitylist: str):
    return ast.literal_eval(activitylist)


def LogLoader(data, batch_size, shuffle):
    return torch.utils.data.DataLoader(
            LogsDataset(
                data['act_prefix'].tolist(),
                data['cts_prefix'].tolist(),
                data['act_next'].tolist(),
                data['cts_next'].tolist()
            ),
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=pad_collate
    )


def GenTimeFeatures(timestamp: str, ts_applyer=None):
    ts = pd.Timestamp(timestamp)

    value = ts.value
    if ts_applyer is not None:
        value = ts_applyer(ts.value)

    weekday = [0] * 7
    weekday[ts.dayofweek] = 1

    intervals = [
        1 if 23 < ts.hour <= 5 else 0,
        1 if 5 < ts.hour <= 12 else 0,
        1 if 12 < ts.hour <= 18 else 0,
        1 if 18 < ts.hour <= 23 else 0
    ]

    additional = [
        1 if ts.is_year_end else 0,
        1 if ts.is_year_start else 0,
        1 if ts.is_month_end else 0,
        1 if ts.is_month_start else 0
    ]

    return [value] + weekday + intervals + additional


def OneHotEncode(labels: list, num_classes=16):
    return F.one_hot(torch.tensor(labels) - 1, num_classes=num_classes).tolist()


def FocalLoss(logits, labels, gamma=0.):
    batch_size, n_classes = logits.shape
    ohe_labels = F.one_hot(labels, num_classes=n_classes)
    probs = torch.softmax(logits, dim=1)
    scale_factors = (1 - probs)**gamma
    return -torch.sum(scale_factors * torch.log(probs) * ohe_labels) / batch_size
