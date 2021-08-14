from collections import deque

import pandas as pd
from tqdm import tqdm

from boosting.config import CASE_ID_COLUMN


def preprocess_traces(df, window_size=1):
    data = df.copy()
    min_size = data.shape[0]

    for case_id in tqdm(data[CASE_ID_COLUMN].unique()):
        case_data = data[data[CASE_ID_COLUMN] == case_id]
        min_size = min(min_size, case_data.shape[0])

    if window_size > min_size - 1:
        raise ValueError('preprocess_traces -> window_size > min_size - 1')

    new_rows = []
    if 'Unnamed: 0' in data.columns:
        data.drop(columns=['Unnamed: 0'], inplace=True)

    new_shape = (data.shape[0] - window_size + 1, window_size * data.shape[1])
    new_columns = [f'col_{i}' for i in range(new_shape[1])] + ['next_activity']

    for case_id in tqdm(data[CASE_ID_COLUMN].unique()):
        window = deque()
        case_data = data[data[CASE_ID_COLUMN] == case_id].reset_index()
        case_data.drop(columns=['Case ID'], inplace=True)

        for i in range(window_size):
            window.append(case_data.T[i].to_list())
        current = []
        for row in window:
            current.extend(row)
        current.append(case_data.loc[window_size, 'Activity ID'])
        new_rows.append(current[:])

        for i in range(window_size, case_data.shape[0] - 1):
            window.popleft()
            window.append(case_data.T[i].to_list())
            current = []
            for row in window:
                current.extend(row)
            current.append(case_data.loc[i + 1, 'Activity ID'])
            new_rows.append(current[:])

    return pd.DataFrame(new_rows, columns=new_columns)
