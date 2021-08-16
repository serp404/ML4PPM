import os
import sys

import pandas as pd

from boosting.classification import get_best_hyperparams, get_classification_result
from boosting.config import FOLDER_PREFIX, MAX_WINDOW_SIZE
from boosting.preprocessing import preprocess_traces
from boosting.utils import print_table_and_result


os.chdir(os.getcwd())


folder_name = sys.argv[1]
dataset_name = sys.argv[2]

PATH = f'{FOLDER_PREFIX}{folder_name}/{dataset_name}'

train = pd.read_csv(f'{PATH}/train.csv')
test = pd.read_csv(f'{PATH}/test.csv')
val = pd.read_csv(f'{PATH}/val.csv')

README_FILE = f'{dataset_name}.md'

with open(README_FILE, 'w') as result_file:
    result_file.truncate()

for window_size in range(1, MAX_WINDOW_SIZE):
    try:
        train = preprocess_traces(train, window_size=window_size)
        test = preprocess_traces(test, window_size=window_size)
        val = preprocess_traces(val, window_size=window_size)

        X_train, y_train = train.drop(columns=['next_activity']), train['next_activity']
        X_test, y_test = test.drop(columns=['next_activity']), test['next_activity']
        X_val, y_val = val.drop(columns=['next_activity']), val['next_activity']

        X_train = X_train.to_numpy()
        y_train = y_train.to_numpy()
        X_val = X_val.to_numpy()
        y_val = y_val.to_numpy()
        X_test = X_test.to_numpy()
        y_test = y_test.to_numpy()

        best_params = get_best_hyperparams(
            X_train, y_train,
            X_test, y_test,
        )

        accuracy = get_classification_result(
            X_train, y_train,
            X_test, y_test,
            params=best_params,
        )

        print_table_and_result(README_FILE, best_params, accuracy, window_size)
    except ValueError:
        break
    except Exception:
        pass
