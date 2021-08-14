import numpy as np
from hyperopt import hp


catboost_space = {
    'depth': hp.quniform('depth', 2, 16, 1),
    'n_estimators': hp.quniform('n_estimators', 80, 124, 1),
    'learning_rate': hp.loguniform('learning_rate', np.log(0.001), np.log(0.5)),
    'l2_leaf_reg': hp.uniform('l2_leaf_reg', 1, 8),
}

CASE_ID_COLUMN = 'Case ID'

MAX_WINDOW_SIZE = 5

FOLDER_PREFIX = '../data/preprocessed_datasets/'