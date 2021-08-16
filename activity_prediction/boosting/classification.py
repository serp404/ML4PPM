from catboost import CatBoostClassifier
from hyperopt import STATUS_OK, fmin, tpe
from sklearn.metrics import accuracy_score

from boosting.config import catboost_space


def get_best_hyperparams(X_train, y_train, X_val, y_val, classifier_name='catboost'):
    def fake_function(params):
        classifier = CatBoostClassifier(**params)
        return {
            'loss': (
                1 - accuracy_score(
                    y_val, classifier.fit(X_train, y_train).predict(X_val)
                )
            ),
            'status': STATUS_OK,
        }

    space = catboost_space

    if not space:
        raise ValueError('Unresolved classifier name')

    best = fmin(fn=fake_function, space=space, algo=tpe.suggest, max_evals=30, verbose=False)
    return best


def get_classification_result(
    X_train, y_train,
    X_test, y_test,
    method='catboost',
    params=None,
):

    classifier = CatBoostClassifier(**params)

    accuracy = accuracy_score(y_test, classifier.fit(X_train, y_train).predict(X_test))

    return accuracy
