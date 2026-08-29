from sklearn.base import BaseEstimator, ClassifierMixin

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier


class NaiveClassifier(BaseEstimator, ClassifierMixin):
    '''Если ускорение по оси Y больше 0 - значит бег'''
    def fit(self, X, y=None):
        return self

    def predict(self, X):
        return (X['acceleration_y'] > 0).astype(int)


def get_model_dict(random_state):
    return {
        "naive": NaiveClassifier(),
        "linear": LogisticRegression(),
        "knn": KNeighborsClassifier(),
        "svm": SVC(random_state=random_state),
        "random_forest": RandomForestClassifier(
            random_state=random_state,
            n_jobs=-1,
        ),
        "xgboost": XGBClassifier(
            random_state=random_state,
            verbosity=0,
            n_jobs=-1,
        ),
        "lightgbm": LGBMClassifier(
            random_state=random_state,
            verbosity=-1,
            n_jobs=-1,
        ),
        "catboost": CatBoostClassifier(
            random_state=random_state,
            verbose=0,
            thread_count=-1,
        ),
    }