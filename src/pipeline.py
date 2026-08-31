from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler

from config.model import NaiveClassifier


def build_pipeline(model, num_cols):
    '''Создает пайплайн предварительной обработки данных'''
    # Для наивного подхода возвращает пустой пайплайн
    if isinstance(model, NaiveClassifier):
        empty_pipeline = Pipeline(steps=[
            ('model', model)
        ])
        return empty_pipeline

    preprocessor = ColumnTransformer(transformers=[
        ('scaler', StandardScaler(), num_cols),
    ])
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', model),
    ])
    return pipeline