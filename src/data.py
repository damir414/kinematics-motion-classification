import pandas as pd
from sklearn.model_selection import train_test_split


def load_data(path):
    df = pd.read_csv(path)
    return df


def preprocess_data(df):
    '''Подготовка данных к обучению'''
    df = df.copy()

    # Дроп константы
    df = df.drop(columns='username')

    # Обработка признаков
    df = df.drop(columns=['date', 'time'])
    df['wrist'] = (df['wrist'] == '0').astype(int)

    return df


def split_data(df, test_size, random_state):
    '''Train, test сплит'''
    X_cols = [col for col in df.columns if col not in ['activity']]
    y_col = 'activity'

    X, y = df[X_cols], df[y_col]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        shuffle=True,
        stratify=y,
        test_size=test_size,
        random_state=random_state,
    )
    return X_train, X_test, y_train, y_test