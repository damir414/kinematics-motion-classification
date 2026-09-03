import pytest
import pandas as pd
from sklearn.datasets import make_classification


@pytest.fixture
def synthetic_data():
    X, y = make_classification(n_samples=50, n_features=7, n_informative=2, n_redundant=5, n_classes=2, random_state=42)
    columns = ['wrist', 'acceleration_x', 'acceleration_y', 'acceleration_z', 'gyro_x', 'gyro_y', 'gyro_z']
    df = pd.DataFrame(X, columns=columns)
    df['activity'] = y
    df.insert(0, 'date', '2026-1-10')
    df.insert(1, 'time', '10:10:10:100000000')
    df.insert(2, 'username', 'ivan')
    return df