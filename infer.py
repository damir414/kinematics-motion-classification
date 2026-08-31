import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

from config.config import Config


class FeaturesData(BaseModel):
    wrist: int
    acceleration_x: float
    acceleration_y: float
    acceleration_z: float
    gyro_x: float
    gyro_y: float
    gyro_z: float


FEATURES = [
    'wrist',
    'acceleration_x',
    'acceleration_y',
    'acceleration_z',
    'gyro_x',
    'gyro_y',
    'gyro_z',
]

cfg = Config()
app = FastAPI()

# Загрузка модели
model_path = cfg.model_folder + cfg.model_name + '.pkl'
model = joblib.load(model_path)


# Api
@app.post('/infer')
def infer(data: FeaturesData):
    features = pd.DataFrame(np.array([
        data.wrist, data.acceleration_x, data.acceleration_y, data.acceleration_z, data.gyro_x, data.gyro_y, data.gyro_z
    ]).reshape(1, -1), columns=FEATURES)
    pred = model.predict(features)
    return {'activity': int(pred[0])}
