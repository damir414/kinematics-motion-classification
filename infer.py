import joblib
import numpy as np
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


cfg = Config()
app = FastAPI()

# Загрузка модели
model_path = cfg.model_folder + cfg.model_name + '.pkl'
model = joblib.load(model_path)


# Api
@app.post('/infer')
def infer(data: FeaturesData):
    features = np.array([
        data.wrist, data.acceleration_x, data.acceleration_y, data.acceleration_z, data.gyro_x, data.gyro_y, data.gyro_z
    ]).reshape(1, -1)
    pred = model.predict(features)
    return {'activity': int(pred[0])}
