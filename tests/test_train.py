import joblib

from config.config import Config
from train import train


def test_train_integration(tmp_path, synthetic_data):
    # Сохранение
    df = synthetic_data
    data_path_str = str(tmp_path / 'data.csv')
    model_folder_str = str(tmp_path) + '/'
    df.to_csv(data_path_str, index=False)

    # Конфиг
    cfg = Config()
    cfg.data_path = data_path_str
    cfg.model_folder = model_folder_str
    cfg.evaluate_check = False
    cfg.importance_check = False

    # Тест
    models_to_test = ['linear', 'naive']
    for model_name in models_to_test:
        cfg.model_name = model_name
        train(cfg)
        model_path = cfg.model_folder + cfg.model_name + '.pkl'
        model = joblib.load(model_path)
        sample = df.drop(columns=['activity']).iloc[[0]]
        pred = model.predict(sample)
        assert pred[0] in [0, 1]
