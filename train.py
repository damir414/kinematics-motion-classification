import joblib

from config.config import Config
from src.data import load_data, preprocess_data, split_data
from src.evaluate import run_evaluate
from src.importance import run_perm_importance
from src.utils import seed_everything


def train(cfg: Config):
    # Подготовка данных
    seed_everything(cfg.random_state)
    df = load_data(cfg.data_path)
    df = preprocess_data(df)
    X_train, X_test, y_train, y_test = split_data(df, cfg.test_size, cfg.random_state)

    # Обучение и сохранение модели
    model = cfg.model_dict[cfg.model_name]
    model.fit(X_train, y_train)
    model_path = cfg.model_folder + cfg.model_name + '.pkl'
    joblib.dump(model, model_path)

    # Оценка модели
    if cfg.evaluate_check:
        evals = run_evaluate(model, X_train, y_train, X_test, y_test, cfg.cv_splits)
        print(evals)

    # Важность признаков
    if cfg.importance_check:
        pi = run_perm_importance(model, X_test, y_test, cfg.n_repeats, cfg.random_state)
        print(pi)


if __name__ == '__main__':
    cfg = Config()
    train(cfg)