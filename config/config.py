from config.model import get_model_dict


class Config:
    # Метрики
    evaluate_check = True
    
    # Важность признаков
    importance_check = True

    # Параметры
    random_state = 42
    test_size = 0.2
    cv_splits = 5
    n_repeats = 10
    
    # Пути
    data_path = 'data/data.csv'
    model_folder = 'model/'

    # Модель
    model_name = 'linear'
    model_dict = get_model_dict(random_state)


