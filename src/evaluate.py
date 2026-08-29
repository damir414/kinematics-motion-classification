import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import cross_validate


def run_evaluate(model, X_train, y_train, X_test, y_test, cv_splits):
    '''Считает метрики на Train, Test, Cross Validation'''

    # Метркии на Train
    y_pred_train = model.predict(X_train)
    accuracy_train = accuracy_score(y_train, y_pred_train)
    precision_train = precision_score(y_train, y_pred_train)
    recall_train = recall_score(y_train, y_pred_train)
    f1_train = f1_score(y_train, y_pred_train)

    # Метркии на Test
    y_pred_test = model.predict(X_test)
    accuracy_test = accuracy_score(y_test, y_pred_test)
    precision_test = precision_score(y_test, y_pred_test)
    recall_test = recall_score(y_test, y_pred_test)
    f1_test = f1_score(y_test, y_pred_test)

    # Метрики на кросс валидации
    cv_scores = cross_validate(
        model, X_train, y_train,
        cv=cv_splits,
        scoring=['accuracy', 'precision_macro', 'recall_macro', 'f1_macro'],
        return_train_score=False,
    )
    accuracy_cv = cv_scores['test_accuracy']
    precision_cv = cv_scores['test_precision_macro']
    recall_cv = cv_scores['test_recall_macro']
    f1_cv = cv_scores['test_f1_macro']

    # Результат
    eval_df = pd.DataFrame({
        # Метркии на Train
        'Train Accuracy': [accuracy_train],
        'Train Precision': [precision_train],
        'Train Recall': [recall_train],
        'Train F1': [f1_train],

        # Метркии на Train
        'Test Accuracy': [accuracy_test],
        'Test Precision': [precision_test],
        'Test Recall': [recall_test],
        'Test F1': [f1_test],

        # Метрики на кросс валидации
        'Mean Accuracy': [np.mean(accuracy_cv)],
        'Mean Precision': [np.mean(precision_cv)],
        'Mean Recall': [np.mean(recall_cv)],
        'Mean F1': [np.mean(f1_cv)],
    })

    return eval_df