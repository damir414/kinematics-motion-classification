import pandas as pd
from sklearn.inspection import permutation_importance


def run_perm_importance(model, X_test, y_test, n_repeats, random_state):
    pi = permutation_importance(
        model,
        X_test,
        y_test,
        n_repeats=n_repeats,
        random_state=random_state,
        scoring='f1_macro'
    )

    pi = pd.DataFrame({
        'feature': X_test.columns,
        'importance': pi.importances_mean
    }).sort_values('importance', ascending=False)

    return pi
