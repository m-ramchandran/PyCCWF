import numpy as np
from scipy.optimize import nnls
from sklearn.linear_model import Lasso, Ridge


def compute_stack_coefficients(X, y, method='nnls', intercept=True):
    """
    Compute stacking coefficients using different methods.

    Parameters:
    -----------
    X : array-like
        Predictor matrix
    y : array-like
        Target values
    method : str
        Method for computing coefficients ('nnls', 'lasso', 'ridge')
    intercept : bool
        Whether to include intercept

    Returns:
    --------
    tuple : (coefficients, intercept)
    """
    if method == 'nnls':
        if intercept:
            X_aug = np.column_stack([np.ones(len(X)), X])
            coeffs = nnls(X_aug, y)[0]
            return coeffs[1:], coeffs[0]
        else:
            coeffs = nnls(X, y)[0]
            return coeffs, 0.0

    elif method == 'lasso':
        model = Lasso(alpha=1.0, positive=True, fit_intercept=intercept)
        model.fit(X, y)
        return model.coef_, model.intercept_

    elif method == 'ridge':
        model = Ridge(alpha=1.0, positive=True, fit_intercept=intercept)
        model.fit(X, y)
        return model.coef_, model.intercept_

    else:
        raise ValueError(f"Unknown method: {method}")


def compute_stacking_predictions(test_preds, coefficients, method):
    """
    Compute predictions using stacking coefficients.

    Parameters:
    -----------
    test_preds : array-like
        Predictions from base models
    coefficients : dict
        Dictionary containing coefficients and intercept
    method : str
        Stacking method

    Returns:
    --------
    array-like : Stacked predictions
    """
    coef = coefficients[method]['coef']
    intercept = coefficients[method]['intercept']
    return intercept + np.dot(test_preds, coef)


