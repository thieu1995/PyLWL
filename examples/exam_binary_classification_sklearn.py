#!/usr/bin/env python
# Created by "Thieu" at 14:46, 02/05/2025 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from pylwl import DataTransformer, LwClassifier


def get_cross_val_score(X, y, cv=3):
    ## Train and test
    model = LwClassifier(kernel="gaussian", tau=1.0)
    return cross_val_score(model, X, y, cv=cv)


def get_pipe_line(X, y):
    ## Split train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)

    ## Train and test
    model = LwClassifier(kernel='gaussian', tau=1.0)

    pipe = Pipeline([
        ("dt", DataTransformer(scaling_methods=("standard", "minmax"))),
        ("pnn", model)
    ])

    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)

    return model.evaluate(y_true=y_test, y_pred=y_pred, list_metrics=["F2S", "CKS", "FBS", "AS", "RS", "PS"])


def get_grid_search(X, y):
    ## Split train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)

    para_grid = {
        'kernel': ("gaussian", "tricube", "epanechnikov", "uniform", "cosine"),
        'tau': np.linspace(1., 20., 50),
    }

    ## Create a gridsearch
    model = LwClassifier()
    clf = GridSearchCV(model, para_grid, cv=3, scoring='accuracy', verbose=2)
    clf.fit(X_train, y_train)
    print("Best parameters found: ", clf.best_params_)
    print("Best model: ", clf.best_estimator_)
    print("Best training score: ", clf.best_score_)
    print(clf)

    ## Predict
    y_pred = clf.predict(X_test)
    return model.evaluate(y_true=y_test, y_pred=y_pred, list_metrics=["F2S", "CKS", "FBS", "AS", "RS", "PS"])


## Load data object
X, y = load_breast_cancer(return_X_y=True)

print(get_cross_val_score(X, y, cv=3))
print(get_pipe_line(X, y))
print(get_grid_search(X, y))
