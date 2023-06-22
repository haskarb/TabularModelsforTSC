import numpy as np
import pandas as pd
from aeon.datasets import load_UCR_UEA_dataset
from aeon.datatypes._panel._convert import from_nested_to_3d_numpy
from sklearn.linear_model import LogisticRegressionCV, RidgeClassifierCV
from sklearn.ensemble import RandomForestClassifier
from aeon.transformations.panel.rocket import MiniRocket
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from typing import Callable

from sklearn.pipeline import make_pipeline

from datasets import dataset_univariate, dataset_asc

random_state = 0


def normalize_vector(channel: np.ndarray) -> np.ndarray:
    """
    Normalize a vector to have zero mean and unit variance.
    """
    if channel.std() == 0:
        return channel - channel.mean()
    return (channel - channel.mean()) / channel.std()


def convert_3d_to_2d(X) -> np.ndarray:
    if isinstance(X, pd.DataFrame):
        X = from_nested_to_3d_numpy(X)

    X_2d = np.zeros((X.shape[0], X.shape[1] * X.shape[2]))

    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            X_2d[i, j * X.shape[2] : (j + 1) * X.shape[2]] = normalize_vector(
                X[i, j, :]
            )

    return X_2d


def fit_model(data: str, model_: str):
    print(f"Data: {data} model: {model_}")
    train_x, train_y = load_UCR_UEA_dataset(data, return_X_y=True, split="train")  # type: ignore
    test_x, test_y = load_UCR_UEA_dataset(data, return_X_y=True, split="test")  # type: ignore

    # create dictionary of models
    models = {
        "RidgeClassifierCV": RidgeClassifierCV(),
        "LogisticRegressionCV": make_pipeline(
            StandardScaler(),
            LogisticRegressionCV(random_state=random_state),
        ),
        "RandomForestClassifier": RandomForestClassifier(
            random_state=random_state
        ),
        "LDA": LinearDiscriminantAnalysis(),
    }
    model = models[model_]

    #check if ts data is multivariate
    if train_x.shape[0] > 1:
        train_x = convert_3d_to_2d(train_x)
        test_x = convert_3d_to_2d(test_x)

    # # convert 3d to univariate-2d
    # train_x = data_fn(train_x)
    # test_x = data_fn(test_x)

    start = time.time()
    model.fit(train_x, train_y)

    # evaluate the model
    yhat = model.predict(test_x)
    end = time.time()
    # calculate accuracy
    acc = np.sum(yhat == test_y) / len(test_y)

    print(f"Data: {data} Accuracy: {acc * 100:.2f}%")

    return acc, (end - start)/60


if __name__ == "__main__":
    import time

    
    linear_model = [
        # "RidgeClassifierCV",
        "LogisticRegressionCV",
        # "RandomForestClassifier",
        # "LDA",
    ]

    for mod in linear_model:
        print(f"Model: {mod}")
        results = pd.DataFrame(columns=["dataset", "accuracy", "time(m)"])
        for item in dataset_asc:
            # print(f"Dataset: {item}")
            try:
                acc, time_ = fit_model(data=item, model_=mod)
            except Exception as e:
                print(f"Error: {e}")
                continue
            temp_df = pd.DataFrame(
                {"dataset": item, "accuracy": [acc], "time(m)": [time_]}
            )

            results = pd.concat([results, temp_df], ignore_index=True)

        results.to_csv(f"LinearClassifier_MTSC_{mod}.csv", index=False)
    # break
