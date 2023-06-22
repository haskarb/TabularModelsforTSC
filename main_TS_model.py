import numpy as np
import pandas as pd
from aeon.datasets import load_UCR_UEA_dataset
from aeon.datatypes._panel._convert import from_nested_to_3d_numpy
from sklearn.linear_model import LogisticRegression, RidgeClassifierCV
from sklearn.ensemble import RandomForestClassifier
from aeon.transformations.panel.rocket import (
    MiniRocketMultivariate,
    Rocket,
    MultiRocketMultivariate,
)
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from typing import Callable

from sklearn.pipeline import make_pipeline

from datasets import dataset_asc

random_state = 0


def fit_model(data: str, model_: str):
    print(f"Data: {data} model: {model_}")
    train_x, train_y = load_UCR_UEA_dataset(data, return_X_y=True, split="train")  # type: ignore
    test_x, test_y = load_UCR_UEA_dataset(data, return_X_y=True, split="test")  # type: ignore

    # create dictionary of models
    ts_models = {
        "rocket": make_pipeline(
            Rocket(random_state=random_state),
            StandardScaler(),
            RidgeClassifierCV(alphas=np.logspace(-3, 3, 10)),
        ),
        "mr": make_pipeline(
            MiniRocketMultivariate(random_state=random_state),
            StandardScaler(),
            RidgeClassifierCV(alphas=np.logspace(-3, 3, 10)),
        ),
        "mur": make_pipeline(
            MultiRocketMultivariate(random_state=random_state),
            StandardScaler(),
            RidgeClassifierCV(alphas=np.logspace(-3, 3, 10)),
        ),
    }

    model = ts_models[model_]

    if model_ == "mr" and train_x.iloc[0].shape[0] < 9:
        train_x = from_nested_to_3d_numpy(train_x)
        test_x = from_nested_to_3d_numpy(test_x)

    start = time.time()
    model.fit(train_x, train_y)

    # evaluate the model
    yhat = model.predict(test_x)
    end = time.time()

    # calculate accuracy
    acc = np.sum(yhat == test_y) / len(test_y)

    print(f"Data: {data} Accuracy: {acc * 100:.2f}%")

    return acc, (end - start) / 60


if __name__ == "__main__":
    import time

    linear_model = [
        "rocket",
        "mr",
        "mur",
    ]

    for mod in linear_model:
        print(f"Model: {mod}")
        results = pd.DataFrame(columns=["dataset", "accuracy", "time(m)"])
        for item in dataset_asc:
            # print(f"Dataset: {item}")
            start = time.time()
            acc, time_ = fit_model(data=item, model_=mod)
            end = time.time()
            temp_df = pd.DataFrame(
                {
                    "dataset": item,
                    "accuracy": [acc],
                    "time(m)": [time_],
                }
            )

            results = pd.concat([results, temp_df], ignore_index=True)

        results.to_csv(f"LinearClassifier_{mod}.csv", index=False)
    # break
