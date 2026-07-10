# Test single model functionality
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from polyid import Parameters, SingleModel
from polyid.models import global100
from polyid.preprocessors import PolymerPreprocessor
from sklearn.preprocessing import RobustScaler

pwd = Path(__file__).parent


@pytest.fixture()
def test_model():
    test_model = SingleModel(
        prediction_columns=["Tg", "Tm"],
        df_validate=pd.read_csv(pwd / "test_data/singlemodel/df_validate.csv"),
        df_train=pd.read_csv(pwd / "test_data/singlemodel/df_train.csv"),
        model_id="test_model",
    )

    scaler = RobustScaler()
    scaler.fit(test_model.df_validate[["Tg", "Tm"]])
    test_model.data_scaler = scaler

    return test_model


# TODO get test_model fixture to be passed to trained_model
@pytest.fixture(scope="module")
def trained_model():
    params = Parameters()
    params.num_messages = 1
    params.num_features = 2
    params.epochs = 3

    test_model = SingleModel(
        prediction_columns=["Tg", "Tm"],
        df_validate=pd.read_csv(pwd / "test_data/singlemodel/df_validate.csv"),
        df_train=pd.read_csv(pwd / "test_data/singlemodel/df_train.csv"),
        model_id="test_model",
    )

    scaler = RobustScaler()
    scaler.fit(test_model.df_validate[["Tg", "Tm"]])
    test_model.data_scaler = scaler
    test_model.generate_preprocessor(preprocessor=PolymerPreprocessor)
    test_model.train(global100, False, params.to_dict(), False)

    return test_model


def test_no_scaler(test_model):
    test_model.data_scaler = None
    assert (
        test_model.df_validate_scaled.loc[1, "Tg"]
        == test_model.df_validate.loc[1, "Tg"]
    )
    assert (
        test_model.df_validate_scaled.loc[0, "Tm"]
        == test_model.df_validate.loc[0, "Tm"]
    )

    assert test_model.df_train_scaled.loc[0, "Tg"] == test_model.df_train.loc[0, "Tg"]
    assert test_model.df_train_scaled.loc[0, "Tm"] == test_model.df_train.loc[0, "Tm"]


def test_scaler(test_model):
    # df_*_scaled should apply the fitted RobustScaler to the prediction columns.
    # Compare against a direct transform rather than hardcoded constants so the
    # test does not go stale when the fixture data changes.
    for attr in ("df_validate", "df_train"):
        raw = getattr(test_model, attr)
        scaled = getattr(test_model, f"{attr}_scaled")
        expected = test_model.data_scaler.transform(raw[["Tg", "Tm"]].values)
        assert np.allclose(
            scaled[["Tg", "Tm"]].values, expected, equal_nan=True
        )


def test_train_singlemodel(test_model):
    params = Parameters()
    params.num_messages = 1
    params.num_features = 2
    params.epochs = 3
    test_model.generate_preprocessor(preprocessor=PolymerPreprocessor)
    test_model.train(global100, False, params.to_dict(), True)

    # Test existence of the model
    assert test_model.model is not None

    # Test preprocessors
    assert np.array_equal(
        test_model.preprocessor(pd.Series({"smiles_polymer": "CCO"}), train=True)[
            "atom"
        ],
        np.array([2, 2, 3, 6, 6, 6, 6, 6, 6]),
    )
    assert np.array_equal(
        test_model.preprocessor(pd.Series({"smiles_polymer": "CCO"}), train=True)[
            "bond"
        ],
        np.array([2, 3, 3, 3, 2, 4, 3, 3, 4, 5, 3, 3, 3, 3, 3, 5]),
    )

    test_model


def test_predict_singlemodel(trained_model):
    try:
        trained_model.predict(trained_model.df_validate)
    except BaseException as e:
        raise e
