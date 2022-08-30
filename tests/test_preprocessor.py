# Test preprocessor functionality
# TODO save / load
from pathlib import Path

import pandas as pd
import pytest
from polyid import MultiModel, Parameters
from polyid.models import global100
from polyid.preprocessors import PolymerPreprocessor, WeightDiscPreprocessor

pwd = Path(__file__).parent


def test_quantile_preprocessor():
    params = Parameters()
    params.num_messages = 1
    params.num_features = 2
    params.epochs = 3
    n_bins = 30

    mm = MultiModel()
    mm.load_dataset(
        pwd / "test_data/preprocessors/220311_stereo_polymers_DP21_R1.csv", ["Tg", "Tm"]
    )
    mm.split_data(2)
    mm.generate_data_scaler()

    mm.generate_preprocessors(preprocessor=WeightDiscPreprocessor, n_bins=30)

    _, row = next(mm.df_input.iterrows())
    row.Mn = None
    none_dict = mm.models[0].preprocessor(row)

    bin_edges = mm.models[0].preprocessor.mn_kbd.bin_edges_[0]

    _, row = next(mm.df_input.iterrows())
    row.Mn = 10e10
    max_dict = mm.models[0].preprocessor(row)

    assert len(bin_edges == n_bins)
    assert none_dict["mn_bin"] == 0
    assert max_dict["mn_bin"] == n_bins + 2


def test_quantile_train():
    params = Parameters()
    params.num_messages = 1
    params.num_features = 2
    params.epochs = 3

    mm = MultiModel()
    mm.load_dataset(
        pwd / "test_data/preprocessors/220311_stereo_polymers_DP21_R1.csv", ["Tg", "Tm"]
    )
    mm.split_data(2)
    mm.generate_data_scaler()
    mm.generate_preprocessors(preprocessor=WeightDiscPreprocessor, n_bins=30)

    mm.train_models(modelbuilder=global100, model_params=params.to_dict())

    assert len(mm.models) == 2
    assert mm.models[0].model
    assert mm.models[1].model
