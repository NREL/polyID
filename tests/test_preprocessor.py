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
    # Verify the Mn/Mw quantile binning: a missing weight lands in bin 0, and a
    # weight above the largest training value lands in the overflow bin
    # (n_bins + 2). Train on a controlled spread of weights so the discretizer
    # can form the requested number of bins (the shared fixture has a single
    # constant Mn/Mw value, which cannot exercise the binning logic).
    n_bins = 30
    preprocessor = WeightDiscPreprocessor(n_bins=n_bins)
    for i in range(1, 61):
        train_row = pd.Series(
            {"smiles_polymer": "CCO", "pm": 0.5, "Mn": float(i * 1000), "Mw": float(i * 2000)}
        )
        preprocessor(train_row, train=True)

    none_row = pd.Series({"smiles_polymer": "CCO", "pm": 0.5, "Mn": None, "Mw": 50000.0})
    none_dict = preprocessor(none_row)

    bin_edges = preprocessor.mn_kbd.bin_edges_[0]

    max_row = pd.Series({"smiles_polymer": "CCO", "pm": 0.5, "Mn": 10e10, "Mw": 50000.0})
    max_dict = preprocessor(max_row)

    assert len(bin_edges) == n_bins + 1
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
