# Test multi model functionality
# TODO save / load
from pathlib import Path

import pandas as pd
import pytest
from polyid import MultiModel, Parameters
from polyid.models import global100
from polyid.preprocessors import PolymerPreprocessor

pwd = Path(__file__).parent


@pytest.fixture(scope="module")
def trained_models():
    params = Parameters()
    params.num_messages = 1
    params.num_features = 2
    params.epochs = 3

    mm = MultiModel()
    mm.load_dataset(pwd / "test_data/multimodel/polymer_input.csv", ["Tg", "Tm"])
    mm.split_data(2)
    mm.generate_data_scaler()
    mm.generate_preprocessors(preprocessor=PolymerPreprocessor)

    mm.train_models(modelbuilder=global100, model_params=params.to_dict())

    return mm


def test_train_models():
    params = Parameters()
    params.num_messages = 1
    params.num_features = 2
    params.epochs = 3

    mm = MultiModel()
    mm.load_dataset(pwd / "test_data/multimodel/polymer_input.csv", ["Tg", "Tm"])
    mm.split_data(2)

    mm.generate_data_scaler()
    mm.generate_preprocessors(preprocessor=PolymerPreprocessor)

    mm.train_models(modelbuilder=global100, model_params=params.to_dict())

    assert len(mm.models) == 2
    assert mm.models[0].model
    assert mm.models[1].model


def test_train_models_no_scale():
    params = Parameters()
    params.num_messages = 1
    params.num_features = 2
    params.epochs = 3

    mm = MultiModel()
    mm.load_dataset(pwd / "test_data/multimodel/polymer_input.csv", ["Tg", "Tm"])
    mm.split_data(2)

    # mm.generate_data_scaler()
    mm.generate_preprocessors(preprocessor=PolymerPreprocessor)

    mm.train_models(modelbuilder=global100, model_params=params.to_dict())

    assert len(mm.models) == 2
    assert mm.models[0].model
    assert mm.models[1].model


def test_predict_models(trained_models):
    predict_df = trained_models.models[0].df_validate
    results_df = trained_models.make_predictions(predict_df)

    predict_df_shape = predict_df.shape
    results_df.to_csv('results.csv')
    predict_df.to_csv('predict.csv')
    assert results_df.shape == (predict_df_shape[0] * 2, 16)


def test_load_kfold():
    mm = MultiModel()
    smiles_monomer = [[i, i] for i in range(10)]
    smiles_monomer = [item for sublist in smiles_monomer for item in sublist]
    test_df = pd.DataFrame(
        data={
            "poly": [*range(20)],
            "pred": [*range(20)],
            "replicate_structure": [0, 1] * 10,
            "distribution": ["[]"] * 20,
            "smiles_monomer": smiles_monomer,
        }
    )

    test_df.to_csv(pwd / "test_data/test_load_kfold_data.csv", index=False)
    mm.load_dataset(pwd / "test_data/test_load_kfold_data.csv", ["pred"])

    kfold_sets = {
        "0": {"train": [0, 1, 2, 3, 4, 5, 6, 7], "validate": [8, 9]},
        "model one": {"train": [0, 2, 4, 6, 8], "validate": [1, 3, 5, 7, 9]},
    }

    mm.split_data(load_kfold=kfold_sets)

    assert True

    assert len(mm.models) == 2
    # Check first model
    assert set(mm.models[0].df_train.data_id) == {0, 1, 2, 3, 4, 5, 6, 7}
    assert set(mm.models[0].df_validate.data_id) == {8, 9}

    # Check second model
    mm.models[1].model_id == "model one"
    assert set(mm.models[1].df_train.data_id) == {0, 2, 4, 6, 8}
    assert set(mm.models[1].df_validate.data_id) == {1, 3, 5, 7, 9}
