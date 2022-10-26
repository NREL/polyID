# Have validation be the test set (rename)
# test,train,val isn't same as kfolds

# Rename SingleModel, MultiModel
# scaler is from MuliModel pre kfold
# Have option for singlemodel scaler if none provided
# indiviual prediction in single
# predicts in multi (return df with aggregate)
# Save to individual folders, ensure can reproduce on reload

from __future__ import annotations

import glob
import json
import pickle as pk
from collections.abc import Callable
from copy import copy
from pathlib import Path
from typing import Dict, List, Union

import nfp
import pandas as pd
import numpy as np
import shortuuid
import tensorflow as tf
from keras.models import load_model as load_keras_model
from nfp import (EdgeUpdate, GlobalUpdate, NodeUpdate,
                 masked_mean_absolute_error)
from sklearn import model_selection
from sklearn.preprocessing import RobustScaler
from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint

from polyid.models.callbacks import PandasLogger
from polyid.preprocessors.features import atom_features_v1, bond_features_v1


def generate_hash(df, hash_cols=["smiles_polymer", "monomers"]) -> str:
    """Function to create unique hash based on values in hash_cols. This is used to ensure data quality when merging replicate structures and multiple k-fold model predictions after generating results

    Args:
        df (_type_): dataframe in which to create hash values for each row
        hash_cols (list): columsn in dataframe which will be used to create hash. Defaults to ['smiles_polymer','monomers'].

    Returns:
        str: dataframe with hash value as index
    """

    ps_hash = df[hash_cols[0]].astype(str)
    for col in hash_cols[1:]:
        ps_hash = ps_hash + df[col].astype(str)
    df.index = ps_hash.apply(shortuuid.uuid)
    df.index.name = "hash-{}".format("-".join(hash_cols))

    return df


class SingleModel:
    """Single Model for predicting polymer properties

    Attributes
    ----------
    prediction_columns: List[str]
        The columns to be used for prediction and training.
    df_validate: pd.DataFrame
        The dataframe used for validation when training.
    df_train: pd.DataFrame
        The dataframe used for training.
    df_test: pd.DataFrame
        DataFrame used for testing when training.
    df_validate_results : pd.DataFrame
        DataFrame containing the validation results of training
    df_loss_log : pd.DataFrame
        The log recording loss values
    data_scaler : RobustScaler
        A scaler to scale the data.
    preprocessor : Callable
        An nfp preprocessor.
    validate_generator : tf.data.Dataset
        Generator used for training for the validation dataset.
    train_generator : tf.data.Dataset
        Generator used for training for the training dataset.
    model_id : str
        The model ID.
    model : keras.engine.functional.Functional
        The keras model.
    """

    def __init__(
        self,
        prediction_columns: List[str],
        df_validate: pd.DataFrame = None,
        df_train: pd.DataFrame = None,
        df_test: pd.DataFrame = None,
        data_scaler: RobustScaler = None,
        model_id: str = None,
    ):
        """SingleModel class to train a tensor flow model with.

        Parameters
        ----------
        prediction_columns : List[str]
            The columns to be used for prediction and training.
        df_validate : pd.DataFrame, optional
            The dataframe used for validation when training, by default None
        df_train : pd.DataFrame, optional
            The dataframe used for training when training, by default None
        df_test : pd.DataFrame, optional
            The dataframe used for testing when training, by default None
        data_scaler : RobustScaler, optional
            A scaler to scale the data, by default None
        model_id : str, optional
            An id for the model, by default None
        """
        self.prediction_columns = prediction_columns
        self.df_validate = df_validate
        self.df_train = df_train
        self.df_test = df_test
        self.df_validate_results = None
        self.df_loss_log = None
        self.kfold_sets = None

        self.data_scaler = data_scaler

        self.preprocessor = None
        self.validate_generator = None
        self.train_generator = None

        self.model_id = model_id
        self.model = None

        self.parameters = None

        self.atom_features = atom_features_v1
        self.bond_features = bond_features_v1

    @property
    def df_train_scaled(self) -> pd.DataFrame:
        """Returns the scaled df_train"""
        if not self.data_scaler:
            return self.df_train
        else:
            return self._scale_data(self.df_train)

    @property
    def df_validate_scaled(self) -> pd.DataFrame:
        """Returns the scaled df_validate"""
        if not self.data_scaler:
            return self.df_validate
        else:
            return self._scale_data(self.df_validate)

    @classmethod
    def load_model(
        cls,
        model_fname: Union[Path, str],
        data_fname: Union[Path, str],
        custom_objects: Dict[Callable] = None,
    ) -> SingleModel:
        """Create a new SingleModel object from saved files.

        Parameters
        ----------
        model_fname : Union[Path, str]
            The filepath for the model .h5 file.
        data_fname : Union[Path, str]
            The filepath for the model data .pk file.
        custom_objects : Dict[Callable], optional
            The custom objects used to create the model. If none are provided defaults
            are nfp GlobalUpdate, EdgeUpdate, NodeUpdate, and masked_mean_absolute_error
            , by default None

        Returns
        -------
        SingleModel
            The initialized SingleModel class.
        """
        custom_objects_dict = {
            "GlobalUpdate": GlobalUpdate,
            "EdgeUpdate": EdgeUpdate,
            "NodeUpdate": NodeUpdate,
            "masked_mean_absolute_error": masked_mean_absolute_error,
        }
        if custom_objects:
            for key, val in custom_objects.items():
                custom_objects_dict[key] = val

        try:
            with open(data_fname, "rb") as f:
                load_dict = pk.load(f)
        except:
            # backwards compability for models saved under polyml instead of polyid
            with open(data_fname, "rb") as f:
                load_dict = RenameUnpickler.renamed_load(f)

        model = cls(prediction_columns=load_dict["prediction_columns"])
        for key, val in load_dict.items():
            setattr(model, key, val)

        model.model = load_keras_model(model_fname, custom_objects=custom_objects_dict)

        return model

    def generate_data_scaler(self):
        self.data_scaler = RobustScaler()
        self.data_scaler.fit(self.df_train[self.prediction_columns].values)

    def generate_preprocessor(
        self,
        preprocessor,
        atom_features: Callable = nfp.preprocessing.features.atom_features_v1,
        bond_features: Callable = nfp.preprocessing.features.bond_features_v1,
        batch_size: int = 1,
        **kwargs,
    ) -> None:
        """Generate a preprocessor for the model.

        Parameters
        ----------
        atom_features : Callable, optional
            A function applied to an rdkit.Atom that returns some representation (i.e.,
            string, integer) of the atom features, by default nfp.atom_features_v1
        bond_features : Callable, optional
            A function applied to an rdkit.Atom that returns some representation (i.e.,
            string, integer) of the bond features, by default nfp.bond_features_v1
        batch_size : int, optional
            The batch size for the padded batch when creating a generator, by default 1
        """
        self.preprocessor = preprocessor(
            atom_features=atom_features, bond_features=bond_features, **kwargs
        )

        for _, row in self.df_train.iterrows():
            self.preprocessor(row, train=True)

        if hasattr(self.preprocessor, "train"):
            self.preprocessor.train = False

        # Check if train/test data exist and create generators
        if self.df_train_scaled is not None:
            self.train_generator = self._create_generator(
                self.df_train_scaled, batch_size
            )
        if self.df_validate_scaled is not None:
            self.validate_generator = self._create_generator(
                self.df_validate_scaled, batch_size
            )

    def train(
        self,
        modelbuilder: Callable,
        model_summary: bool = False,
        model_params: dict = None,
        verbose: bool = True,
        save_folder: str = None,
        save_training: bool = False,
        callbacks: List = None,
    ) -> None:
        """Train a SingleModel

        Parameters
        ----------
        modelbuilder : Callable
            A function that returns a tensorflow model.
        model_summary : bool, optional
            Whether or not to print the model summary, by default False
        model_params : dict, optional
            A dictionary of model parameters used for training, by default {}
        verbose : bool, optional
            Whether or not to print training output, by default True
        save_folder: str, optional
            The folder to save results in. If not specified model is not saved,
            by default none.
        save_training: bool, False
            Whether or not to save training data.
        """
        tf_model = modelbuilder(
            self.preprocessor,
            model_summary=model_summary,
            prediction_columns=self.prediction_columns,
            params=model_params,
        )
        print(tf_model.summary)

        hist = tf_model.fit(
            self.train_generator,
            validation_data=self.validate_generator,
            epochs=model_params["epochs"],
            verbose=verbose,
            callbacks=callbacks,
        )

        self.model = hist.model
        self.parameters = model_params

        # Record validation results
        self.df_validate_results = self.predict(self.df_validate)
        for col in self.prediction_columns:
            self.df_validate_results[col + "_err"] = (
                self.df_validate_results[col] - self.df_validate_results[col + "_pred"]
            )

        if save_folder:
            self._save_model(save_folder, save_training)

    def predict(self, df_prediction: pd.DataFrame) -> pd.DataFrame:
        """Make a prediction using a trained model and a dataframe with unscaled values

        Parameters
        ----------
        df_prediction : pd.DataFrame
            A dataframe containing a column of smiles labeled as "smiles_polymer".

        Returns
        -------
        pd.DataFrame
            A dataframe that is df_prediction appended with the prediction columns
            of the model.
        """
        # Predict, inverse scale, append to original df
        prediction_generator = self._create_generator(df_prediction, predict=True)

        predictions = self.model.predict(prediction_generator)
        if self.data_scaler:
            predictions = self.data_scaler.inverse_transform(predictions)

        df_prediction_results = pd.DataFrame(
            data=predictions, columns=[f"{col}_pred" for col in self.prediction_columns]
        )

        df_prediction_results.index = df_prediction.index
        df_prediction_results = df_prediction.join(df_prediction_results, how="outer")

        return df_prediction_results

    def to_dict(self, include_training_data: bool = False) -> Dict:
        """Returns a dictionary of SingleModel values.

        Parameters
        ----------
        include_training_data : bool, optional
            Whether or not to include training data, that is data frames used for
            training, by default False

        Returns
        -------
        Dict
            A dictionary containing model values.
        """
        return_dict = {
            "prediction_columns": self.prediction_columns,
            "data_scaler": self.data_scaler,
            "preprocessor": self.preprocessor,
            "model": self.model,
        }

        if include_training_data:
            return_dict["df_train"] = self.df_train
            return_dict["df_validate"] = self.df_validate

            if self.df_test is not None:
                return_dict["df_test"] = self.df_test

        return return_dict

    def _create_generator(
        self,
        df: pd.DataFrame,
        train: bool = False,
        batch_size: int = 1,
        predict: bool = False,
    ) -> tf.data.Dataset:
        """Creates a generator from a given dataframe, using the preprocessor already
        generated in the class"""
        # TODO do I ever use train=True here?
        if not predict:
            return (
                tf.data.Dataset.from_generator(
                    lambda: (
                        (
                            self.preprocessor(row, train=False),
                            row[self.prediction_columns].values,
                        )
                        for _, row in df.iterrows()
                    ),
                    output_signature=(
                        self.preprocessor.output_signature,
                        tf.TensorSpec(
                            (len(self.prediction_columns),), dtype=tf.float32
                        ),
                    ),
                )
                .cache()
                .shuffle(buffer_size=200)
                .padded_batch(batch_size=batch_size)
                .prefetch(tf.data.experimental.AUTOTUNE)
            )

        else:  # Prediction generator
            return (
                tf.data.Dataset.from_generator(
                    lambda: (
                        self.preprocessor(row, train=train) for _, row in df.iterrows()
                    ),
                    output_signature=self.preprocessor.output_signature,
                )
                .padded_batch(batch_size=batch_size)
                .prefetch(tf.data.experimental.AUTOTUNE)
            )

    def _scale_data(self, df):
        """Scale prediction columns of a dataframe."""
        df = df.copy()
        df_scaled = self.data_scaler.transform(df[self.prediction_columns].values)
        for i, predict_column in enumerate(self.prediction_columns):
            df[predict_column] = df_scaled[:, i]

        return df

    def _save_model(
        self, folder: Union[Path, str], save_training: bool = False
    ) -> None:
        """Save the model to two files, a .m5 and a .pk containing the preprocessor,
        data_scaler, prediction_columns, and parameters.

        Parameters
        ----------
        folder : Union[Path, str]
            The folder in which to save the files.
        save_training : bool
            Whether or not to save data related to training, by default False.
        """
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        # self.model.save(folder / f"model_final_{self.model_id}.h5")

        output_dict = {
            "prediction_columns": self.prediction_columns,
            "data_scaler": self.data_scaler,
            "preprocessor": self.preprocessor,
            "model_id": self.model_id,
            "parameters": self.parameters,
            "atom_features": self.atom_features,
            "bond_features": self.bond_features,
        }

        if save_training:
            output_dict["df_train"] = self.df_train
            output_dict["df_test"] = self.df_test
            output_dict["df_validate"] = self.df_validate
            output_dict["df_validate_results"] = self.df_validate_results
            output_dict["df_loss_log"] = self.df_loss_log

        with open(folder / f"model_{self.model_id}_data.pk", "wb") as f:
            pk.dump(output_dict, f)


# TODO: add attributes to docstring
class MultiModel:
    def __init__(self):
        """Manages multiple SingleModels allowing for the training/prediction of
        multiple models at once.

        Parameters
        -------
        df_polymer: pd.DataFrame
            The polymer DataFrame used for model training.
        df_polymer_scaled: pd.DataFrame
            The scaled polymer DataFrame used for training.
        prediction_columns: List[str]
            A list of strings representing the columns to be used for training
            and prediction.
        prediction_units: dict
            A dictionary mapping the prediction columns to their units.
        models: List[SingleModels]
            A list containing the SingleModels.
        data_scaler: RobustScaler
            A function that scales the data.
        """
        self.prediction_columns = list()
        self.prediction_units = dict()
        self.models = list()
        self.data_scaler = None
        return None

    @classmethod
    def load_models(
        cls, folder: Union[Path, str], custom_objects: Dict[Callable] = None,nmodels: Union[int,list]=None
    ) -> MultiModel:
        """Load models to create a MultiModel Class

        Parameters
        ----------
        folder : Union[Path, str]
            The folder the individual models are found in.
        custom_objects : Dict[Callable], optional
            The custom objects used to create the model. If none are provided defaults
            are nfp GlobalUpdate, EdgeUpdate, NodeUpdate, and masked_mean_absolute_error
            , by default None
        nmodels : Union[int, list], optional
            Used to import models 0 through nmodels if value passed is an int. If value
            passed is a list, then those models will be imported. Should be a list of 
            integers. by default None

        Returns
        -------
        MultiModel
            A MultiModel class populated with the save data.
        """
        folder = Path(folder)
        custom_objects_dict = {
            "GlobalUpdate": GlobalUpdate,
            "EdgeUpdate": EdgeUpdate,
            "NodeUpdate": NodeUpdate,
            "masked_mean_absolute_error": masked_mean_absolute_error,
        }

        if custom_objects:
            for key, val in custom_objects.items():
                custom_objects_dict[key] = val

        model_folders = glob.glob(str(folder / "model_*"))
        data_path = Path(folder / "parameters.pk")

        mm = cls()
        with open(data_path, "rb") as f:
            data_dicts = pk.load(f)
            for key, val in data_dicts.items():
                try:
                    setattr(mm, key, val)
                except Exception:
                    print(f"Can't set attribute {key}")
        # Wipe models just to be safe
        mm.models = []

        if nmodels == None:
            for model_folder in model_folders:
                model_path = Path(model_folder) / (model_folder.rsplit("/")[-1] + ".h5")
                data_path = Path(model_folder) / (model_folder.rsplit("/")[-1] + "_data.pk")
                mm.models.append(
                    SingleModel.load_model(
                        model_path, data_path, custom_objects=custom_objects_dict
                    )
                )
        elif type(nmodels)==int:
            nmodels = list(np.arange(0,nmodels))
            for nmodel in nmodels:
                model_path = Path(str(folder / 'model_{}/model_{}.h5'.format(nmodel,nmodel)))
                data_path =  Path(str(folder / 'model_{}/model_{}_data.pk'.format(nmodel,nmodel)))
                mm.models.append(
                    SingleModel.load_model(
                        model_path, data_path, custom_objects=custom_objects_dict
                    )
                )
        elif type(nmodels)==list:
            for nmodel in nmodels:
                model_path = Path(str(folder / 'model_{}/model_{}.h5'.format(nmodel,nmodel)))
                data_path =  Path(str(folder / 'model_{}/model_{}_data.pk'.format(nmodel,nmodel)))
                mm.models.append(
                    SingleModel.load_model(
                        model_path, data_path, custom_objects=custom_objects_dict
                    )
                )
        else:
            print("Error: nmodel type not recognized. Must be int or list.")

        return mm

    @property
    def df_polymer(self) -> pd.DataFrame:
        """Returns the polymer dataframe for entries containing one or more of
        prediction columns."""
        return self.df_input.dropna(
            subset=self.prediction_columns, how="all"
        ).reset_index(drop=True)

    @property
    def df_loss_log(self) -> pd.DataFrame:
        """Returns the loss log for all models."""
        df_loss_list = []
        for model in self.models:
            df_loss = model.df_loss_log.copy()
            df_loss["model_id"] = model.model_id
            df_loss_list.append(df_loss)

        if df_loss_list:
            return pd.concat(df_loss_list).reindex()
        else:
            return None

    def load_dataset(self, fname: str, prediction_columns: List[str]) -> None:
        """Load a dataset of polymer properties to be used for prediction.

        Parameters
        ----------
        fname : str
            The file location of the dataset.
        prediction_columns : List[str]
            The columns to train a model on and use for prediction.
        """
        prediction_columns = list(prediction_columns)
        fname = Path(fname)
        # Read file in as pandas df
        if fname.suffix == ".csv":
            df_load = pd.read_csv(fname)
        elif fname.suffix == ".xlsx":
            df_load = pd.read_excel(fname)
        else:
            raise Exception("File must be in .csv or .xlsx format.")

        # Check to see that units columns all have the same value within column,
        # otherwise output warning
        self.prediction_units = dict()
        for col in prediction_columns:
            col_to_check = col + "_units"
            if col_to_check in df_load.columns:
                units_set = list(set(df_load[col_to_check]))
                self.prediction_units[col] = units_set
                # TODO implement logger
                if len(units_set) > 1:
                    print(
                        f"Warning: The units for {col_to_check.strip('_units')}"
                        " are not uniform."
                    )
            else:
                self.prediction_units[col] = None

        # Assign values to class
        self.prediction_columns = list(prediction_columns)
        self.df_input = df_load

    def make_predictions(self, df_prediction: pd.DataFrame) -> pd.DataFrame:
        """Make predictions using each model.

        Parameters
        ----------
        df_prediction : pd.DataFrame
            Dataframe with a smiles_polymer column that will be used to predict.

        Returns
        -------
        pd.DataFrame
            The df_prediction containing predictions of all models.
        """
        df_results = pd.DataFrame()
        for model in self.models:
            df_result = model.predict(df_prediction)
            df_result["model_id"] = model.model_id
            df_results = df_results.append(df_result)

        return df_results

    def make_aggregate_predictions(
        self,
        df_prediction: pd.DataFrame,
        funcs: Union[Callable, str, List, Dict] = None,
        groupby_monomers: bool = True,
        groupby_pm: bool = True,
        additional_groupby: list = None,
    ) -> pd.DataFrame:
        """Returns the aggregate prediction of all models.

        Parameters
        ----------
        df_prediction : pd.DataFrame
            The dataframe to predict properties based on a smiles_polymer column.
        funcs : list, optional
            Aggregation functions used by Pandas aggregate call,
            by default ["mean"]
        groupby_monomers : bool, optional
            Whether to group by monomers. If False polymer structure is used,
            by default True
        groupby_pm : bool, optional
            Whether to use the pm value for grouping, by default True

        Returns
        -------
        pd.DataFrame
            A DataFrame containing the aggregated predictions.
        """
        if not (
            isinstance(additional_groupby, list)
            or isinstance(additional_groupby, tuple)
        ):
            additional_groupby = []

        if not funcs:
            funcs = ["mean"]

        df_prediction = self.make_predictions(df_prediction)

        # Required columns
        groupby_columns = ["pm", "distribution"]
        groupby_columns.extend(additional_groupby)
        if not groupby_pm or "pm" not in df_prediction.columns:
            groupby_columns.remove("pm")

        if groupby_monomers:
            groupby_columns.append("smiles_monomer")
        else:
            groupby_columns.append("smiles_polymer")

        # Get columns for grouping, and columns to drop
        drop_columns = set(df_prediction.columns) - set(groupby_columns)
        drop_columns = drop_columns - set(self.prediction_columns)
        drop_columns = drop_columns - set(
            [col + "_units" for col in self.prediction_columns]
        )
        drop_columns = drop_columns - set(
            [col + "_pred" for col in self.prediction_columns]
        )

        df_prediction = df_prediction.drop(columns=drop_columns)

        df_prediction = df_prediction.groupby(groupby_columns).agg(funcs).reset_index()
        df_prediction.columns = ["_".join(col) for col in df_prediction.columns]

        return df_prediction

    def split_data(
        self,
        kfolds: int = 2,
        stratify: bool = True,
        shuffle: bool = True,
        load_kfold: Union[str, dict] = None,
        kfold_save: str = None,
    ) -> None:
        """Split the data into k-folds.

        Parameters
        ----------
        kfolds : int, optional
            The number of kfolds, must be greater than 1, by default 1
        stratify : bool, optional
            Whether or not to generate mechanism stratified k-folds, by default True
        shuffle : bool, optional
            Whether or not to shuffle the data before splitting, by default True
        load_kfold: Union[str, dict], optional
            A file path or dictionary containing pregenerated kfold information
        kfold_save: str
            A file path to save the kfold split information
        """
            # Remove any existing models
        self.models = []
        # Assign a kfold-id to each column

        # getting just the 0 replicate structuctures
        data_zero_rep = (
            self.df_polymer[self.df_polymer.replicate_structure == 0]
            .copy()
            .reset_index()
        )

        #replacing nans for logical statements later
        data_zero_rep.distribution = data_zero_rep.distribution.fillna(0)

        data = self.df_polymer.copy()

        # Assign a data_id column to aid in kfolds; data id acts as a unique identifer for creating stratified splits. It should be a unique integer. 
        if "data_id" not in data:
            data["data_id"] = 0
            #replacing nans for logical statements later
            data.distribution = data.distribution.fillna(0)
            for i, row in data_zero_rep.iterrows():

                if "pm" in row:
                    idxs = data[
                        (data["smiles_monomer"] == row.smiles_monomer) & 
                        (data["distribution"] == row.distribution)&
                        (data["pm"]==row.pm)
                    ].index.tolist()

                else: 
                    idxs = data[(data["smiles_monomer"] == row.smiles_monomer) & (
                        data["distribution"] == row.distribution
                    )].index.tolist()

                data.loc[idxs, "data_id"] = i

        if not load_kfold:
            if stratify:
                kfold_sets = model_selection.StratifiedKFold(
                    n_splits=kfolds, shuffle=shuffle
                )
                kfold_sets = kfold_sets.split(data.data_id.values, data.mechanism)
            else:
                kfold_sets = model_selection.KFold(n_splits=kfolds, shuffle=shuffle)
                kfold_sets = kfold_sets.split(data.data_id.values)

            self.kfold_sets = {
                i: {
                    "train": [*map(int, list(val[0]))],
                    "validate": [*map(int, list(val[1]))],
                }
                for i, val in enumerate(kfold_sets)
            }
        else:
            if isinstance(load_kfold, dict):
                self.kfold_sets = load_kfold
            else:
                with open(load_kfold) as f:
                    self.kfold_sets = json.load(f)

        if kfold_save:
            with open(kfold_save, "w") as f:
                json.dump(self.kfold_sets, f)

        for i, kfold in self.kfold_sets.items():
            df_train = data[data.data_id.isin(kfold["train"])]
            df_validate = data[data.data_id.isin(kfold["validate"])]

            self.models.append(
                SingleModel(
                    prediction_columns=self.prediction_columns,
                    df_validate=df_validate,
                    df_train=df_train,
                    model_id=str(i),
                )
            )

    def train_model(
        self,
        model_i: int,
        modelbuilder: Callable,
        model_summary: bool = False,
        model_params: dict = None,
        verbose: bool = True,
        save_folder: str = None,
        save_training: bool = False,
        save_report_log: bool = False,
        callbacks: List = None,
    ) -> None:
        """Train a single model

        model_i : int
            Index of the model to train.
        modelbuilder : Callable
            A function that returns a tensorflow model.
        model_summary : bool, optional
            Whether or not to print out the model summary, by default False
        model_params : dict, optional
            A dictionary of parameters necessary for training,
            uses default Parameters class
        verbose : bool, optional
            Whether or not to print training messages, by default True
        save_folder: str, optional
            The folder to save results in. If not specified model is not saved,
            by default none
        save_training: bool, False
            Whether or not to save training data, by default False
        save_report_log: bool, False
            Whether or not to save the real time training info in a .csv
        callbacks: List, optional
            Callbacks for tensor flow training, by default a checkpoint and csv logger
        """
        if type(callbacks) != list:
            callbacks_i = []
        else:
            callbacks_i = copy(callbacks)

        # Log values in a Pandas dataframe (by reference)
        loss_log = PandasLogger(self.models[model_i])
        callbacks_i.append(loss_log)

        if save_folder:
            # checkpoint that saves the actual best models
            save_subfolder = save_folder / f"model_{model_i}"
            checkpoint = ModelCheckpoint(
                save_subfolder / f"model_{model_i}.h5",
                save_best_only=True,
                save_freq="epoch",
                verbose=verbose,
            )
            callbacks_i.append(checkpoint)

            # Save the logs to a csv
            if save_report_log:
                callbacks_i.append(CSVLogger(save_subfolder / "log.csv"))
        else:
            save_subfolder = None

        self.models[model_i].train(
            modelbuilder,
            model_summary,
            model_params,
            verbose,
            save_subfolder,
            save_training,
            callbacks_i,
        )

        # #TODO delete loss callback, not sure if necessary
        del loss_log

    def _save_model_state(self, save_folder, model_params, save_training=True):
        save_folder = Path(save_folder)
        save_folder.mkdir(parents=True, exist_ok=True)

        with open(save_folder / "parameters.pk", "wb") as f:
            save_dict = {
                "prediction_columns": self.prediction_columns,
                "parameters": model_params,
            }

            if save_training:
                save_dict["df_input"] = self.df_input

            pk.dump(save_dict, f)

    # TODO: better handling of model params
    def train_models(
        self,
        modelbuilder: Callable,
        model_summary: bool = False,
        model_params: dict = None,
        verbose: bool = True,
        save_folder: str = None,
        save_training: bool = False,
        callbacks: List = None,
    ) -> None:
        """Train the models.

        Parameters
        ----------
        modelbuilder : Callable
            A function that returns a tensorflow model.
        model_summary : bool, optional
            Whether or not to print out the model summary, by default False
        model_params : dict, optional
            A dictionary of parameters necessary for training, by default {}
        verbose : bool, optional
            Whether or not to print training messages, by default True
        save_folder: str, optional
            The folder to save results in. If not specified model is not saved,
            by default none
        save_training: bool, False
            Whether or not to save training data, by default False
        callbacks: List, optional
            Callbacks for tensor flow training, by default a checkpoint and csv logger
        """
        if save_folder:
            save_folder = Path(save_folder)
            self._save_model_state(save_folder, model_params, save_training)

        for i, _ in enumerate(self.models):
            self.train_model(
                i,
                modelbuilder,
                model_summary,
                model_params,
                verbose,
                save_folder,
                save_training,
                callbacks,
            )

    def generate_preprocessors(
        self,
        preprocessor,
        atom_features: Callable = nfp.preprocessing.features.atom_features_v1,
        bond_features: Callable = nfp.preprocessing.features.bond_features_v2,
        batch_size=1,
        **kwargs,
    ) -> None:
        """Generate preprocessors for the models.

        Parameters
        ----------
        atom_features : Callable, optional
            A function applied to an rdkit.Atom that returns some representation (i.e.,
            string, integer) of the atom features, by default nfp.atom_features_v1
        bond_features : Callable, optional
            A function applied to an rdkit.Atom that returns some representation (i.e.,
            string, integer) of the bond features, by default nfp.bond_features_v1
        explicit_hs : bool, optional
            Whether to tell RDkit to add H's to a molecule, by default False
        batch_size : int, optional
            The batch size for the padded batch when creating a generator, by default 1
        """
        for model in self.models:
            model.generate_preprocessor(
                preprocessor, atom_features, bond_features, batch_size, **kwargs
            )

    def generate_data_scaler(self) -> None:
        """Generate a data scaler for all of the models."""
        for model in self.models:
            model.generate_data_scaler()

    def dump_training_data(self, fname: Union[Path, str]) -> None:
        """Dumps the training data to a file.

        fname : str
            The file location to save the dataset to.
        """
        with open(fname, "wb") as f:
            pk.dump(self.__dict__, f)

    @classmethod
    def load_training_data(cls, fname: Union[Path, str]) -> MultiModel:
        """Loads the output from dump_training_data.

        fname : str
            The file location of the dataset to load.
        """
        mm = cls()
        with open(fname, "rb") as f:
            data_dicts = pk.load(f)
            for key, val in data_dicts.items():
                try:
                    setattr(mm, key, val)
                except Exception:
                    print(f"Can't set attribute {key}")

        return mm


class RenameUnpickler(pk.Unpickler):
    """For handling the renaming of modules previously named polyml. Backwards compatibilty for older models.
    
    example:

    params = RenameUnpickler.load_pickle(filepath_of_picklefile)
    """

    def find_class(self, module, name):
        renamed_module = module
        if "polyml" in module:
            renamed_module = module.replace("polyml", "polyid")

        return super(RenameUnpickler, self).find_class(renamed_module, name)

    def renamed_load(file_obj):
        return RenameUnpickler(file_obj).load()

    def load_pickle(filepath):
        with open(filepath, 'rb') as file:
            return RenameUnpickler.renamed_load(file)