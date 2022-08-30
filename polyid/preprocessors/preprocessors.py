from functools import partial
from inspect import getmembers
from pdb import pm
from random import random
from typing import Callable, Dict, List, Union

import networkx as nx
import numpy as np
import pandas as pd
import scipy.stats as spstats
import tensorflow as tf
from nfp.preprocessing.mol_preprocessor import SmilesPreprocessor
from nfp.preprocessing.tokenizer import Tokenizer
from numpy import linspace, log, log10
from sklearn.preprocessing import KBinsDiscretizer, StandardScaler


class PolymerPreprocessor(SmilesPreprocessor):
    """Preprocessor with pandas df rows as inputs."""

    def create_nx_graph(self, row: pd.Series, **kwargs) -> nx.DiGraph:
        return super().create_nx_graph(row.smiles_polymer, **kwargs)


# class GlobalPreprocessor(PolymerPreprocessor):
#     """Preprocessor using custom rows."""

#     def __init__(self, custom_properties: Dict, **kwargs,) -> None:
#         super(PolymerPreprocessor, self).__init__(**kwargs)

#         self.custom_properties = custom_properties

#     def create_nx_graph(self, row: pd.Series, **kwargs) -> nx.DiGraph:
#         nx_graph = super().create_nx_graph(row, **kwargs)
#         for custom_property in self.custom_properties:
#             nx_graph.graph[custom_property] = row[custom_property]

#         return nx_graph

#     def get_graph_features(self, graph_data: dict) -> Dict[str, np.ndarray]:
#         return {"pm": graph_data["pm"]}

#     @property
#     def output_signature(self) -> Dict[str, tf.TensorSpec]:
#         signature = super().output_signature
#         for custom_property, custom_property_dtype in self.custom_properties.items():
#             signature[custom_property] = tf.TensorSpec(
#                 shape=tf.TensorShape([]), dtype=custom_property_dtype
#             )
#         return signature

#     @property
#     def padding_values(self) -> Dict[str, tf.constant]:
#         padding_values = super().padding_values
#         padding_values["pm"] = tf.constant(0, dtype=tf.float16)
#         for custom_property, custom_property_dtype in self.custom_properties.items():
#             padding_values[custom_property] = tf.TensorSpec(
#                 shape=tf.TensorShape([]), dtype=custom_property_dtype
#             )
#         return padding_values


class PmPreprocessor(PolymerPreprocessor):
    """Preprocesses with the inclusion of a Pm column."""

    def create_nx_graph(self, row: pd.Series, **kwargs) -> nx.DiGraph:
        nx_graph = super().create_nx_graph(row)
        nx_graph.graph["pm"] = row.pm
        return nx_graph

    def get_graph_features(self, graph_data: dict) -> Dict[str, np.ndarray]:
        return {"pm": graph_data["pm"]}

    @property
    def output_signature(self) -> Dict[str, tf.TensorSpec]:
        signature = super().output_signature
        signature["pm"] = tf.TensorSpec(shape=tf.TensorShape([]), dtype="float32")
        return signature

    @property
    def padding_values(self) -> Dict[str, tf.constant]:
        padding_values = super().padding_values
        padding_values["pm"] = tf.constant(0, dtype=tf.float16)
        return padding_values


class PmRandomPreprocessor(PolymerPreprocessor):
    """preprocesses using a pm column"""

    def create_nx_graph(self, row: pd.Series, **kwargs) -> nx.DiGraph:
        nx_graph = super().create_nx_graph(row)
        nx_graph.graph["pm"] = row.pm
        return nx_graph

    def get_graph_features(self, graph_data: dict) -> Dict[str, np.ndarray]:
        pm = random()
        return {"pm": pm}

    @property
    def output_signature(self) -> Dict[str, tf.TensorSpec]:
        signature = super().output_signature
        signature["pm"] = tf.TensorSpec(shape=tf.TensorShape([]), dtype="float32")
        return signature

    @property
    def padding_values(self) -> Dict[str, tf.constant]:
        padding_values = super().padding_values
        padding_values["pm"] = tf.constant(0, dtype=tf.float16)
        return padding_values


class PmMnPreprocessor(PolymerPreprocessor):
    """preprocesses using continuous pm and mn column"""

    def create_nx_graph(self, row: pd.Series, **kwargs) -> nx.DiGraph:
        nx_graph = super().create_nx_graph(row)
        nx_graph.graph["pm"] = row.pm
        nx_graph.graph["mn"] = row.Mn
        return nx_graph

    def get_graph_features(self, graph_data: dict) -> Dict[str, np.ndarray]:
        return {"pm": graph_data["pm"], "mn": graph_data["mn"]}

    @property
    def output_signature(self) -> Dict[str, tf.TensorSpec]:
        signature = super().output_signature
        signature["pm"] = tf.TensorSpec(shape=tf.TensorShape([]), dtype="float32")
        signature["mn"] = tf.TensorSpec(shape=tf.TensorShape([]), dtype="float32")
        return signature

    @property
    def padding_values(self) -> Dict[str, tf.constant]:
        padding_values = super().padding_values
        padding_values["pm"] = tf.constant(0, dtype=tf.float16)
        padding_values["mn"] = tf.constant(0, dtype=tf.float32)
        return padding_values


class PmMnExistPreprocessor(PolymerPreprocessor):
    """preprocesses using a pm and mn column"""

    def __init__(self, p_mask_na: float = 0.0, **kwargs):
        super(PolymerPreprocessor, self).__init__(**kwargs)
        self.p_mask_na = p_mask_na

    def create_nx_graph(self, row: pd.Series, **kwargs) -> nx.DiGraph:
        nx_graph = super().create_nx_graph(row)
        nx_graph.graph["pm"] = row.pm

        val = row.Mn
        if (
            isinstance(val, (int, float, complex))
            and not isinstance(val, bool)
            and not np.isnan(val)
            and np.isfinite(val)
        ):
            nx_graph.graph["mn_data"] = 1
        else:
            nx_graph.graph["mn_data"] = 0

        return nx_graph

    def get_graph_features(self, graph_data: dict) -> Dict[str, np.ndarray]:
        mn_data = graph_data["mn_data"]
        pm = graph_data["pm"]

        return {"pm": graph_data["pm"], "mn_data": mn_data}

    @property
    def output_signature(self) -> Dict[str, tf.TensorSpec]:
        signature = super().output_signature
        signature["pm"] = tf.TensorSpec(shape=tf.TensorShape([]), dtype="float32")
        signature["mn_data"] = tf.TensorSpec(shape=tf.TensorShape([]), dtype="float32")
        return signature

    @property
    def padding_values(self) -> Dict[str, tf.constant]:
        padding_values = super().padding_values
        padding_values["pm"] = tf.constant(0, dtype=tf.float16)
        padding_values["mn_data"] = tf.constant(0, dtype=tf.float32)
        return padding_values


class PmMnRandomPreprocessor(PolymerPreprocessor):
    """preprocesses using a pm and mn column"""

    def create_nx_graph(self, row: pd.Series, **kwargs) -> nx.DiGraph:
        nx_graph = super().create_nx_graph(row)
        nx_graph.graph["pm"] = row.pm

        return nx_graph

    def get_graph_features(self, graph_data: dict) -> Dict[str, np.ndarray]:
        p = random()
        if p < 0.5:
            mn_data = 0
        else:
            mn_data = 1

        return {"pm": graph_data["pm"], "mn_data": mn_data}

    @property
    def output_signature(self) -> Dict[str, tf.TensorSpec]:
        signature = super().output_signature
        signature["pm"] = tf.TensorSpec(shape=tf.TensorShape([]), dtype="float32")
        signature["mn_data"] = tf.TensorSpec(shape=tf.TensorShape([]), dtype="float32")
        return signature

    @property
    def padding_values(self) -> Dict[str, tf.constant]:
        padding_values = super().padding_values
        padding_values["pm"] = tf.constant(0, dtype=tf.float16)
        padding_values["mn_data"] = tf.constant(0, dtype=tf.float32)
        return padding_values


class PmBinPreprocessor(PolymerPreprocessor):
    """preprocesses using a pm column and binning it"""

    def __init__(self, n_bins: int = 10, **kwargs):
        super(PolymerPreprocessor, self).__init__(**kwargs)
        self.n_bins = n_bins
        self.bins = linspace(0, 1, n_bins + 1)

    def create_nx_graph(self, row: pd.Series, **kwargs) -> nx.DiGraph:
        nx_graph = super().create_nx_graph(row)
        nx_graph.graph["pm"] = row.pm
        return nx_graph

    def get_graph_features(self, graph_data: dict) -> Dict[str, np.ndarray]:
        pm = graph_data["pm"]
        if isinstance(pm, (int, float, complex)) and not isinstance(pm, bool):
            pm = float(pm)
            if pm == 0:
                bin_i = 1
            elif pm == 1:
                bin_i = self.n_bins
            elif pm > 0 and pm < 1:
                for i, right_val in enumerate(self.bins):
                    if pm < right_val:
                        bin_i = i
                        break
            else:  # pm values outside of [0, 1]
                bin_i = 0
        else:  # pm values outside of [0, 1]
            bin_i = 0

        return {"pm_bin": bin_i, "pm": pm}

    @property
    def output_signature(self) -> Dict[str, tf.TensorSpec]:
        signature = super().output_signature
        signature["pm"] = tf.TensorSpec(shape=tf.TensorShape([]), dtype="float16")
        signature["pm_bin"] = tf.TensorSpec(shape=tf.TensorShape([]), dtype="int64")

        return signature

    @property
    def padding_values(self) -> Dict[str, tf.constant]:
        padding_values = super().padding_values
        padding_values["pm"] = tf.constant(0, dtype=tf.float16)
        padding_values["pm_bin"] = tf.constant(0, dtype=tf.int64)

        return padding_values


class WeightBinPreprocessor(PolymerPreprocessor):
    """preprocesses using a Mw and Mn column and binning it"""

    def __init__(
        self,
        n_bins: int = 20,
        max_mn: float = 10000000,
        max_mw: float = 10000000,
        method: Union[str, Callable] = "log10",
        mn_lam: float = 0,
        mw_lam: float = 0,
        **kwargs,
    ):
        super(PolymerPreprocessor, self).__init__(**kwargs)
        self.n_bins = n_bins

        self.max_mn = max_mn
        self.max_mw = max_mw

        self.max_mn_scaled = None
        self.max_mw_scaled = None

        self.mw_bins = None
        self.mn_bins = None

        self.method = method

        self.mn_lam = mn_lam
        self.mw_lam = mw_lam

        if method == "box-cox":
            assert self.mn_lam, "No mn_lam specified"
            assert self.mw_lam, "No mw_lam specified"

    def create_nx_graph(self, row: pd.Series, **kwargs) -> nx.DiGraph:
        nx_graph = super().create_nx_graph(row)
        nx_graph.graph["pm"] = row.pm
        nx_graph.graph["mn"] = row.Mn
        nx_graph.graph["mw"] = row.Mw

        return nx_graph

    def _scale_data(self, mn, mw):
        return_mn = True if mn else False
        return_mw = True if mw else False

        if isinstance(self.method, Callable):
            mn_scaled = self.method(mn)
            mw_scaled = self.method(mw)

            self.max_mn_scaled = self.method(self.max_mn)
            self.max_mw_scaled = self.method(self.max_mw)

        elif self.method == "log10":
            mn_scaled = log10(mn)
            mw_scaled = log10(mw)

            self.max_mn_scaled = log10(self.max_mn)
            self.max_mw_scaled = log10(self.max_mw)

        elif self.method == "log":
            mn_scaled = log(mn)
            mw_scaled = log(mw)

            self.max_mn_scaled = log(self.max_mn)
            self.max_mw_scaled = log(self.max_mw)

        elif self.method == "box-cox":
            mn_scaled, self.max_mn_scaled = spstats.boxcox(
                [mn, self.max_mn], lmbda=self.mn_lam
            )
            mw_scaled, self.max_mw_scaled = spstats.boxcox(
                [mw, self.max_mw], lmbda=self.mw_lam
            )

        elif self.method == "linear":
            mn_scaled = mn
            mw_scaled = mw

        else:
            assert False, "Specify a valid scaling method."

        if not return_mn:
            mn_scaled = None
        if not return_mw:
            mw_scaled = None

        return mn_scaled, mw_scaled

    def get_graph_features(self, graph_data: dict) -> Dict[str, np.ndarray]:
        pm = graph_data["pm"]
        mn = graph_data["mn"]
        mw = graph_data["mw"]

        mn_scaled, mw_scaled = self._scale_data(mn, mw)

        mn_bin_i = self._get_bin_number(mn_scaled, self.max_mn_scaled, 0)
        mw_bin_i = self._get_bin_number(mw_scaled, self.max_mw_scaled, 0)

        return {"mn_bin": mn_bin_i, "mw_bin": mw_bin_i, "pm": pm}

    def _get_bin_number(self, val, max_val, min_val):
        if (
            isinstance(val, (int, float, complex))
            and not isinstance(val, bool)
            and not np.isnan(val)
            and np.isfinite(val)
        ):
            if val > max_val:
                bin_i = self.n_bins + 1
            else:
                binwidth = (max_val - min_val) / self.n_bins
                bin_i = int((val - min_val) / binwidth) + 1
        else:
            bin_i = 0

        return bin_i

    @property
    def output_signature(self) -> Dict[str, tf.TensorSpec]:
        signature = super().output_signature
        signature["pm"] = tf.TensorSpec(shape=tf.TensorShape([]), dtype="float16")
        signature["mw_bin"] = tf.TensorSpec(shape=tf.TensorShape([]), dtype="int64")
        signature["mn_bin"] = tf.TensorSpec(shape=tf.TensorShape([]), dtype="int64")

        return signature

    @property
    def padding_values(self) -> Dict[str, tf.constant]:
        padding_values = super().padding_values
        padding_values["pm"] = tf.constant(0, dtype=tf.float16)
        padding_values["mw_bin"] = tf.constant(0, dtype=tf.int64)
        padding_values["mn_bin"] = tf.constant(0, dtype=tf.int64)

        return padding_values


class WeightDiscPreprocessor(PolymerPreprocessor):
    """preprocesses using a Mw and Mn column and binning it. Uses KBinsDiscretizer and accepts encoding and strategy args."""

    def __init__(
        self,
        n_bins: int = 20,
        encode="ordinal",
        strategy="quantile",
        p_mask_na: float = 0,
        **kwargs,
    ):
        super(PolymerPreprocessor, self).__init__(**kwargs)
        self.n_bins = n_bins
        self.p_mask_na = p_mask_na

        self.training_mn = []
        self.training_mw = []
        self.train = False

        # Track whether or not to refit
        self._new_since_trained = False
        self.mn_kbd = KBinsDiscretizer(
            n_bins=self.n_bins, encode=encode, strategy=strategy
        )
        self.mw_kbd = KBinsDiscretizer(
            n_bins=self.n_bins, encode=encode, strategy=strategy
        )

    def create_nx_graph(self, row: pd.Series, **kwargs) -> nx.DiGraph:
        nx_graph = super().create_nx_graph(row)
        nx_graph.graph["pm"] = row.pm
        nx_graph.graph["mn"] = row.Mn
        nx_graph.graph["mw"] = row.Mw

        return nx_graph

    def __call__(
        self,
        structure,
        *args,
        train: bool = False,
        max_num_nodes=None,
        max_num_edges=None,
        **kwargs,
    ) -> Dict[str, np.ndarray]:
        """Convert an input graph structure into a featurized set of node, edge,
         and graph-level features.

        Parameters
        ----------
        structure
            An input graph structure (i.e., molecule, crystal, etc.)
        train
            A training flag passed to `Tokenizer` member attributes
        max_num_nodes
            A size attribute passed to `get_node_features`, defaults to the
            number of nodes in the current graph
        max_num_edges
            A size attribute passed to `get_edge_features`, defaults to the
            number of edges in the current graph
        kwargs
            Additional features or parameters passed to `construct_nx_graph`

        Returns
        -------
        Dict[str, np.ndarray]
            A dictionary of key, array pairs as a single sample.
        """
        nx_graph = self.create_nx_graph(structure, *args, **kwargs)

        max_num_edges = len(nx_graph.edges) if max_num_edges is None else max_num_edges
        assert (
            len(nx_graph.edges) <= max_num_edges
        ), "max_num_edges too small for given input"

        max_num_nodes = len(nx_graph.nodes) if max_num_nodes is None else max_num_nodes
        assert (
            len(nx_graph.nodes) <= max_num_nodes
        ), "max_num_nodes too small for given input"

        # Make sure that Tokenizer classes are correctly initialized
        for _, tokenizer in getmembers(self, lambda x: type(x) == Tokenizer):
            tokenizer.train = train

        node_features = self.get_node_features(nx_graph.nodes(data=True), max_num_nodes)
        edge_features = self.get_edge_features(nx_graph.edges(data=True), max_num_edges)
        graph_features = self.get_graph_features(nx_graph.graph, train=train)
        connectivity = self.get_connectivity(nx_graph, max_num_edges)

        return {**node_features, **edge_features, **graph_features, **connectivity}

    def get_graph_features(
        self, graph_data: dict, train: bool
    ) -> Dict[str, np.ndarray]:
        pm = graph_data["pm"]
        mn = graph_data["mn"]
        mw = graph_data["mw"]

        # Train data
        if train:
            if mn:
                self.training_mn.append(mn)
            if mw:
                self.training_mw.append(mw)

            self._new_since_trained = True

            mn_bin_i = None
            mw_bin_i = None

        else:
            if self._new_since_trained:
                self._new_since_trained = False

                # Not sure how to format this best in np, so use pandas
                training_mn = np.array(self.training_mn)
                training_mn = training_mn[np.isfinite(training_mn)]
                df_mn = pd.DataFrame(training_mn)

                training_mw = np.array(self.training_mw)
                training_mw = training_mw[np.isfinite(training_mw)]
                df_mw = pd.DataFrame(training_mw)

                self.mn_kbd.fit(df_mn)
                self.mw_kbd.fit(df_mw)

            mn_bin_i = self._get_bin_number(mn, self.mn_kbd)
            mw_bin_i = self._get_bin_number(mw, self.mw_kbd)

            # Mask if not training with some probability
            p = random()
            if p < self.p_mask_na:
                mn_bin_i = 0
                mw_bin_i = 0
                mw = None
                mn = None

        return {"mn_bin": mn_bin_i, "mw_bin": mw_bin_i, "pm": pm, "mw": mw, "mn": mn}

    def _get_bin_number(self, val, kbd):
        # TODO there has to be a better way of checking a number is a number...
        if (
            isinstance(val, (int, float, complex))
            and not isinstance(val, bool)
            and not np.isnan(val)
            and np.isfinite(val)
        ):
            if val > kbd.bin_edges_[-0][-1]:
                bin_i = self.n_bins + 2
            else:
                bin_i = kbd.transform([[val]])[0, 0] + 1
        else:
            bin_i = 0

        return bin_i

    @property
    def output_signature(self) -> Dict[str, tf.TensorSpec]:
        signature = super().output_signature
        signature["pm"] = tf.TensorSpec(shape=tf.TensorShape([]), dtype="float16")
        signature["mw_bin"] = tf.TensorSpec(shape=tf.TensorShape([]), dtype="int64")
        signature["mn_bin"] = tf.TensorSpec(shape=tf.TensorShape([]), dtype="int64")
        signature["mn"] = tf.TensorSpec(shape=tf.TensorShape([]), dtype="float64")
        signature["mw"] = tf.TensorSpec(shape=tf.TensorShape([]), dtype="float64")

        return signature

    @property
    def padding_values(self) -> Dict[str, tf.constant]:
        padding_values = super().padding_values
        padding_values["pm"] = tf.constant(0, dtype=tf.float16)
        padding_values["mw_bin"] = tf.constant(0, dtype=tf.int64)
        padding_values["mn_bin"] = tf.constant(0, dtype=tf.int64)
        padding_values["mw"] = tf.constant(0, dtype=tf.int64)
        padding_values["mn"] = tf.constant(0, dtype=tf.int64)

        return padding_values


class PmMnScaledPreprocessor(PolymerPreprocessor):
    """preprocesses by scaling pm and mn"""

    def __init__(
        self,
        scale_pm=False,
        scale_mn=True,
        **kwargs,
    ):
        super(PolymerPreprocessor, self).__init__(**kwargs)

        self.scale_pm = scale_pm
        self.scale_mn = scale_mn
        self.train = True

        self.training_mn = []
        self.training_pm = []

        # Track whether or not to refit
        self._new_since_trained = False
        self.mn_scaler = StandardScaler()
        self.pm_scaler = StandardScaler()

    def __call__(
        self,
        structure,
        *args,
        train: bool = False,
        max_num_nodes=None,
        max_num_edges=None,
        **kwargs,
    ) -> Dict[str, np.ndarray]:
        """Convert an input graph structure into a featurized set of node, edge,
         and graph-level features.

        Parameters
        ----------
        structure
            An input graph structure (i.e., molecule, crystal, etc.)
        train
            A training flag passed to `Tokenizer` member attributes
        max_num_nodes
            A size attribute passed to `get_node_features`, defaults to the
            number of nodes in the current graph
        max_num_edges
            A size attribute passed to `get_edge_features`, defaults to the
            number of edges in the current graph
        kwargs
            Additional features or parameters passed to `construct_nx_graph`

        Returns
        -------
        Dict[str, np.ndarray]
            A dictionary of key, array pairs as a single sample.
        """
        nx_graph = self.create_nx_graph(structure, *args, **kwargs)

        max_num_edges = len(nx_graph.edges) if max_num_edges is None else max_num_edges
        assert (
            len(nx_graph.edges) <= max_num_edges
        ), "max_num_edges too small for given input"

        max_num_nodes = len(nx_graph.nodes) if max_num_nodes is None else max_num_nodes
        assert (
            len(nx_graph.nodes) <= max_num_nodes
        ), "max_num_nodes too small for given input"

        # Make sure that Tokenizer classes are correctly initialized
        for _, tokenizer in getmembers(self, lambda x: type(x) == Tokenizer):
            tokenizer.train = train

        node_features = self.get_node_features(nx_graph.nodes(data=True), max_num_nodes)
        edge_features = self.get_edge_features(nx_graph.edges(data=True), max_num_edges)
        graph_features = self.get_graph_features(nx_graph.graph, train=train)
        connectivity = self.get_connectivity(nx_graph, max_num_edges)

        return {**node_features, **edge_features, **graph_features, **connectivity}

    def create_nx_graph(self, row: pd.Series, **kwargs) -> nx.DiGraph:
        nx_graph = super().create_nx_graph(row)
        nx_graph.graph["pm"] = row.pm
        nx_graph.graph["mn"] = row.Mn

        return nx_graph

    def get_graph_features(self, graph_data: dict, train) -> Dict[str, np.ndarray]:
        pm = graph_data["pm"]
        mn = graph_data["mn"]

        # Train data
        if train:
            if mn:
                self.training_mn.append(mn)
                self.training_pm.append(pm)

            self._new_since_trained = True

            pm_scaled = None
            mn_scaled = None

        else:
            if self._new_since_trained:
                self._new_since_trained = False

                # Not sure how to format this best in np, so use pandas
                training_mn = np.array(self.training_mn)
                training_mn = training_mn[np.isfinite(training_mn)]

                df_mn = pd.DataFrame(training_mn)
                df_pm = pd.DataFrame(self.training_pm)

                self.mn_scaler.fit(df_mn)
                self.pm_scaler.fit(df_pm)

            if self.scale_pm:
                pm_scaled = self.pm_scaler.transform([[pm]])[0, 0]
            else:
                pm_scaled = pm

            if self.scale_mn:
                mn_scaled = self.mn_scaler.transform([[mn]])[0, 0]
            else:
                mn_scaled = mn

        return {"mn": mn_scaled, "pm": pm_scaled}

    @property
    def output_signature(self) -> Dict[str, tf.TensorSpec]:
        signature = super().output_signature
        signature["pm"] = tf.TensorSpec(shape=tf.TensorShape([]), dtype="float64")
        signature["mn"] = tf.TensorSpec(shape=tf.TensorShape([]), dtype="float64")

        return signature

    @property
    def padding_values(self) -> Dict[str, tf.constant]:
        padding_values = super().padding_values
        padding_values["pm"] = tf.constant(0, dtype=tf.float64)
        padding_values["mn"] = tf.constant(0, dtype=tf.float64)

        return padding_values
