# coding: utf-8

from __future__ import annotations

import os
import json
import time
import random
from collections import defaultdict
from getpass import getuser
from copy import deepcopy
from dataclasses import dataclass
from typing import Callable

import numpy as np
import tensorflow as tf

from util import (
    epsilon, load_sample, phi_mpi_to_pi, calc_new_columns, get_device, L2Metric, EarlyStopping, ReduceLROnPlateau,
)
from util import debug_layer  # noqa
from custom_layers import CustomEmbeddingLayer
from multi_dataset import MultiDataset


this_dir = os.path.dirname(os.path.realpath(__file__))

# whether to use a gpu
use_gpu: bool = True
# forces deterministic behavior on gpus, which can be slower, but it is observed on some gpus that weird numeric effects
# can occur (e.g. all batches are fine, and then one batch leads to a tensor being randomly transposed, or operations
# not being applied at all), and whether the flag is needed or not might also depend on the tf and cuda version
deterministic_ops: bool = True
# run in eager mode (for proper debuggin, also consider decorating methods in question with @util.debug_layer)
eager_mode: bool = False
# limit the cpu to a reduced number of threads
limit_cpus: bool | int = False
# profile the training
run_profiler: bool = False
# data directory
data_dir: str = os.environ["TN_SKIMS_2017"]
# cache dir for data
cache_dir: str | None = os.path.join(os.environ["TN_DATA_BASE"], "cache")
# where tensorboard logs should be written
tensorboard_dir: str | None = os.getenv("TN_TENSORBOARD_DIR", f"/tmp/{getuser()}/tensorboard")
# model save dir
model_dir: str = os.getenv("TN_MODEL_DIR", os.path.join(this_dir, "models"))
# fallback model save dir (in case kerberos permissions were lost in the meantime)
model_fallback_dir: str | None = f"/tmp/{getuser()}/models"

# apply settings
if use_gpu and deterministic_ops:
    tf.config.experimental.enable_op_determinism()
device = get_device(device="gpu" if use_gpu else "cpu", num_device=0)
if limit_cpus:
    tf.config.threading.set_intra_op_parallelism_threads(int(limit_cpus))
    tf.config.threading.set_inter_op_parallelism_threads(int(limit_cpus))
if eager_mode:
    # note: running the following with False would still trigger partial eager mode in keras
    tf.config.run_functions_eagerly(eager_mode)

# common column configs (TODO: they should live in a different file)
dynamic_columns = {
    "DeepMET_ResolutionTune_phi": (("DeepMET_ResolutionTune_px", "DeepMET_ResolutionTune_py"), (lambda x, y: np.arctan2(y, x))),
    "met_dphi": (("met_phi", "DeepMET_ResolutionTune_phi"), (lambda a, b: phi_mpi_to_pi(a - b))),
    "dmet_resp_px": (("DeepMET_ResponseTune_px", "DeepMET_ResponseTune_py", "DeepMET_ResolutionTune_phi"), (lambda x, y, p: np.cos(-p) * x - np.sin(-p) * y)),
    "dmet_resp_py": (("DeepMET_ResponseTune_px", "DeepMET_ResponseTune_py", "DeepMET_ResolutionTune_phi"), (lambda x, y, p: np.sin(-p) * x + np.cos(-p) * y)),
    "dmet_reso_px": (("DeepMET_ResolutionTune_px", "DeepMET_ResolutionTune_py", "DeepMET_ResolutionTune_phi"), (lambda x, y, p: np.cos(-p) * x - np.sin(-p) * y)),
    "dmet_reso_py": (("DeepMET_ResolutionTune_px", "DeepMET_ResolutionTune_py", "DeepMET_ResolutionTune_phi"), (lambda x, y, p: np.sin(-p) * x + np.cos(-p) * y)),
    "met_px": (("met_et", "met_dphi"), (lambda a, b: a * np.cos(b))),
    "met_py": (("met_et", "met_dphi"), (lambda a, b: a * np.sin(b))),
    "dau1_dphi": (("dau1_phi", "DeepMET_ResolutionTune_phi"), (lambda a, b: phi_mpi_to_pi(a - b))),
    "dau2_dphi": (("dau2_phi", "DeepMET_ResolutionTune_phi"), (lambda a, b: phi_mpi_to_pi(a - b))),
    "genNu1_dphi": (("genNu1_phi", "DeepMET_ResolutionTune_phi"), (lambda a, b: phi_mpi_to_pi(a - b))),
    "genNu2_dphi": (("genNu2_phi", "DeepMET_ResolutionTune_phi"), (lambda a, b: phi_mpi_to_pi(a - b))),
    "dau1_px": (("dau1_pt", "dau1_dphi"), (lambda a, b: a * np.cos(b))),
    "dau1_py": (("dau1_pt", "dau1_dphi"), (lambda a, b: a * np.sin(b))),
    "dau1_pz": (("dau1_pt", "dau1_eta"), (lambda a, b: a * np.sinh(b))),
    "dau1_m": (("dau1_px", "dau1_py", "dau1_pz", "dau1_e"), (lambda x, y, z, e: np.sqrt(e ** 2 - (x ** 2 + y ** 2 + z ** 2)))),
    "dau2_px": (("dau2_pt", "dau2_dphi"), (lambda a, b: a * np.cos(b))),
    "dau2_py": (("dau2_pt", "dau2_dphi"), (lambda a, b: a * np.sin(b))),
    "dau2_pz": (("dau2_pt", "dau2_eta"), (lambda a, b: a * np.sinh(b))),
    "dau2_m": (("dau2_px", "dau2_py", "dau2_pz", "dau2_e"), (lambda x, y, z, e: np.sqrt(e ** 2 - (x ** 2 + y ** 2 + z ** 2)))),
    "ditau_deltaphi": (("dau1_dphi", "dau2_dphi"), (lambda a, b: np.abs(phi_mpi_to_pi(a - b)))),
    "ditau_deltaeta": (("dau1_eta", "dau2_eta"), (lambda a, b: np.abs(a - b))),
    "genNu1_px": (("genNu1_pt", "genNu1_dphi"), (lambda a, b: a * np.cos(b))),
    "genNu1_py": (("genNu1_pt", "genNu1_dphi"), (lambda a, b: a * np.sin(b))),
    "genNu1_pz": (("genNu1_pt", "genNu1_eta"), (lambda a, b: a * np.sinh(b))),
    "genNu2_px": (("genNu2_pt", "genNu2_dphi"), (lambda a, b: a * np.cos(b))),
    "genNu2_py": (("genNu2_pt", "genNu2_dphi"), (lambda a, b: a * np.sin(b))),
    "genNu2_pz": (("genNu2_pt", "genNu2_eta"), (lambda a, b: a * np.sinh(b))),
    "bjet1_dphi": (("bjet1_phi", "DeepMET_ResolutionTune_phi"), (lambda a, b: phi_mpi_to_pi(a - b))),
    "bjet1_px": (("bjet1_pt", "bjet1_dphi"), (lambda a, b: a * np.cos(b))),
    "bjet1_py": (("bjet1_pt", "bjet1_dphi"), (lambda a, b: a * np.sin(b))),
    "bjet1_pz": (("bjet1_pt", "bjet1_eta"), (lambda a, b: a * np.sinh(b))),
    "bjet2_dphi": (("bjet2_phi", "DeepMET_ResolutionTune_phi"), (lambda a, b: phi_mpi_to_pi(a - b))),
    "bjet2_px": (("bjet2_pt", "bjet2_dphi"), (lambda a, b: a * np.cos(b))),
    "bjet2_py": (("bjet2_pt", "bjet2_dphi"), (lambda a, b: a * np.sin(b))),
    "bjet2_pz": (("bjet2_pt", "bjet2_eta"), (lambda a, b: a * np.sinh(b))),
}

# possible values of categorical inputs (TODO: they should live in a different file)
embedding_expected_inputs = {
    "pairType": [0, 1, 2],
    "dau1_decayMode": [-1, 0, 1, 10, 11],  # -1 for e/mu
    "dau2_decayMode": [0, 1, 10, 11],
    "dau1_charge": [-1, 1],
    "dau2_charge": [-1, 1],
    "spin": [0, 2],
}


@dataclass
class ActivationSetting:
    # name of the activation as understood by tf.keras.layers.Activation
    name: str
    # name of the kernel initializer as understood by tf.keras.layers.Dense
    weight_init: str
    # whether to apply batch normalization before or after the activation (and if at all)
    batch_norm: tuple[bool, bool]
    # name of the dropout layer under tf.keras.layers
    dropout_name: str = "Dropout"


# setting for typical activations (TODO: they should live in a different file)
activation_settings = {
    "elu": ActivationSetting("ELU", "he_uniform", (True, False)),
    "relu": ActivationSetting("ReLU", "he_uniform", (False, True)),
    "prelu": ActivationSetting("PReLU", "he_normal", (True, False)),
    "selu": ActivationSetting("selu", "lecun_normal", (False, False), "AlphaDropout"),
    "tanh": ActivationSetting("tanh", "glorot_normal", (True, False)),
    "softmax": ActivationSetting("softmax", "glorot_normal", (True, False)),
    "swish": ActivationSetting("swish", "glorot_uniform", (True, False)),
}


@dataclass
class Sample:
    name: str
    class_weight: float
    labels: list[int]
    spin: int = -1
    mass: float = -1.0


def main(
    model_name: str = "test_lreg50_elu_bn",
    data_dir: str = data_dir,
    cache_dir: str | None = cache_dir,
    tensorboard_dir: str | None = tensorboard_dir,
    model_dir: str = model_dir,
    model_fallback_dir: str | None = model_fallback_dir,
    samples: list[Sample] = [
        # Sample("SKIM_ggF_Radion_m250", 1.0, [1, 0, 0], 0, 250.0),
        # Sample("SKIM_ggF_Radion_m260", 1.0, [1, 0, 0], 0, 260.0),
        # Sample("SKIM_ggF_Radion_m270", 1.0, [1, 0, 0], 0, 270.0),
        # Sample("SKIM_ggF_Radion_m280", 1.0, [1, 0, 0], 0, 280.0),
        # Sample("SKIM_ggF_Radion_m300", 1.0, [1, 0, 0], 0, 300.0),
        Sample("SKIM_ggF_Radion_m320", 1.0, [1, 0, 0], 0, 320.0),
        Sample("SKIM_ggF_Radion_m350", 1.0, [1, 0, 0], 0, 350.0),
        Sample("SKIM_ggF_Radion_m400", 1.0, [1, 0, 0], 0, 400.0),
        Sample("SKIM_ggF_Radion_m450", 1.0, [1, 0, 0], 0, 450.0),
        Sample("SKIM_ggF_Radion_m500", 1.0, [1, 0, 0], 0, 500.0),
        Sample("SKIM_ggF_Radion_m550", 1.0, [1, 0, 0], 0, 550.0),
        Sample("SKIM_ggF_Radion_m600", 1.0, [1, 0, 0], 0, 600.0),
        Sample("SKIM_ggF_Radion_m650", 1.0, [1, 0, 0], 0, 650.0),
        Sample("SKIM_ggF_Radion_m700", 1.0, [1, 0, 0], 0, 700.0),
        Sample("SKIM_ggF_Radion_m750", 1.0, [1, 0, 0], 0, 750.0),
        Sample("SKIM_ggF_Radion_m800", 1.0, [1, 0, 0], 0, 800.0),
        Sample("SKIM_ggF_Radion_m850", 1.0, [1, 0, 0], 0, 850.0),
        Sample("SKIM_ggF_Radion_m900", 1.0, [1, 0, 0], 0, 900.0),
        Sample("SKIM_ggF_Radion_m1000", 1.0, [1, 0, 0], 0, 1000.0),
        Sample("SKIM_ggF_Radion_m1250", 1.0, [1, 0, 0], 0, 1250.0),
        Sample("SKIM_ggF_Radion_m1500", 1.0, [1, 0, 0], 0, 1500.0),
        Sample("SKIM_ggF_Radion_m1750", 1.0, [1, 0, 0], 0, 1750.0),
        # Sample("SKIM_ggF_BulkGraviton_m250", 1.0, [1, 0, 0], 2, 250.0),
        # Sample("SKIM_ggF_BulkGraviton_m260", 1.0, [1, 0, 0], 2, 260.0),
        # Sample("SKIM_ggF_BulkGraviton_m270", 1.0, [1, 0, 0], 2, 270.0),
        # Sample("SKIM_ggF_BulkGraviton_m280", 1.0, [1, 0, 0], 2, 280.0),
        # Sample("SKIM_ggF_BulkGraviton_m300", 1.0, [1, 0, 0], 2, 300.0),
        Sample("SKIM_ggF_BulkGraviton_m320", 1.0, [1, 0, 0], 2, 320.0),
        Sample("SKIM_ggF_BulkGraviton_m350", 1.0, [1, 0, 0], 2, 350.0),
        Sample("SKIM_ggF_BulkGraviton_m400", 1.0, [1, 0, 0], 2, 400.0),
        Sample("SKIM_ggF_BulkGraviton_m450", 1.0, [1, 0, 0], 2, 450.0),
        Sample("SKIM_ggF_BulkGraviton_m500", 1.0, [1, 0, 0], 2, 500.0),
        Sample("SKIM_ggF_BulkGraviton_m550", 1.0, [1, 0, 0], 2, 550.0),
        Sample("SKIM_ggF_BulkGraviton_m600", 1.0, [1, 0, 0], 2, 600.0),
        Sample("SKIM_ggF_BulkGraviton_m650", 1.0, [1, 0, 0], 2, 650.0),
        Sample("SKIM_ggF_BulkGraviton_m700", 1.0, [1, 0, 0], 2, 700.0),
        Sample("SKIM_ggF_BulkGraviton_m750", 1.0, [1, 0, 0], 2, 750.0),
        Sample("SKIM_ggF_BulkGraviton_m800", 1.0, [1, 0, 0], 2, 800.0),
        Sample("SKIM_ggF_BulkGraviton_m850", 1.0, [1, 0, 0], 2, 850.0),
        Sample("SKIM_ggF_BulkGraviton_m900", 1.0, [1, 0, 0], 2, 900.0),
        Sample("SKIM_ggF_BulkGraviton_m1000", 1.0, [1, 0, 0], 2, 1000.0),
        Sample("SKIM_ggF_BulkGraviton_m1250", 1.0, [1, 0, 0], 2, 1250.0),
        Sample("SKIM_ggF_BulkGraviton_m1500", 1.0, [1, 0, 0], 2, 1500.0),
        Sample("SKIM_ggF_BulkGraviton_m1750", 1.0, [1, 0, 0], 2, 1750.0),
        Sample("SKIM_DY_amc_incl", 1.0, [0, 1, 0], -1, -1.0),
        Sample("SKIM_TT_fullyLep", 1.0, [0, 0, 1], -1, -1.0),
        Sample("SKIM_TT_semiLep", 1.0, [0, 0, 1], -1, -1.0),
        # Sample("SKIM_GluGluHToTauTau", 1.0, [0, 0, 0, 0, 1, 0], -1, -1.0),
        # Sample("SKIM_ttHToTauTau", 1.0, [0, 0, 0, 1], -1, -1.0),
    ],
    # additional columns to load
    extra_columns: list[str] = [
        "EventNumber",
    ],
    # selections to apply before training
    selections: list[tuple[tuple[str, ...], Callable]] = [
        (("nbjetscand",), (lambda a: a > 1)),
        (("pairType",), (lambda a: a < 3)),
        (("nleps",), (lambda a: a == 0)),
        (("isOS",), (lambda a: a == 1)),
        (("dau2_deepTauVsJet",), (lambda a: a >= 5)),
        (
            ("pairType", "dau1_iso", "dau1_eleMVAiso", "dau1_deepTauVsJet"),
            (lambda a, b, c, d: (((a == 0) & (b < 0.15)) | ((a == 1) & (c == 1)) | ((a == 2) & (d >= 5)))),
        ),
    ],
    # categorical input features for the network
    cat_input_names: list[str] = [
        "pairType", "dau1_decayMode", "dau2_decayMode", "dau1_charge", "dau2_charge",
    ],
    # continuous input features to the network
    cont_input_names: list[str] = [
        "met_px", "met_py", "dmet_resp_px", "dmet_resp_py", "dmet_reso_px",
        "met_cov00", "met_cov01", "met_cov11",
        "ditau_deltaphi", "ditau_deltaeta",
        *[
            f"dau{i}_{feat}"
            for i in [1, 2]
            for feat in ["px", "py", "pz", "e", "dxy", "dz", "iso"]
        ],
        *[
            f"bjet{i}_{feat}"
            for i in [1, 2]
            for feat in [
                "px", "py", "pz", "e", "btag_deepFlavor", "cID_deepFlavor", "pnet_bb", "pnet_cc", "pnet_b", "pnet_c",
                "pnet_g", "pnet_uds", "pnet_pu", "pnet_undef", "HHbtag",
            ]
        ],
    ],
    # number of layers and units
    units: list[int] = [125] * 5,
    # dimension of the embedding layer output will be embedding_output_dim x len(cat_input_names)
    embedding_output_dim: int = 5,
    # activation function after each hidden layer
    activation: str = "elu",
    # scale for the l2 loss term (which is already normalized to the number of weights)
    l2_norm: float = 50.0,
    # dropout percentage
    dropout_rate: float = 0.0,
    # batch norm between layers
    batch_norm: bool = True,
    # batch size
    batch_size: int = 4096,
    # learning rate to start with
    initial_learning_rate: float = 3e-3,
    # half the learning rate if the validation loss hasn't improved in this many validation steps
    learning_rate_patience: int = 8,
    # how even the learning rate is halfed before training is stopped
    max_learning_rate_reductions: int = 6,
    # stop training if the validation loss hasn't improved since this many validation steps
    early_stopping_patience: int = 10,
    # maximum number of epochs to even cap early stopping
    max_epochs: int = 10000,
    # how frequently to calulcate the validation loss
    validate_every: int = 500,
    # add the generator spin for the signal samples as categorical input -> network parameterized in spin
    parameterize_spin: bool = True,
    # add the generator mass for the signal samples as continuous input -> network parameterized in mass
    parameterize_mass: bool = True,
    # number of the fold to train for (0-9, events with event numbers ending in the fold number are not used at all!)
    fold_index: int = 0,
    # how many of the 9 training folds to use for validation
    validation_folds: int = 3,
    # seed for random number generators
    seed: int = 1,
) -> tuple[tf.keras.Model, str]:
    # some checks
    assert "spin" not in cat_input_names
    assert "mass" not in cont_input_names
    assert 0 <= fold_index <= 9
    assert 1 <= validation_folds <= 8

    # set the seed to everything (Python, NumPy, TensorFlow, Keras)
    tf.keras.utils.set_random_seed(seed)

    # determine which columns to read
    columns_to_read = set()
    for name in cont_input_names + cat_input_names:
        columns_to_read.add(name)
    for sel_columns, _ in selections:
        columns_to_read |= set(sel_columns)
    columns_to_read |= set(extra_columns)
    # expand dynamic columns, keeping track of those that are needed
    all_dyn_names = set(dynamic_columns)
    dyn_names = set()
    while (to_expand := columns_to_read & all_dyn_names):
        for name in to_expand:
            columns_to_read |= set(dynamic_columns[name][0])
        columns_to_read -= to_expand
        dyn_names |= to_expand

    # order dynamic columns to be added
    all_dyn_names = list(dynamic_columns)
    dyn_names = sorted(dyn_names, key=all_dyn_names.index)

    # get lists of embedded feature values
    possible_cont_input_values = [deepcopy(embedding_expected_inputs[name]) for name in cat_input_names]

    # scan samples and their labels to construct relative weights such that each class starts with equal importance
    labels_to_samples = defaultdict(list)
    for sample in samples:
        labels_to_samples[tuple(sample.labels)].append(sample.name)

    # keep track of spins, masses, number of events per sample, and relative batch weights per sample
    spins: set[int] = set()
    masses: set[float] = set()
    all_n_events: list[int] = []
    batch_weights: list[float] = []

    # lists for collection data to be forwarded into the MultiDataset
    cont_inputs_train, cont_inputs_valid = [], []
    cat_inputs_train, cat_inputs_valid = [], []
    labels_train, labels_valid = [], []
    event_weights_train, event_weights_valid = [], []

    # prepare fold indices to use
    train_fold_indices: list[int] = [i for i in range(10) if i != fold_index]
    valid_fold_indices: list[int] = []
    while len(valid_fold_indices) < validation_folds:
        valid_fold_indices.append(train_fold_indices.pop(random.randint(0, len(train_fold_indices) - 1)))

    # helper to flatten rec arrays
    flatten_rec = lambda r, t: r.astype([(n, t) for n in r.dtype.names], copy=False).view(t).reshape((-1, len(r.dtype)))

    # loop through samples
    for sample in samples:
        rec, event_weights = load_sample(
            data_dir,
            sample.name,
            sample.class_weight,
            list(columns_to_read),
            selections,
            # maxevents=10000,
            cache_dir=cache_dir,
        )
        all_n_events.append(n_events := len(event_weights))

        # compute the batch weight, i.e. the weight that ensure that each class is equally represented in each batch
        batch_weights.append(1 / len(labels_to_samples[tuple(sample.labels)]))

        # add dynamic columns
        rec = calc_new_columns(rec, {name: dynamic_columns[name] for name in dyn_names})

        # prepare arrays
        cont_inputs = flatten_rec(rec[cont_input_names], np.float32)
        cat_inputs = flatten_rec(rec[cat_input_names], np.int32)
        labels = np.array([sample.labels] * n_events, dtype=np.float32)

        # add spin and mass if given
        if parameterize_mass:
            if sample.mass > -1:
                masses.add(float(sample.mass))
            cont_inputs = np.append(cont_inputs, (np.ones(n_events, dtype=np.float32) * sample.mass)[:, None], axis=1)
        if parameterize_spin:
            if sample.spin > -1:
                spins.add(int(sample.spin))
            cat_inputs = np.append(cat_inputs, (np.ones(n_events, dtype=np.int32) * sample.spin)[:, None], axis=1)

        # training and validation mask using event number and fold indices
        last_digit = rec["EventNumber"] % 10
        train_mask = np.any(last_digit[..., None] == train_fold_indices, axis=1)
        valid_mask = np.any(last_digit[..., None] == valid_fold_indices, axis=1)

        # fill dataset lists
        cont_inputs_train.append(cont_inputs[train_mask])
        cont_inputs_valid.append(cont_inputs[valid_mask])

        cat_inputs_train.append(cat_inputs[train_mask])
        cat_inputs_valid.append(cat_inputs[valid_mask])

        labels_train.append(labels[train_mask])
        labels_valid.append(labels[valid_mask])

        event_weights_train.append(event_weights[train_mask][..., None])
        event_weights_valid.append(event_weights[valid_mask][..., None])

    # determine contiuous input means and variances
    cont_input_means = (
        np.sum(np.concatenate([inp * bw / len(inp) for inp, bw in zip(cont_inputs_train, batch_weights)]), axis=0) /
        sum(batch_weights)
    )
    cont_input_vars = (
        np.sum(np.concatenate([inp**2 * bw / len(inp) for inp, bw in zip(cont_inputs_train, batch_weights)]), axis=0) /
        sum(batch_weights)
    ) - cont_input_means**2

    # handle masses
    masses = tf.constant(sorted(masses), dtype=tf.float32)
    mass_probs = tf.ones_like(masses)  # all masses equally probable when sampling for backgrounds
    if parameterize_mass:
        assert len(masses) > 0
        cont_input_names.append("mass")
        # replace mean and var with unweighted values
        cont_input_means[-1] = np.mean(masses.numpy())
        cont_input_vars[-1] = np.var(masses.numpy())

    # handle spins
    spins = tf.constant(sorted(spins), dtype=tf.int32)
    spin_probs = tf.ones_like(spins, dtype=tf.float32)  # all spins equally probable when sampling for backgrounds
    if parameterize_spin:
        assert len(spins) > 0
        cat_input_names.append("spin")
        # add to possible embedding values
        possible_cont_input_values.append(embedding_expected_inputs["spin"])

    with device:
        # live transformation of inputs to inject spin and mass for backgrounds
        @tf.function
        def transform(cont_inputs, cat_inputs, labels, weights):
            if parameterize_mass:
                idxs_0 = tf.where(cont_inputs[:, -1] < 0)
                idxs_1 = (cont_inputs.shape[1] - 1) * tf.ones_like(idxs_0)
                idxs = tf.concat([idxs_0, idxs_1], axis=-1)
                random_masses = tf.gather(masses, tf.random.categorical([mass_probs], tf.shape(idxs_0)[0]))[0]
                cont_inputs = tf.tensor_scatter_nd_update(cont_inputs, idxs, random_masses)
            if parameterize_spin:
                idxs_0 = tf.where(cat_inputs[:, -1] < 0)
                idxs_1 = (cat_inputs.shape[1] - 1) * tf.ones_like(idxs_0)
                idxs = tf.concat([idxs_0, idxs_1], axis=-1)
                random_spins = tf.gather(spins, tf.random.categorical([spin_probs], tf.shape(idxs_0)[0]))[0]
                cat_inputs = tf.tensor_scatter_nd_update(cat_inputs, idxs, random_spins)
            return cont_inputs, cat_inputs, labels, weights

        # build datasets
        dataset_train = MultiDataset(
            data=zip(zip(cont_inputs_train, cat_inputs_train, labels_train, event_weights_train), batch_weights),
            batch_size=batch_size,
            kind="train",
            transform_data=transform,
            seed=seed,
        )
        dataset_valid = MultiDataset(
            data=zip(zip(cont_inputs_valid, cat_inputs_valid, labels_valid, event_weights_valid), batch_weights),
            batch_size=batch_size,
            kind="valid",
            yield_valid_rest=True,
            transform_data=transform,
            seed=seed,
        )

        # create the model
        model = create_model(
            n_cont_inputs=len(cont_input_names),
            n_cat_inputs=len(cat_input_names),
            n_classes=labels_train[0].shape[1],
            embedding_expected_inputs=possible_cont_input_values,
            embedding_output_dim=embedding_output_dim,
            cont_input_means=cont_input_means,
            cont_input_vars=cont_input_vars,
            units=units,
            activation=activation,
            batch_norm=batch_norm,
            l2_norm=l2_norm,
            dropout_rate=dropout_rate,
        )

        # compile
        model.compile(
            loss="categorical_crossentropy",
            optimizer=tf.keras.optimizers.Adam(learning_rate=initial_learning_rate),
            metrics=[
                tf.keras.metrics.CategoricalCrossentropy(name="ce"),
                L2Metric(model, name="l2"),
                tf.keras.metrics.CategoricalAccuracy(name="acc"),
            ],
            jit_compile=False,
            run_eagerly=eager_mode,
        )

        # helper to connect the lr scheduler and early stopping through the skip_monitoring_fn callback
        def skip_es_monitoring(es_callback: EarlyStopping, epoch: int) -> bool:
            final_lr_reached = abs(float(model.optimizer.lr) - lr_callback.min_lr) <= epsilon
            # skip when the final lr was not yet reached
            skip = not final_lr_reached
            # when not skipping for the first time, port the best model weights over from the lr scheduler
            if not skip and not es_callback.monitoring_active:
                print(f"Epoch {epoch}: porting best metric and model weights from lr to es callback")
                es_callback.best = lr_callback.best
                es_callback.best_weights = lr_callback.best_weights
                es_callback.best_epoch = -1
            return not final_lr_reached

        # callbacks
        fit_callbacks = [
            # drop learning rate when plateau is reached
            lr_callback := ReduceLROnPlateau(
                model=model,
                monitor="val_ce",
                factor=0.5,
                patience=learning_rate_patience,
                min_lr=initial_learning_rate * 0.5**max_learning_rate_reductions,
                min_delta=0.0001,
                verbose=1,
            ),
            # early stopping (patience set to twice the patience passed to the ReduceLROnPlateau callback)
            EarlyStopping(
                monitor="val_ce",
                patience=2 * learning_rate_patience,
                restore_best_weights=True,
                min_delta=0.0001,
                verbose=1,
                skip_monitoring_fn=skip_es_monitoring,
            ),
            # tensorboard
            tf.keras.callbacks.TensorBoard(
                log_dir=os.path.join(tensorboard_dir, model_name),
                histogram_freq=1,
                write_graph=True,
                profile_batch=(500, 1500) if run_profiler else 0,
            ) if tensorboard_dir else None,
        ]

        # some logs
        model.summary()
        print(f"training samples  : {len(dataset_train):_}")
        print(f"validation samples: {len(dataset_valid):_}")

        # training
        t_start = time.perf_counter()
        try:
            model.fit(
                x=dataset_train.create_keras_generator(input_names=["cont_input", "cat_input"]),
                validation_data=dataset_valid.create_keras_generator(input_names=["cont_input", "cat_input"]),
                shuffle=False,  # already shuffled
                epochs=max_epochs,
                steps_per_epoch=validate_every,
                validation_freq=1,
                validation_steps=dataset_valid.batches_per_cycle,
                callbacks=list(filter(None, fit_callbacks)),
            )
            t_end = time.perf_counter()
        except KeyboardInterrupt:
            t_end = time.perf_counter()
            print("\n\ndetected manual interrupt!\n")
            print("type 's' to gracefully stop training and save the model,")
            try:
                inp = input("or any other key to terminate directly without saving: ")
            except KeyboardInterrupt:
                inp = ""
            if inp != "s":
                print("model not saved")
                return
            print("")
        print(f"training took {t_end - t_start:.2f} seconds")

        # perform one final validation round for verification of the best model
        print("performing final round of validation")
        results_valid = model.evaluate(
            x=dataset_valid.create_keras_generator(input_names=["cont_input", "cat_input"]),
            batch_size=batch_size,
            steps=dataset_valid.batches_per_cycle,
            return_dict=True,
        )

        # model saving
        def save_model(path):
            print(f"saving model at {path}")

            if not os.path.exists(os.path.dirname(path)):
                os.makedirs(os.path.dirname(path))

            # save the model
            tf.keras.saving.save_model(model, path, overwrite=True, save_format="tf")

            # save an accompanying json file with hyper-parameters, input names and other info
            meta = {
                "model_name": model_name,
                "sample_names": [sample.name for sample in samples],
                "input_names": {
                    "cont": cont_input_names,
                    "cat": cat_input_names,
                },
                "fold_index": fold_index,
                "validation_folds": validation_folds,
                "architecture": {
                    "units": units,
                    "embedding_output_dim": embedding_output_dim,
                    "activation": activation,
                    "l2_norm": l2_norm,
                    "drop_out": dropout_rate,
                    "batch_norm": batch_norm,
                    "batch_size": batch_size,
                    "initial_learning_rate": initial_learning_rate,
                    "final_learning_rate": float(model.optimizer.lr.numpy()),
                    "parameterize_spin": parameterize_spin,
                    "parameterize_mass": parameterize_mass,
                },
                "result": {
                    **results_valid,
                    "steps_trained": int(model.optimizer.iterations.numpy()),
                },
            }
            with open(os.path.join(path, "meta.json"), "w") as f:
                json.dump(meta, f, indent=4)

            return path

        # save at actual location, fallback to tmp dir
        try:
            model_path = save_model(os.path.join(model_dir, model_name))
        except OSError as e:
            if not model_fallback_dir:
                raise e
            print(f"saving at default path failed: {e}")
            model_path = save_model(os.path.join(model_fallback_dir, model_name))

    return model, model_path

    # TODOs (in order):
    # - proper model names
    # - other samples
    # - read inputs from root files directly (making use of cache)
    # - binary or multi-class?
    # - prepend tauNN
    # - ensembling encapuslation after trainings for simple export, should also include spin and mass loop
    # - skip connections, hyper-opt, symmetric CCE and group weight


# functional model builder for later use with hyperparameter optimization tools
# via https://www.tensorflow.org/tutorials/keras/keras_tuner
def create_model(
    *,
    n_cont_inputs: int,
    n_cat_inputs: int,
    n_classes: int,
    embedding_expected_inputs: list[list[int]],
    embedding_output_dim: int,
    cont_input_means: np.ndarray,
    cont_input_vars: np.ndarray,
    units: list[int],
    activation: str,
    batch_norm: bool,
    l2_norm: float,
    dropout_rate: float,
):
    # checks
    assert len(embedding_expected_inputs) == n_cat_inputs
    assert len(cont_input_means) == len(cont_input_vars) == n_cont_inputs
    assert len(units) > 0

    # get activation settings
    act_settings: ActivationSetting = activation_settings[activation]

    # prepare l2 regularization, use a dummy value as it is replaced after the model is built
    l2_reg = tf.keras.regularizers.l2(1.0) if l2_norm > 0 else None

    # input layers
    x_cont = tf.keras.Input(n_cont_inputs, dtype=tf.float32, name="cont_input")
    x_cat = tf.keras.Input(n_cat_inputs, dtype=tf.int32, name="cat_input")

    # normalize continuous inputs
    norm_layer = tf.keras.layers.Normalization(mean=cont_input_means, variance=cont_input_vars, name="norm")
    a = norm_layer(x_cont)

    # embedding layer
    if n_cat_inputs > 0:
        embedding_layer = CustomEmbeddingLayer(
            output_dim=embedding_output_dim,
            expected_inputs=embedding_expected_inputs,
            name="cat_embedding",
        )
        embed_cat = embedding_layer(x_cat)

        # combine with continuous inputs
        a = tf.keras.layers.Concatenate(name="concat")([a, embed_cat])

    # add layers programatically
    for i, n_units in enumerate(units, 1):
        # dense
        dense_layer = tf.keras.layers.Dense(
            n_units,
            use_bias=True,
            kernel_initializer=act_settings.weight_init,
            kernel_regularizer=l2_reg,
            name=f"dense_{i}")
        a = dense_layer(a)

        # batch norm before activation if requested
        batchnorm_layer = tf.keras.layers.BatchNormalization(dtype=tf.float32, name=f"norm_{i}")
        if batch_norm and act_settings.batch_norm[0]:
            a = batchnorm_layer(a)

        # activation
        a = tf.keras.layers.Activation(act_settings.name, name=f"act_{i}")(a)

        # batch norm after activation if requested
        if batch_norm and act_settings.batch_norm[1]:
            a = batchnorm_layer(a)

        # add random unit dropout
        if dropout_rate:
            dropout_cls = getattr(tf.keras.layers, act_settings.dropout_name)
            a = dropout_cls(dropout_rate, name=f"do_{i}")(a)

    # add the output layer
    output_layer = tf.keras.layers.Dense(
        n_classes,
        activation="softmax",
        use_bias=True,
        kernel_initializer=activation_settings["softmax"].weight_init,
        kernel_regularizer=l2_reg,
        name="output",
    )
    y = output_layer(a)

    # build the model
    model = tf.keras.Model(inputs=[x_cont, x_cat], outputs=[y], name="bbtautau_classifier")

    # normalize the l2 regularization to the number of weights in dense layers
    if l2_norm > 0:
        n_weights = sum(map(
            tf.keras.backend.count_params,
            [layer.kernel for layer in model.layers if isinstance(layer, tf.keras.layers.Dense)],
        ))
        for layer in model.layers:
            if isinstance(layer, tf.keras.layers.Dense) and layer.kernel_regularizer is not None:
                layer.kernel_regularizer.l2[...] = l2_norm / n_weights

    return model


if __name__ == "__main__":
    main()
