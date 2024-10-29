from __future__ import annotations

import os
import re
import json
import time
import shutil
import hashlib
import pickle
from collections import defaultdict
from getpass import getuser
from copy import deepcopy
from operator import or_
from functools import reduce
# from typing import Any

import numpy as np
import awkward as ak
import tensorflow as tf
# import tensorflow_probability as tfp
from law.util import human_duration
from tabulate import tabulate
import scipy.stats
import glob
from tqdm import tqdm
import seaborn as sns
import pandas as pd
import mplhep as hep
hep.style.use("CMS")



from tautaunn.multi_dataset import MultiDataset
from tautaunn.tf_util import (
    get_device, ClassificationModelWithValidationBuffers, L2Metric, ReduceLRAndStop, EmbeddingEncoder,
    LivePlotWriter, metric_class_factory, BetterModel, CustomMetricSum,
)
from tautaunn.util import load_sample_root, calc_new_columns, create_model_name, transform_data_dir_cache
from tautaunn.config import Sample, activation_settings, dynamic_columns, embedding_expected_inputs
from tautaunn.output_scaling_layer import CustomOutputScalingLayer



import matplotlib.pyplot as plt
from sklearn.metrics import (roc_curve, RocCurveDisplay, roc_auc_score, confusion_matrix, auc)
from sklearn.preprocessing import label_binarize
from config import label_sets
from argparse import ArgumentParser

from matplotlib.colors import ListedColormap
petroff10 = ListedColormap([
    "#3f90da", "#ffa90e", "#bd1f01", "#94a4a2", "#832db6", "#a96b59",
    "#e76300", "#b9ac70", "#717581", "#92dadd",
])

process_info = {
    "HH": {
        "color": petroff10.colors[0],
    },
    "TT": {
        "color": petroff10.colors[1],
    },
    "DY": {
        "color": petroff10.colors[2],
    },
}

# Definiere den Pfad zum Cache-Verzeichnis
#cached_data = "/gpfs/dust/cms/user/yamralim/taunn_data/training_cache/alldata_b591dd4502.pkl"
cached_data = "/gpfs/dust/cms/user/yamralim/taunn_data/training_cache/alldata_15c27d0e4e_evaluation.pkl"



# Modellpfad e1, f1
#spin_model_path_seed1 = '/gpfs/dust/cms/user/yamralim/taunn_data/store/RegTraining/dev1_spin_loss/tautaureg_PSnew_baseline_LSmulti3_SSdefault_FSdefault_daurot_composite-default_extended_pair_ED10_LU5x128+4x128_CTfcn_ACTelu_BNy_LT50_DO0_BS2048_OPadamw_LR1.0e-03_YEARy_SPINy_MASSy_CW1_ZW0.01_FI0_SD1/'
#spin_model_path_seed2 = '/gpfs/dust/cms/user/yamralim/taunn_data/store/RegTraining/dev1_spin_loss/tautaureg_PSnew_baseline_LSmulti3_SSdefault_FSdefault_daurot_composite-default_extended_pair_ED10_LU5x128+4x128_CTfcn_ACTelu_BNy_LT50_DO0_BS2048_OPadamw_LR1.0e-03_YEARy_SPINy_MASSy_CW1_ZW0.01_FI0_SD2/'
#spin_model_path_seed3 = '/gpfs/dust/cms/user/yamralim/taunn_data/store/RegTraining/dev1_spin_loss/tautaureg_PSnew_baseline_LSmulti3_SSdefault_FSdefault_daurot_composite-default_extended_pair_ED10_LU5x128+4x128_CTfcn_ACTelu_BNy_LT50_DO0_BS2048_OPadamw_LR1.0e-03_YEARy_SPINy_MASSy_CW1_ZW0.01_FI0_SD1/'
#spin_model_path_seed4 = '/gpfs/dust/cms/user/yamralim/taunn_data/store/RegTraining/dev1_spin_loss/tautaureg_PSnew_baseline_LSmulti3_SSdefault_FSdefault_daurot_composite-default_extended_pair_ED10_LU5x128+4x128_CTfcn_ACTelu_BNy_LT50_DO0_BS2048_OPadamw_LR1.0e-03_YEARy_SPINy_MASSy_CW1_ZW0.01_FI0_SD1/'
#spin_model_path_seed5 = '/gpfs/dust/cms/user/yamralim/taunn_data/store/RegTraining/dev1_spin_loss/tautaureg_PSnew_baseline_LSmulti3_SSdefault_FSdefault_daurot_composite-default_extended_pair_ED10_LU5x128+4x128_CTfcn_ACTelu_BNy_LT50_DO0_BS2048_OPadamw_LR1.0e-03_YEARy_SPINy_MASSy_CW1_ZW0.01_FI0_SD1/'


#e0.5
#spin_model_path_seed2 = '/gpfs/dust/cms/user/yamralim/taunn_data/store/RegTraining/dev1_spin_loss/tautaureg_PSnew_baseline_LSmulti3_SSdefault_FSdefault_daurot_composite-default_extended_pair_ED10_LU5x128+4x128_CTfcn_ACTelu_BNy_LT50_DO0_BS2048_OPadamw_LR1.0e-03_YEARy_SPINy_MASSy_CW1_ZW0.01_FI0_SD2'

#e0
#spin_model_path_seed1 = '/gpfs/dust/cms/user/yamralim/taunn_data/store/RegTraining/dev1_spin_loss_e0/tautaureg_PSnew_baseline_LSmulti3_SSdefault_FSdefault_daurot_composite-default_extended_pair_ED10_LU5x128+4x128_CTfcn_ACTelu_BNy_LT50_DO0_BS2048_OPadamw_LR1.0e-03_YEARy_SPINy_MASSy_CW1_ZW0.01_FI0_SD1'

#f0
# spin_model_path_seed1 = '/gpfs/dust/cms/user/yamralim/taunn_data/store/RegTraining/dev1_spin_loss_f0/tautaureg_PSnew_baseline_LSmulti3_SSdefault_FSdefault_daurot_composite-default_extended_pair_ED10_LU5x128+4x128_CTfcn_ACTelu_BNy_LT50_DO0_BS2048_OPadamw_LR1.0e-03_YEARy_SPINy_MASSy_CW1_ZW0.01_FI0_SD1'
# spin_model_path_seed2 = '/gpfs/dust/cms/user/yamralim/taunn_data/store/RegTraining/dev1_spin_loss_f0/tautaureg_PSnew_baseline_LSmulti3_SSdefault_FSdefault_daurot_composite-default_extended_pair_ED10_LU5x128+4x128_CTfcn_ACTelu_BNy_LT50_DO0_BS2048_OPadamw_LR1.0e-03_YEARy_SPINy_MASSy_CW1_ZW0.01_FI0_SD2'
# spin_model_path_seed3 = '/gpfs/dust/cms/user/yamralim/taunn_data/store/RegTraining/dev1_spin_loss_f0/tautaureg_PSnew_baseline_LSmulti3_SSdefault_FSdefault_daurot_composite-default_extended_pair_ED10_LU5x128+4x128_CTfcn_ACTelu_BNy_LT50_DO0_BS2048_OPadamw_LR1.0e-03_YEARy_SPINy_MASSy_CW1_ZW0.01_FI0_SD3'
# spin_model_path_seed4 = '/gpfs/dust/cms/user/yamralim/taunn_data/store/RegTraining/dev1_spin_loss_f0/tautaureg_PSnew_baseline_LSmulti3_SSdefault_FSdefault_daurot_composite-default_extended_pair_ED10_LU5x128+4x128_CTfcn_ACTelu_BNy_LT50_DO0_BS2048_OPadamw_LR1.0e-03_YEARy_SPINy_MASSy_CW1_ZW0.01_FI0_SD4'
#spin_model_path_seed5 = '/gpfs/dust/cms/user/yamralim/taunn_data/store/RegTraining/dev1_spin_loss_f0/tautaureg_PSnew_baseline_LSmulti3_SSdefault_FSdefault_daurot_composite-default_extended_pair_ED10_LU5x128+4x128_CTfcn_ACTelu_BNy_LT50_DO0_BS2048_OPadamw_LR1.0e-03_YEARy_SPINy_MASSy_CW1_ZW0.01_FI0_SD5'


#learningrate0.0005
spin_model_path_seed5 = '/gpfs/dust/cms/user/yamralim/taunn_data/store/RegTraining/dev1_spin_loss_f0/tautaureg_PSnew_baseline_LSmulti3_SSdefault_FSdefault_daurot_composite-default_extended_pair_ED10_LU5x128+4x128_CTfcn_ACTelu_BNy_LT50_DO0_BS2048_OPadamw_LR5.0e-04_YEARy_SPINy_MASSy_CW1_ZW0.01_FI0_SD5'

# baseline_model_path_seed1 = '/gpfs/dust/cms/user/yamralim/taunn_data/store/RegTraining/dev1_baseline/tautaureg_PSnew_baseline_LSmulti3_SSdefault_FSdefault_daurot_composite-default_extended_pair_ED10_LU5x128+4x128_CTfcn_ACTelu_BNy_LT50_DO0_BS4096_OPadamw_LR1.0e-03_YEARy_SPINy_MASSy_CW1_ZW0.01_FI0_SD1/'
# baseline_model_path_seed2 = '/gpfs/dust/cms/user/yamralim/taunn_data/store/RegTraining/dev1_baseline_z/tautaureg_PSnew_baseline_LSmulti3_SSdefault_FSdefault_daurot_composite-default_extended_pair_ED10_LU5x128+4x128_CTfcn_ACTelu_BNy_LT50_DO0_BS4096_OPadamw_LR1.0e-03_YEARy_SPINy_MASSy_CW1_ZW0.01_FI0_SD2'
# baseline_model_path_seed3 = '/gpfs/dust/cms/user/yamralim/taunn_data/store/RegTraining/dev1_baseline_z/tautaureg_PSnew_baseline_LSmulti3_SSdefault_FSdefault_daurot_composite-default_extended_pair_ED10_LU5x128+4x128_CTfcn_ACTelu_BNy_LT50_DO0_BS4096_OPadamw_LR1.0e-03_YEARy_SPINy_MASSy_CW1_ZW0.01_FI0_SD3'
# baseline_model_path_seed4 = '/gpfs/dust/cms/user/yamralim/taunn_data/store/RegTraining/dev1_baseline_z/tautaureg_PSnew_baseline_LSmulti3_SSdefault_FSdefault_daurot_composite-default_extended_pair_ED10_LU5x128+4x128_CTfcn_ACTelu_BNy_LT50_DO0_BS4096_OPadamw_LR1.0e-03_YEARy_SPINy_MASSy_CW1_ZW0.01_FI0_SD4'
baseline_model_path_seed5 = '/gpfs/dust/cms/user/yamralim/taunn_data/store/RegTraining/dev1_baseline_z/tautaureg_PSnew_baseline_LSmulti3_SSdefault_FSdefault_daurot_composite-default_extended_pair_ED10_LU5x128+4x128_CTfcn_ACTelu_BNy_LT50_DO0_BS4096_OPadamw_LR1.0e-03_YEARy_SPINy_MASSy_CW1_ZW0.01_FI0_SD5'



zw0_model_path_seed1 = '/gpfs/dust/cms/user/yamralim/taunn_data/store/RegTraining/dev1_baseline/tautaureg_PSnew_baseline_LSmulti3_SSdefault_FSdefault_daurot_composite-default_extended_pair_ED10_LU5x128+4x128_CTfcn_ACTelu_BNy_LT50_DO0_BS4096_OPadamw_LR1.0e-03_YEARy_SPINy_MASSy_CW1_ZW0_FI0_SD1/'
# zw0_model_path_seed2 = '/gpfs/dust/cms/user/yamralim/taunn_data/store/RegTraining/dev1_baseline_z/tautaureg_PSnew_baseline_LSmulti3_SSdefault_FSdefault_daurot_composite-default_extended_pair_ED10_LU5x128+4x128_CTfcn_ACTelu_BNy_LT50_DO0_BS4096_OPadamw_LR1.0e-03_YEARy_SPINy_MASSy_CW1_ZW0_FI0_SD2'
# zw0_model_path_seed3 = '/gpfs/dust/cms/user/yamralim/taunn_data/store/RegTraining/dev1_baseline_z/tautaureg_PSnew_baseline_LSmulti3_SSdefault_FSdefault_daurot_composite-default_extended_pair_ED10_LU5x128+4x128_CTfcn_ACTelu_BNy_LT50_DO0_BS4096_OPadamw_LR1.0e-03_YEARy_SPINy_MASSy_CW1_ZW0_FI0_SD3'
# zw0_model_path_seed4 = '/gpfs/dust/cms/user/yamralim/taunn_data/store/RegTraining/dev1_baseline_z/tautaureg_PSnew_baseline_LSmulti3_SSdefault_FSdefault_daurot_composite-default_extended_pair_ED10_LU5x128+4x128_CTfcn_ACTelu_BNy_LT50_DO0_BS4096_OPadamw_LR1.0e-03_YEARy_SPINy_MASSy_CW1_ZW0_FI0_SD4'
# zw0_model_path_seed5 = '/gpfs/dust/cms/user/yamralim/taunn_data/store/RegTraining/dev1_baseline_z/tautaureg_PSnew_baseline_LSmulti3_SSdefault_FSdefault_daurot_composite-default_extended_pair_ED10_LU5x128+4x128_CTfcn_ACTelu_BNy_LT50_DO0_BS4096_OPadamw_LR1.0e-03_YEARy_SPINy_MASSy_CW1_ZW0_FI0_SD5'

def filter_list(input_list, filter_value, default_value=-1):
    return_mask = list()
    for entry in input_list:
        return_mask.append(np.where((entry[..., -1] == filter_value) | (entry[..., -1] == default_value), True, False))
    return return_mask

def load_cache(cache_path, model_path, spin=0, mass=700.):
    with open(cache_path, "rb") as f:
        (
            cont_inputs_eval,
            cat_inputs_eval,
            targets_eval,
            z_targets_eval,
            labels_eval,
            event_weights_eval,
            additional_weights_eval,
            yield_factors,
        ) = pickle.load(f)
    #from IPython import embed; embed()
    # filter relevant events
    mask_cont_list = filter_list(cont_inputs_eval, mass)
    mask_cat_list = filter_list(cat_inputs_eval, spin)
    # final set of condidtions is the 'and' of these masks
    final_criteria = list()
    for mask_cont, mask_cat in zip(mask_cont_list, mask_cat_list):
        final_criteria.append(mask_cont & mask_cat)

    def create_final_list(input_list, final_criteria, fill_value=None):
        final_list = list()
        for entry, mask in zip(input_list, final_criteria):
            tmp = entry[mask]
            if len(tmp) > 0:
                if fill_value:
                    tmp[..., -1] = fill_value
                final_list.append(tmp)
        return final_list

    evaluation_data = {
        "cat_input": tf.concat(create_final_list(cat_inputs_eval, final_criteria, spin), axis=0),
        "cont_input": tf.concat(create_final_list(cont_inputs_eval, final_criteria, mass), axis=0)
    }

    # 3. Modell laden
    loaded_model = tf.keras.models.load_model(model_path, compile=False)

    # 4. Modell evaluieren
    cont_input = evaluation_data['cont_input']
    evaluation_results = loaded_model(evaluation_data)

    ind_1_px = 5
    ind_1_py = 6
    ind_1_pz = 7
    ind_1_e = 8

    ind_2_px = 9
    ind_2_py = 10
    ind_2_pz = 11
    ind_2_e = 12

    dau1_px = cont_input[:,ind_1_px]
    dau1_py = cont_input[:,ind_1_py]
    dau1_pz = cont_input[:,ind_1_pz]
    dau1_e = cont_input[:,ind_1_e]
    dau1 = np.concatenate([dau1_px[:, np.newaxis], dau1_py[:, np.newaxis], dau1_pz[:, np.newaxis], dau1_e[:, np.newaxis]], axis=1)

    dau2_px = cont_input[:,ind_2_px]
    dau2_py = cont_input[:,ind_2_py]
    dau2_pz = cont_input[:,ind_2_pz]
    dau2_e = cont_input[:,ind_2_e]
    dau2 = np.concatenate([dau2_px[:, np.newaxis], dau2_py[:, np.newaxis], dau2_pz[:, np.newaxis], dau2_e[:, np.newaxis]], axis=1)

    bjet1_px = cont_input[:,13]
    bjet1_py = cont_input[:,14]
    bjet1_pz = cont_input[:,15]
    bjet1_e = cont_input[:,16]
    bjet1 = np.concatenate([bjet1_px[:, np.newaxis], bjet1_py[:, np.newaxis], bjet1_pz[:, np.newaxis], bjet1_e[:, np.newaxis]], axis=1)


    bjet2_px = cont_input[:,21]
    bjet2_py = cont_input[:,22]
    bjet2_pz = cont_input[:,23]
    bjet2_e = cont_input[:,24]
    bjet2 = np.concatenate([bjet2_px[:, np.newaxis], bjet2_py[:, np.newaxis], bjet2_pz[:, np.newaxis], bjet2_e[:, np.newaxis]], axis=1)




    regression_output = evaluation_results['regression_output_hep']

    classification_output_softmax = evaluation_results['classification_output_softmax']
    classification_output = evaluation_results['classification_output']
    #from IPython import embed; embed(header="anfang")


    # Wahrscheinlichkeiten und echte Labels für ROC
    y_true = tf.concat(create_final_list(labels_eval, final_criteria), axis=0).numpy()
    y_scores_cls_softmax = classification_output_softmax.numpy()
    y_scores_reg = regression_output.numpy()
    return y_true, y_scores_cls_softmax, y_scores_reg, dau1, dau2, bjet1, bjet2

def plot_roc(y_true, y_score, label_set, used_set, file_name, **kwargs):

    fig, ax0 = plt.subplots()
    fpr_list = []
    tpr_list = []
    auc_list = []

    for ind, dataset in label_set.get(used_set).items():
        dataset_name = dataset.get("name")
        fpr, tpr, _ = roc_curve(
            y_true[:, ind],
            y_score[:, ind],
        )
        auc_value = roc_auc_score(
            y_true[:, ind],
            y_score[:, ind],
        )

        # FPR, TPR und AUC-Werte speichern
        fpr_list.append(fpr)
        tpr_list.append(tpr)
        auc_list.append(auc_value)

        ax0.plot(
            fpr,
            tpr,
            label=f"{dataset_name} vs. rest (AUC: {auc_value:0.3f})",
            color=process_info[dataset_name]["color"],
        )

    hep.cms.label("private work", data=False, lumi=None)
    ax0.legend(loc="best")
    # fig.suptitle("ROC Curve for Multi-Class Classification", size=14)
    ax0.set_xlabel("False Positive Rate")
    ax0.set_ylabel("True Positive Rate")
    fig.savefig(f"{file_name}_ROC.png")

    return fpr_list, tpr_list, auc_list






def plot_confusion_matrix(y_true, y_score, label_set, used_set, file_name, **kwargs):
    final_predicted_cls = np.argmax(y_score, axis=-1)
    final_truth_cls = np.argmax(y_true, axis=-1)
    matrix = confusion_matrix(y_true=final_truth_cls, y_pred=final_predicted_cls,
                              labels=[0, 1, 2], normalize="true")

    fig, ax = plt.subplots()
    im = ax.imshow(matrix, interpolation="none")

    labels = [dataset.get("name") for ind, dataset in label_set.get(used_set).items()]
    ax.set_xticks(np.arange(len(labels)), labels=labels)
    ax.set_yticks(np.arange(len(labels)), labels=labels)

    for i in range(len(labels)):
        for j in range(len(labels)):
            # Schriftgröße basierend auf Position in der Diagonale oder nicht
            fontsize = 24 if i == j else 20  # Erhöhe die Schriftgröße nur für Diagonalelemente
            text = ax.text(
                j, i, round(matrix[i, j], 3),
                ha="center", va="center", color="w" if matrix[i, j] < 0.5 else "black",
                fontsize=fontsize
            )

    hep.cms.label("private work", data=False, lumi=None)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    fig.savefig(f"{file_name}_confusion_matrix.png")
    return


def plot_distribution(y_true, y_score, label_set, used_set, file_name, **kwargs):
    final_truth_cls = np.argmax(y_true, axis=-1)
    this_label_set = label_set.get(used_set)

    # distribution
    for dnn_class in range(y_score.shape[1]):
        fi, ax = plt.subplots()
        hep.cms.label("private work", data=False, lumi=None)
        for j, dataset in this_label_set.items():
            label=dataset.get("name")
            ax.hist(
                y_score[final_truth_cls==j][:, dnn_class],
                bins=np.linspace(start=0, stop=1, num=51),
                label=label,
                density=True,
                histtype="step",
                color=process_info[label]["color"],
            )
            # axs[j].set_title(f"{label} prediction")

        dnn_node_name=this_label_set[dnn_class].get("name")
        ax.set_xlabel(f'Output for {dnn_node_name} node')
        ax.set_ylabel("Density")
        ax.legend(loc='upper right')

        fi.savefig(f"{file_name}_distribution_{dnn_node_name}.png")
        fi.clear()


def auc_plot_comparison(auc_lists, labels, file_name, **kwargs):

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(labels))

    # Scatterplot für jede Dataset-Liste
    colors = ['blue', 'orange', 'green']
    markers = ['o', '^', 's']
    labels_datasets = {
        "spin_SD5": 'KL classifer',
        "baseline_SD5": 'Baseline',
        "zw0_SD1": 'TautauNN',
    }
    #from IPython import embed; embed()

    for (auc_key, auc_values), color, marker in zip(auc_lists.items(), colors, markers):
        ax.scatter(
            x, auc_values,
            color=color,
            marker=marker,
            edgecolor=color,
            facecolor='none',
            s=100,
            label=labels_datasets.get(auc_key, auc_key),
        )

    # Titel und Achsenbeschriftungen
    #ax.set_title('AUC Values Comparison for Different Datasets')
    #ax.set_xlabel('Labels', fontsize=14)
    hep.cms.label("private work", data=False, lumi=None)
    ax.set_xlabel('Process', loc="center")  # X-axis label
    ax.set_ylabel('AUC', loc="center")  # Y-axis label
    # Set number of ticks for x-axis
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation='horizontal')  # Horizontale Beschriftungen
    ax.set_ylim(0.9, 1.1)
    # Legende hinzufügen
    ax.legend(loc='best')

    # Save the figure
    fig.savefig(f"{file_name}_AUC_comparision.png", bbox_inches='tight')
    plt.show()




def get_mtt(y_scores, dau1, dau2, process, file_name):

    label = ['HH', 'TT', 'DY']
    colors = {'HH': 'lightblue', 'TT': 'lightgreen', 'DY': 'lightcoral'}
    plotting_info = dict()
    # import global configuration
    plotting_info.update(process_info)

    # regressed neutrinos
    reg1_px = y_scores[:, 0]
    reg1_py = y_scores[:, 1]
    reg1_pz = y_scores[:, 2]
    reg1_e = np.sqrt(reg1_px**2 + reg1_py**2 + reg1_pz**2)

    reg2_px = y_scores[:, 3]
    reg2_py = y_scores[:, 4]
    reg2_pz = y_scores[:, 5]
    reg2_e = np.sqrt(reg2_px**2 + reg2_py**2 + reg2_pz**2)

    dau1_px = dau1[:, 0]
    dau1_py = dau1[:, 1]
    dau1_pz = dau1[:, 2]
    dau1_e = dau1[:, 3]

    dau2_px = dau2[:, 0]
    dau2_py = dau2[:, 1]
    dau2_pz = dau2[:, 2]
    dau2_e = dau2[:, 3]

    # vis + reg neutrinos (total mass)
    m_tt_reg = (
        (dau1_e + dau2_e + reg1_e + reg2_e)**2 -
        (dau1_px + dau2_px + reg1_px + reg2_px)**2 -
        (dau1_py + dau2_py + reg1_py + reg2_py)**2 -
        (dau1_pz + dau2_pz + reg1_pz + reg2_pz)**2
    )**0.5

    # Find the dominant process
    dominant_process = np.argmax(process, axis=1)

    # Create a DataFrame for easy plotting
    df = pd.DataFrame({
        'm_tt_reg': m_tt_reg,
        'process': [label[i] for i in dominant_process]
    })

    # Median und 68% Konfidenzintervall (sigma)
    process_stats = df.groupby('process')['m_tt_reg'].agg(['median']).reset_index()

    # Calculate 68% confidence interval (sigma)
    sigma_values, overlap_values, sigma_mu_values = [], [], []
    data_hh = df[df['process'] == 'HH']['m_tt_reg']
    q_low_hh, q_high_hh = np.percentile(data_hh, [16, 84])

    for process_name in process_stats['process']:
        data = df[df['process'] == process_name]['m_tt_reg']
        q_low, q_high = np.percentile(data, [16, 84])
        sigma = (q_high - q_low) / 2
        sigma_mu = sigma / process_stats[process_stats['process'] == process_name]['median'].values[0]
        overlap = ((data > q_low_hh) & (data < q_high_hh)).mean()

        sigma_values.append(sigma)
        sigma_mu_values.append(sigma_mu)
        overlap_values.append(overlap)

    process_stats['sigma'] = sigma_values
    process_stats['sigma/mu'] = sigma_mu_values
    process_stats['overlap'] = overlap_values
    # overlap_values = {
    #     'HH': overlap_hh,  # Assuming these variables are calculated
    #     'TT': overlap_tt,
    #     'DY': overlap_dy
    # }

    # Create labels with median, sigma, sigma/mu, and overlap
    for _, row in process_stats.iterrows():
        label = (f"{row['process']} (μ: {row['median']:.2f} GeV, σ: {row['sigma']:.2f} GeV, "
                 f"σ/μ: {row['sigma/mu']:.2f}, OL: {row['overlap']:.4f})")
        plotting_info[row['process']]["label"] = label

    plt.figure(figsize=(10, 8))

    # Daten plotten mit Seaborn
    for process_name, infos in plotting_info.items():
        filtered_data = df[df['process'] == process_name]
        sns.histplot(
            filtered_data,
            x="m_tt_reg",
            bins=60,
            binrange=(0, 300),
            stat="density",
            common_norm=False,
            element="step",
            color=infos["color"],
            label=infos["label"],
        )

    plt.xlabel(r"$\tau\tau$ Mass [GeV]")
    plt.ylabel("Density")

    # CMS Label
    hep.cms.label("private work", data=False, lumi=None)

    # Achsen anpassen
    plt.xlim(0, 350)
    plt.ylim(0, 0.13)

    # Legende anpassen: Schriftgröße und Position
    plt.legend(fontsize=18, loc='upper right', bbox_to_anchor=(1, 1), borderaxespad=0.)

    # Plot speichern
    plt.savefig(f"{file_name}_m_tt.png", bbox_inches='tight')  # Plot eng zuschneiden
    plt.close()

    return m_tt_reg, overlap_values, sigma_values



def overlap_plot_comparison(overlap_dict, labels, file_name='overlap_comparison_plot.png'):
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(labels))

    # Define colors and markers for different datasets
    colors = ['blue', 'red']
    markers = ['o', '^']
    labels_datasets = {
        "DY": "Dataset DY",
        "TT": "Dataset TT"
    }

    x_ticks = {
        "spin_SD5": 'KL classifer',
        "baseline_SD5": 'Baseline',
        "zw0_SD1": 'TautauNN',
    }

    x_ticks_label = [nice_name for x_tick, nice_name in x_ticks.items()]


    dy_values, tt_values = [], []
    for dataset_name, overlap_value in overlap_dict.items():
        dy_values.append(overlap_value[0])
        tt_values.append(overlap_value[-1])

    overlap_values = {
        "DY": dy_values,
        "TT": tt_values
    }

    #from IPython import embed; embed()
    # Iterate through the dictionary and plot each point
    for (dataset_name, overlap_value), color, marker in zip(overlap_values.items(), colors, markers):
        ax.scatter(
            x=x,  # Assuming you want to use the overlap keys for x-axis
            y=overlap_value,  # y-values from the overlap_dict
            color=color,
            marker=marker,
            edgecolor=color,  # Edge color for markers
            facecolor=color,  # Marker without fill
            s=100,  # Size of the markers
            label=dataset_name,  # Use custom label if exists
        )

    hep.cms.label("private work", data=False, lumi=None)
    # Additional plot settings (optional)
    ax.set_xlabel('Networks', loc="center")  # X-axis label
    ax.set_ylabel('Overlap Value', loc="center")  # Y-axis label
    ax.legend(frameon=True)  # Show legend
    ax.grid(True)  # Optional: add grid for better visualization
    ax.set_xticks(np.arange(min(x),max(x)+1))
    ax.set_xticklabels(x_ticks_label)
    ax.set_ylim(0,max(ax.get_ylim()) * 1.1)
    ax.set_xlim(-0.5,3.5)

    fig.savefig(f"{file_name}__overlap_comparison_plot.png", bbox_inches='tight')

    # Show the plot
    plt.show()


def Sigma_plot_comparison(sigma_dict, labels, file_name='overlap_comparison_plot.png'):
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(labels))

    # Define colors and markers for different datasets
    colors = ['blue', 'red', 'green']
    markers = ['o', '^', 's']
    labels_datasets = {
        "DY": "Dataset DY",
        "TT": "Dataset TT",
        "HH": "Dataset HH"
    }

    x_ticks = {
        "spin_SD5": 'KL classifer',
        "baseline_SD5": 'Baseline',
        "zw0_SD1": 'TautauNN',
    }

    x_ticks_label = [nice_name for x_tick, nice_name in x_ticks.items()]


    tt_values, dy_values, hh_values = [], [], []
    for dataset_name, sigma_value in sigma_dict.items():
        tt_values.append(sigma_value[-1])
        dy_values.append(sigma_value[0])
        hh_values.append(sigma_value[1])


    sigma_values = {
        "TT": tt_values,
        "DY": dy_values,
        "HH": hh_values

    }

    #from IPython import embed; embed()
    # Iterate through the dictionary and plot each point
    for (dataset_name, sigma_value), color, marker in zip(sigma_values.items(), colors, markers):
        ax.scatter(
            x=x,
            y=sigma_value,
            color=color,
            marker=marker,
            edgecolor=color,
            facecolor=color,
            s=100,
            label=dataset_name,
        )

    hep.cms.label("private work", data=False, lumi=None)
    # Additional plot settings (optional)
    ax.set_xlabel('Networks', loc="center")  # X-axis label
    ax.set_ylabel(r'$\sigma$ Value', loc="center")  # Y-axis label
    ax.legend(frameon=True)  # Show legend
    ax.grid(True)  # Optional: add grid for better visualization
    ax.set_xticks(np.arange(min(x),max(x)+1))
    ax.set_xticklabels(x_ticks_label)
    ax.set_ylim(0,max(ax.get_ylim()) * 1.1)
    ax.set_xlim(-0.5,3.5)

    fig.savefig(f"{file_name}__sigma_comparison_plot.png", bbox_inches='tight')

    # Show the plot
    plt.show()
from IPython import embed; embed()

def main(spin=0, mass=700., **kwargs):

    architecture_dict = {
        # "spin_SD1": spin_model_path_seed1,
        # "spin_SD2": spin_model_path_seed2,
        # "spin_SD3": spin_model_path_seed3,
        # "spin_SD4": spin_model_path_seed4,
        "spin_SD5": spin_model_path_seed5,
        # "baseline_SD1": baseline_model_path_seed1,
        # "baseline_SD2": baseline_model_path_seed2,
        # "baseline_SD3": baseline_model_path_seed3,
        # "baseline_SD4": baseline_model_path_seed4,
        "baseline_SD5": baseline_model_path_seed5,
        "zw0_SD1": zw0_model_path_seed1,
        # "zw0_SD2": zw0_model_path_seed2,
        # "zw0_SD3": zw0_model_path_seed3,
        # "zw0_SD4": zw0_model_path_seed4,
        # "zw0_SD5": zw0_model_path_seed5,
    }
    labels = ['HH', 'TT', 'DY']

    auc_lists = dict()
    overlap_dict = {}
    sigma_dict = {}


    arch_process_bar = tqdm(architecture_dict.items())
    for arch_name, model_path in arch_process_bar:
        arch_process_bar.set_description(f"Processing architecture {arch_name}")
        y_true, y_softmax, y_scores_reg, dau1, dau2, bjet1, bjet2 = load_cache(cached_data, model_path, spin=spin, mass=mass)

        fpr_list, tpr_list, auc_list = plot_roc(y_true, y_softmax, label_sets, "multi3", f"{arch_name}__mass_{mass}_spin_{spin}")
        auc_lists[arch_name] = auc_list[:]

        plot_confusion_matrix(y_true, y_softmax, label_sets, "multi3", f"{arch_name}__mass_{mass}_spin_{spin}")

        plot_distribution(y_true, y_softmax, label_sets, "multi3", f"{arch_name}__mass_{mass}_spin_{spin}")

        #get_mtt(y_scores_reg, dau1, dau2, y_true, f"{arch_name}__mass_{mass}_spin_{spin}")
        m_tt_reg, overlap_values, sigma_values = get_mtt(y_scores_reg, dau1, dau2, y_true, f"{arch_name}__mass_{mass}_spin_{spin}")
        #from IPython import embed; embed()
        overlap_dict[arch_name] = overlap_values
        sigma_dict[arch_name] = sigma_values
        #auc_lists = [auc_list_spin, auc_list_baseline, auc_list_zw0]
    auc_plot_comparison(auc_lists, labels, f"mass_{mass}_spin_{spin}")
    overlap_plot_comparison(overlap_dict, labels, f"mass_{mass}_spin_{spin}")
    Sigma_plot_comparison(sigma_dict, labels, f"mass_{mass}_spin_{spin}")
    #from IPython import embed; embed()

def parse_arguments():
    parser = ArgumentParser()

    parser.add_argument('-m', '--mass',
        help="mass hypothesis to use in evaluation",
        default=700,
        type=float,
    )
    parser.add_argument('-s', '--spin',
        help="spin hypothesis to use during evaluation",
        default=0,
        type=int,
        choices=[0, 2],
    )

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_arguments()
    main(**vars(args))


# def L_baseline(z, t):
#     """Calculates the L_baseline function for given values of z and t."""
#     term1 = z * np.arctanh((z - t) / (z + t - 2 * z * t))
#     term2 = 0.5 * np.log(np.abs(z - 1))
#     term3 = -0.5 * np.log(np.abs(t - 1))
#     return term1 + term2 + term3

# def L_derivative(z, t):
#     """Calculates the derivative of L_baseline with respect to z."""
#     # Calculate the derivative
#     term = (z - t) / (z + t - 2 * z * t)
#     dL_dz = np.arctanh(term) + z * (1 / (1 - term**2)) * (1 / (z + t - 2 * z * t)) * (1 - 2 * t)
#     return dL_dz

# # Define the z and t values for three cases
# cases = [(0.6, 0.2), (0.5, 0.5), (0.4, 0.8)]
# z_values = np.linspace(0, 1, 400)  # z values for the plots

# # Plot the function and its derivative for each case
# for i, (z, t) in enumerate(cases):
#     L_values = L_baseline(z_values, t)  # Calculate the function for different z
#     dL_dz_values = L_derivative(z_values, t)  # Calculate the derivative for different z

#     plt.figure(figsize=(10, 5))
#     plt.plot(z_values, L_values, label=f'L_baseline', color='blue')
#     plt.plot(z_values, dL_dz_values, label=f'Derivative', color='orange', linestyle='--')
#     plt.axhline(y=0, color='black', linestyle='--', lw=0.7)
#     #plt.title(f'L_baseline and its Derivative for z={z}, t={t}')
#     plt.xlabel('z values')
#     plt.ylabel('L_baseline / Derivative')
#     plt.legend()
#     plt.grid()

#     # Set the axis limits
#     plt.xlim(-0.1, 1.7)
#     plt.ylim(-1, 1)

#     # Save each plot separately
#     plt.tight_layout()
#     plt.savefig(f'L_baseline_plot_z{z}_t{t}.png')  # Save the plot as a PNG file
#     plt.show()
#     plt.close()





