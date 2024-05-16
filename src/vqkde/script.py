
import argparse
import os
from inspect import getfullargspec

import matplotlib.pyplot as plt

import numpy as np
import math
import tensorflow as tf
import qmc.tf.layers as qmc_layers
import qmc.tf.models as qmc_models

from data import load_data
from data._dmkde_data import _predict_features, _create_U_train

from models import *
from models.HEA import *
from models._raw_kde import _raw_kde

params = {
   'axes.labelsize': 12,
   'legend.fontsize': 12,
   'xtick.labelsize': 10,
   'ytick.labelsize': 10,
   'text.usetex': False,
   'figure.figsize': [7.0, 6.0]
   }

MODELS = ["raw_kde", "dmkde_qeff", "dmkde_qrff", "vqkde_qeff", "vqkde_qrff", "vqkde_qeff_hea", "vqkde_qrff_hea"]
DATASETS = ["potential_1", "potential_2", "star_eight"]

NUM_QUBITS_FFS = 5 ## set 6 for the final experiments
NUM_ANCILLA_QUBITS = 2 # set 2 for the final experiments

GAMMA_DICT = {"binomial": 2., "potential_1": 4., "potential_2": 16., "arc": 4., "star_eight": 16.}
RANDOM_STATE_QRFF_DICT = {"binomial": 324, "potential_1": 125, "potential_2": 178, "arc": 7, "star_eight": 1224}
RANDOM_STATE_QEFF_DICT = {"binomial": 3, "potential_1": 15, "potential_2": 78, "arc": 73, "star_eight": 24}
EPOCHS_DICT  = {"binomial": 0, "potential_1": 8, "potential_2": 0, "arc": 60, "star_eight": 60}
LEARNING_RATE_DICT = {"binomial": 0.0005, "potential_1": 0.0005, "potential_2": 0.005, "arc": 0.0005, "star_eight": 0.0005}
GRID_PARAMS_DICT = {
    "potential_1": {
        "x_range": (-4, 4),
        "y_range": (-4, 4),
        "x_step": 8/120,
        "y_step": 8/120
    },
    "potential_2": {
        "x_range": (-4, 4),
        "y_range": (-3, 3),
        "x_step": 8/120,
        "y_step": 6/120
    },
    "star_eight": {
        "x_range": (-7, 7),
        "y_range": (-7, 7),
        "x_step": 14/120,
        "y_step": 14/120
    }
}


def _raw_kde_exec(X_train, X_plot, X_test, GAMMA):
    raw_kde_probability_train = np.array([_raw_kde(x_temp[np.newaxis,:], X_train, GAMMA) for x_temp in X_train])
    raw_kde_probability_test = np.array([_raw_kde(x_temp[np.newaxis,:], X_train, GAMMA) for x_temp in X_test])
    raw_kde_probability = np.array([_raw_kde(x_temp[np.newaxis,:], X_train, GAMMA) for x_temp in X_plot])

    return raw_kde_probability_train, raw_kde_probability_test, raw_kde_probability


def _dmkde_classical_qeff_exec(X_train, X_test, X_plot, GAMMA, DIM_X, N_FFS, RANDOM_STATE_QEFF):
    fm_qeff_x = QFeatureMapQuantumEnhancedFF(DIM_X, dim=N_FFS, gamma=GAMMA, random_state= RANDOM_STATE_QEFF)
    qmd = qmc_models.ComplexQMDensity(fm_qeff_x, N_FFS)
    qmd.compile()
    qmd.fit(X_train, epochs=1, batch_size = 1) ### must keep the batch size = 1
    #qmd.fit(X_train[600:601], epochs=1, batch_size = 1) ### must keep the batch size = 1, uncomment for single point prediction
    predictions_classical_qeff = tf.cast(tf.math.pow((GAMMA/(tf.constant(math.pi))), DIM_X/2)*qmd.predict(X_plot, batch_size = 1), tf.float32).numpy()
    predictions_classical_qeff_train = tf.cast(tf.math.pow((GAMMA/(tf.constant(math.pi))), DIM_X/2)*qmd.predict(X_train, batch_size = 1), tf.float32).numpy()
    predictions_classical_qeff_test = tf.cast(tf.math.pow((GAMMA/(tf.constant(math.pi))), DIM_X/2)*qmd.predict(X_test, batch_size = 1), tf.float32).numpy()

    return predictions_classical_qeff_train, predictions_classical_qeff_test, predictions_classical_qeff


def _dmkde_classical_qrff_exec(N_FFS, GAMMA, RANDOM_STATE_QRFF, DIM_X, X_train, X_test, X_plot, type_ffs = "qrff"):
    if type_ffs == "rff":
        fm_x = qmc_layers.QFeatureMapRFF(DIM_X, dim=N_FFS, gamma=GAMMA/2, random_state=RANDOM_STATE_QRFF)
        qmd = qmc_models.QMDensity(fm_x, N_FFS)
    elif type_ffs == "qrff":
        fm_x = qmc_layers.QFeatureMapComplexRFF(DIM_X, dim=N_FFS, gamma=GAMMA/2, random_state= RANDOM_STATE_QRFF)
        qmd = qmc_models.ComplexQMDensity(fm_x, N_FFS)
    qmd.compile()
    qmd.fit(X_train, epochs=1)
    #qmd.fit(X_train[600:601], epochs=1) # uncomment for single point prediction

    predictions_classical = tf.cast(tf.math.pow((GAMMA/(tf.constant(math.pi))), DIM_X/2)*qmd.predict(X_plot), tf.float32).numpy()
    predictions_classical_train = tf.cast(tf.math.pow((GAMMA/(tf.constant(math.pi))), DIM_X/2)*qmd.predict(X_train), tf.float32).numpy()
    predictions_classical_test = tf.cast(tf.math.pow((GAMMA/(tf.constant(math.pi))), DIM_X/2)*qmd.predict(X_test), tf.float32).numpy()

    return predictions_classical_train, predictions_classical_test, predictions_classical


def _vqkde_qeff_exec(X_train, X_test, X_plot, GAMMA, RANDOM_STATE_QEFF, EPOCHS, DIM_X, N_TRAINING_DATA):
    raw_kde_probability_train = np.array(
                [_raw_kde(x_temp[np.newaxis,:], X_train, GAMMA) for x_temp in X_train])
            
    LEARNING_RATE = 0.0005 ### Hyperparameter original 0.0005
    y_expected =  raw_kde_probability_train*(np.pi/GAMMA)
    vc = VQKDE_QEFF(dim_x_param = DIM_X, n_qeff_qubits = NUM_QUBITS_FFS, n_ancilla_qubits =  NUM_ANCILLA_QUBITS, gamma=GAMMA, learning_rate = LEARNING_RATE, random_state = RANDOM_STATE_QEFF, n_training_data = N_TRAINING_DATA)
    vc.fit(X_train, y_expected, batch_size=16, epochs = EPOCHS)

    predictions_train = vc.predict(X_train)
    predictions_test = vc.predict(X_test)
    predictions_plot = vc.predict(X_plot)

    return predictions_train, predictions_test, predictions_plot


def _vqkde_qrff_exec(
        X_train, X_test, X_plot, GAMMA, DIM_X, RANDOM_STATE_QRFF, N_FFS, LEARNING_RATE, N_TRAINING_DATA, EPOCHS):
    r = np.random.RandomState(RANDOM_STATE_QRFF)
    weights_ffs_temp = r.normal(0, 1, (DIM_X, N_FFS))

    X_feat_train = _predict_features(X_train, weights_ffs_temp, GAMMA)
    X_feat_test = _predict_features(X_test, weights_ffs_temp, GAMMA)
    X_feat_plot = _predict_features(X_plot, weights_ffs_temp, GAMMA)
    y_expected = np.array([_raw_kde(x_temp[np.newaxis,:], X_train, GAMMA) for x_temp in X_train])*(np.pi/GAMMA)

    U_train_conjTrans = np.array([np.conjugate(_create_U_train(X_feat_train[i]).T) for i in range(len(X_feat_train))])
    U_test_conjTrans = np.array([np.conjugate(_create_U_train(X_feat_test[i]).T) for i in range(len(X_feat_test))])
    U_plot_conjTrans = np.array([np.conjugate(_create_U_train(X_feat_plot[i]).T) for i in range(len(X_feat_plot))])

    vc = VQKDE_QRFF(dim_x_param = DIM_X, n_qrff_qubits = NUM_QUBITS_FFS, n_ancilla_qubits =  NUM_ANCILLA_QUBITS, learning_rate = LEARNING_RATE, n_training_data = N_TRAINING_DATA, gamma=GAMMA) # best 8 epochs

    vc.fit(U_train_conjTrans, y_expected, batch_size=16, epochs = EPOCHS)

    predictions_train = vc.predict(U_train_conjTrans)
    predictions_test = vc.predict(U_test_conjTrans)
    predictions_plot = vc.predict(U_plot_conjTrans)

    return predictions_train, predictions_test, predictions_plot


def _vqkde_qeff_hea_exec(
        X_train, X_test, X_plot, GAMMA, DIM_X, N_TRAINING_DATA, LEARNING_RATE, RANDOM_STATE_QEFF, NUM_LAYERS_HEA, EPOCHS):
    y_expected = np.array([_raw_kde(x_temp[np.newaxis,:], X_train, GAMMA) for x_temp in X_train])*(np.pi/GAMMA)

    vc = VQKDE_QEFF_HEA(dim_x_param = DIM_X, n_qeff_qubits = NUM_QUBITS_FFS, n_ancilla_qubits =  NUM_ANCILLA_QUBITS, gamma=GAMMA, num_layers_hea = NUM_LAYERS_HEA, learning_rate = LEARNING_RATE, random_state = RANDOM_STATE_QEFF, n_training_data = N_TRAINING_DATA)

    vc.fit(X_train, y_expected, batch_size=16, epochs = EPOCHS)

    predictions_train = vc.predict(X_train)
    predictions_test = vc.predict(X_test)
    predictions_plot = vc.predict(X_plot)

    return predictions_train, predictions_test, predictions_plot

    
def _vqkde_qrff_hea_exec(
        X_train, X_test, X_plot, GAMMA, DIM_X, RANDOM_STATE_QRFF, N_FFS, LEARNING_RATE, N_TRAINING_DATA, EPOCHS, NUM_LAYERS_HEA):
    r = np.random.RandomState(RANDOM_STATE_QRFF)
    weights_ffs_temp = r.normal(0, 1, (DIM_X, N_FFS))

    X_feat_train = _predict_features(X_train, weights_ffs_temp, GAMMA)
    X_feat_test = _predict_features(X_test, weights_ffs_temp, GAMMA)
    X_feat_plot = _predict_features(X_plot, weights_ffs_temp, GAMMA)
    y_expected =  np.array([_raw_kde(x_temp[np.newaxis,:], X_train, GAMMA) for x_temp in X_train])*(np.pi/GAMMA)

    U_train_conjTrans = np.array([np.conjugate(_create_U_train(X_feat_train[i]).T) for i in range(len(X_feat_train))])
    U_test_conjTrans = np.array([np.conjugate(_create_U_train(X_feat_test[i]).T) for i in range(len(X_feat_test))])
    U_plot_conjTrans = np.array([np.conjugate(_create_U_train(X_feat_plot[i]).T) for i in range(len(X_feat_plot))])

    vc = VQKDE_QRFF_HEA(dim_x_param = DIM_X, n_qrff_qubits = NUM_QUBITS_FFS, n_ancilla_qubits =  NUM_ANCILLA_QUBITS, gamma=GAMMA, num_layers_hea = NUM_LAYERS_HEA, learning_rate = LEARNING_RATE, n_training_data = N_TRAINING_DATA)
    vc.fit(U_train_conjTrans, y_expected, batch_size=16, epochs = EPOCHS)

    predictions_train = vc.predict(U_train_conjTrans)
    predictions_test = vc.predict(U_test_conjTrans)
    predictions_plot = vc.predict(U_plot_conjTrans)

    return predictions_train, predictions_test, predictions_plot


def _get_required_dict(fn, kwargs):
    fn_args = set(getfullargspec(fn).args)
    fn_required_args = fn_args - set(getfullargspec(run).args)
    return {item: kwargs[item] for item in fn_required_args if item in kwargs}


def _run(model, **kwargs):
    exec_functions = {
        "raw_kde": _raw_kde_exec,
        "dmkde_qeff": _dmkde_classical_qeff_exec,
        "dmkde_qrff": _dmkde_classical_qrff_exec,
        "vqkde_qeff": _vqkde_qeff_exec,
        "vqkde_qrff": _vqkde_qrff_exec,
        "vqkde_qeff_hea": _vqkde_qeff_hea_exec,
        "vqkde_qrff_hea": _vqkde_qrff_hea_exec,
    }

    if model in exec_functions:
        fn = exec_functions[model]
        required_dict = _get_required_dict(fn, kwargs)
        predictions_train, predictions_test, predictions_plot = fn(**required_dict)
        return predictions_train, predictions_test, predictions_plot

    return predictions_train, predictions_test, predictions_plot
    

def main():
    parser = argparse.ArgumentParser(description="Test models on datasets.")
    parser.add_argument('--o', type=str, default='results', help='Output directory for saving results')
    parser.add_argument('--model', type=str, nargs='+', choices=MODELS, help='Model to run')
    parser.add_argument('--dataset', type=str, nargs='+', choices=DATASETS, help='Dataset to use')
    parser.add_argument('--alldata', action='store_true', help='Run on all datasets')
    parser.add_argument('--allmodels', action='store_true', help='Run on all models')
    parser.add_argument('--all', action='store_true', help='Run on all models and data')
    
    args = parser.parse_args()

    selected_models = args.model if (args.model and args.allmodels!=None and args.all!=None) else MODELS
    selected_datasets = args.dataset if (args.dataset and args.alldata!=None and args.all!=None) else DATASETS

    for dataset_name in selected_datasets:
        X_train, X_train_densities, X_test, X_test_densities = load_data(dataset=dataset_name) 

        grid = GRID_PARAMS_DICT[dataset_name]

        x, y = np.mgrid[grid["x_range"][0]:grid["x_range"][1]:grid["x_step"], grid["y_range"][0]:grid["y_range"][1]:grid["y_step"]]
        pos = np.dstack((x, y))
        X_plot = pos.reshape([14400,2])

        training_dict = {
            "x": x,
            "y": y,
            "X_train": X_train, 
            "X_train_densities": X_train_densities, 
            "X_test": X_test, 
            "X_test_densities": X_test_densities,
            "X_plot": X_plot,
            "GAMMA": GAMMA_DICT[dataset_name],
            "RANDOM_STATE_QRFF": RANDOM_STATE_QRFF_DICT[dataset_name],
            "RANDOM_STATE_QEFF": RANDOM_STATE_QEFF_DICT[dataset_name],
            "LEARNING_RATE": LEARNING_RATE_DICT[dataset_name],
            "EPOCHS": EPOCHS_DICT[dataset_name],
            "DIM_X": X_train.shape[1],
            "N_TRAINING_DATA": X_train.shape[0],
            "N_FFS": 2**NUM_QUBITS_FFS,
            "NUM_LAYERS_HEA": 3,
        }

        model_output_dir = os.path.join(args.o, dataset_name)
        os.makedirs(model_output_dir, exist_ok=True)

        for model_name in selected_models:
            print("#####################################################")
            print(f"Running model {model_name} in dataset {dataset_name}", end="\n")
            
            predictions_train, predictions_test, predictions_plot = _run(model_name, **training_dict)
            
            plt.rcParams.update(params)
            plt.title(f"{model_name} - {dataset_name}")
            plt.contourf(x, y, predictions_plot.reshape([120,120]))
            plt.colorbar()
            plt.savefig(os.path.join(model_output_dir, f"{model_name}_{dataset_name}.pdf")) 

            print("#####################################################", end="\n")


if __name__ == "__main__":
    main()