
import argparse
import os
from inspect import getfullargspec
import matplotlib.pyplot as plt
import numpy as np

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

GAMMA_DICT = {"binomial": 2., "potential_1": 4., "potential_2": 16., "arc": 4., "star": 16.}
RANDOM_STATE_QRFF_DICT = {"binomial": 324, "potential_1": 125, "potential_2": 178, "arc": 7, "star": 1224}
RANDOM_STATE_QEFF_DICT = {"binomial": 3, "potential_1": 15, "potential_2": 78, "arc": 73, "star": 24}
EPOCHS_DICT  = {"binomial": 0, "potential_1": 8, "potential_2": 0, "arc": 60, "star": 60}
LEARNING_RATE_DICT = {"binomial": 0.0005, "potential_1": 0.0005, "potential_2": 0.005, "arc": 0.0005, "star": 0.0005}


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
    y_expected =  np.array([_raw_kde(x_temp[np.newaxis,:], X_train, GAMMA) for x_temp in X_train])*(np.pi/GAMMA)

    U_train_conjTrans = np.array([np.conjugate(_create_U_train(X_feat_train[i]).T) for i in range(len(X_feat_train))])
    U_test_conjTrans = np.array([np.conjugate(_create_U_train(X_feat_test[i]).T) for i in range(len(X_feat_test))])
    U_plot_conjTrans = np.array([np.conjugate(_create_U_train(X_feat_plot[i]).T) for i in range(len(X_feat_plot))])

    vc = VQKDE_QRFF(dim_x_param = DIM_X, n_qrff_qubits = NUM_QUBITS_FFS, n_ancilla_qubits =  NUM_ANCILLA_QUBITS, learning_rate = LEARNING_RATE, n_training_data = N_TRAINING_DATA, gamma=GAMMA) # best 8 epochs

    vc.fit(U_train_conjTrans, y_expected, batch_size=16, epochs = EPOCHS)

    predictions_train = vc.predict(U_train_conjTrans)
    predictions_test = vc.predict(U_test_conjTrans)
    predictions_plot = vc.predict(U_plot_conjTrans)

    return predictions_train, predictions_test, predictions_plot

def run(model, **kwargs):

    match model:
        case "raw_kde":
            pass
        case "dmkde_qeff":
            pass
        case "dmkde_qrff":
            pass
        case "vqkde_qeff":
            fn_args = set(getfullargspec(_vqkde_qeff_exec).args)
            fn_required_args = fn_args - set(getfullargspec(run).args)

            required_dict = {
                item: kwargs[item] for item in fn_required_args if item in kwargs
            }

            predictions_train, predictions_test, predictions_plot = _vqkde_qeff_exec(**required_dict)
            return predictions_train, predictions_test, predictions_plot
        
        case "vqkde_qrff":
            pass
        case "vqkde_qeff_hea":
            pass
        case "vqkde_qrff_hea":
            pass
    


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

        x, y = np.mgrid[-7:7:(14/120), -7:7:(14/120)]
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
        }

        for model_name in selected_models:
            model_output_dir = os.path.join(args.o, model_name)
            os.makedirs(model_output_dir, exist_ok=True)

            predictions_train, predictions_test, predictions_plot = run(model_name, **training_dict)
            
            plt.rcParams.update(params)
            plt.contourf(x, y, predictions_plot.reshape([120,120]))
            plt.colorbar()
            plt.savefig(os.path.join(model_output_dir, f"{model_name}_{dataset_name}.pdf")) 


if __name__ == "__main__":
    main()