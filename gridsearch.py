import numpy as np
import pandas as pd
from training_model import training_model


def grid_search(x, y):
    learning_rates = np.logspace(-5, -1, 5)
    data_sizes = np.array([0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
    test_loss = np.zeros((len(data_sizes), len(learning_rates)))
    train_loss = np.zeros_like(test_loss)
    nepochs = np.zeros_like(test_loss)
    has_finished_mat = np.zeros_like(test_loss)
    for a_iter, data_size in enumerate(data_sizes):
        for b_iter, learning_rate in enumerate(learning_rates):
            (
                total_loss_slice,
                loss_slice,
                nepoch,
                hatY_slice,
                hatY_pred_slice,
                x_train,
                x_test_slice,
                y_train,
                y_test_slice,
                has_finished,
            ) = training_model(
                x, y, data_size=data_size, learning_rate=learning_rate, batch_size=128
            )
            test_loss[a_iter, b_iter] = total_loss_slice.item()
            train_loss[a_iter, b_iter] = loss_slice.item()
            nepochs[a_iter, b_iter] = nepoch
            has_finished_mat[a_iter, b_iter] = has_finished

    df = pd.DataFrame(
        test_loss,
        columns=["1e-05", "0.0001", "0.001", "0.01", "0.1"],  # Example learning rates
        index=[0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
    )  # Example data sizes
    df.to_csv("Adam_test_loss.csv")

    df = pd.DataFrame(
        train_loss,
        columns=["1e-05", "0.0001", "0.001", "0.01", "0.1"],  # Example learning rates
        index=[0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
    )  # Example data sizes
    df.to_csv("Adam_train_loss.csv")

    df = pd.DataFrame(
        nepochs,
        columns=["1e-05", "0.0001", "0.001", "0.01", "0.1"],  # Example learning rates
        index=[0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
    )  # Example data sizes
    df.to_csv("Adam_nepochs.csv")

    df = pd.DataFrame(
        has_finished_mat,
        columns=["1e-05", "0.0001", "0.001", "0.01", "0.1"],  # Example learning rates
        index=[0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
    )  # Example data sizes
    df.to_csv("Adam_has_finished.csv")
