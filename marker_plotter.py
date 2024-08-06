import array
from ast import Tuple
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.colors import LinearSegmentedColormap
import pandas as pd
import numpy as np
from pyparsing import col


def marker_plot(x_train,y_train,data_train,x_test,y_test,data_test):
    plt.figure(figsize=(12,9))
    x_train=np.array(x_train)
    x_test=np.array(x_test)
    y_train=np.array(y_train)
    y_test=np.array(y_test)
    data_train=np.array(data_train)
    data_test=np.array(data_test)
    dif_test = np.abs(data_test-y_test)
    dif_train = np.abs(data_train-y_train)
    dif=np.array((dif_test, dif_train))
    # First subplot for the even indexed data
    sc1 = plt.scatter(x_test[:, 1], x_test[:, 0], c=data_test-y_test, marker='D', label = "Datos de test", vmax=dif.max(), vmin=-dif.max(), cmap="RdBu", s=20)
    sc2 = plt.scatter(x_train[:, 1], x_train[:, 0], c=data_train-y_train, marker='*', label = "Datos de entrenamiento", vmax=dif.max(), vmin=-dif.max(), cmap= "RdBu", s=20)
    plt.xlabel('N', fontsize=16)
    plt.xticks([8,20,28,50,82,126, 198], fontsize=16)
    plt.yticks([8,20,28,50,82,110], fontsize=16)
    for tick in [8, 20, 28, 50, 82]:
        plt.axvline(x=tick, color='gray', linestyle='--', linewidth=0.5)
        plt.axhline(y=tick, color='gray', linestyle='--', linewidth=0.5)
    plt.axvline(x=126, color="gray", linestyle= "--", linewidth = 0.5)
    plt.axvline(x=198, color="gray", linestyle= "--", linewidth = 0.5)
    plt.axhline(y=110, color='gray', linestyle='--', linewidth=0.5)
    plt.ylabel('Z', fontsize=16)
    plt.title('Datos de test y entrenamiento')
    legend_elements = [plt.Line2D([0], [0], marker='D', color='w', markerfacecolor='black', label='Datos de test', markersize=8),
                       plt.Line2D([0], [0], marker='*', color='w', markerfacecolor='black', label='Datos de entrenamiento',  markersize=15)]
    plt.legend(handles=legend_elements)
    cbar1 = plt.colorbar(sc1,ax=plt.gca(), label="Diferencia entre predicción y datos reales (MeV)")
    cbar1.set_label(fontsize=16)
    #cbar2 = plt.colorbar(sc2, ax=plt.gca(), label= "Diferencia (Datos entrenamiento)")
    plt.show()

def marker_plot_subplot(x_train,y_train,data_train,x_test,y_test,data_test, x_train2,y_train2,data_train2,x_test2,y_test2,data_test2):
    dif_values_test = data_test - y_test
    dif_values_train = data_train - y_train
    dif_values_test2 = data_test2 - y_test2
    dif_values_train2 = data_train2 - y_train2

    plt.figure(figsize=(18, 9)) 

    error_train1, error_test1 = calculate_relative_error(y_test=y_test, data_test=data_test, y_train=y_train, data_train=data_train)
    error_train2, error_test2 = calculate_relative_error(y_test=y_test2, data_test=data_test2, y_train=y_train2, data_train=data_train2)
    # dif_test_max = np.abs(dif_values_test).max()
    # dif_train_max = np.abs(dif_values_train).max()
    # dif_test2_max = np.abs(dif_values_test2).max()
    # dif_train2_max = np.abs(dif_values_train2).max()
    # dif = np.array((dif_test_max, dif_test2_max, dif_train2_max, dif_train_max))

    error_test_max = np.abs(error_test1).max()
    error_train_max = np.abs(error_train1).max()
    error_test2_max = np.abs(error_test2).max()
    error_train2_max = np.abs(error_train2).max()
    errors = np.array((error_test_max, error_train_max, error_test2_max, error_train2_max))

    # vmin = -dif.max()+3
    # vmax = dif.max()-3

    vmin = 0
    vmax = 100
    
    # First subplot
    ax1 = plt.subplot(1, 2, 1)
    # sc1 = ax1.scatter(x_test[:, 1], x_test[:, 0], c=error_test1, marker='D', label="", vmin=vmin, vmax=vmax, cmap="RdBu", s=18)
    # sc2 = ax1.scatter(x_train[:, 1], x_train[:, 0], c=error_train1, marker='*', label="", vmin=vmin, vmax=vmax, cmap="RdBu", s=18)
    
    sc1 = ax1.scatter(x_test[:, 1], x_test[:, 0], c=error_test1, marker='D', label="", vmin=vmin, vmax=vmax, s=18)
    sc2 = ax1.scatter(x_train[:, 1], x_train[:, 0], c=error_train1, marker='*', label="", vmin=vmin, vmax=vmax, s=18)
    
    ax1.set_xlabel('N', fontsize=14)
    ax1.set_xticks([8, 20, 28, 50, 82, 126, 198], fontsize=16)
    ax1.set_yticks([8, 20, 28, 50, 82, 110], fontsize=16)
    for tick in [8, 20, 28, 50, 82]:
        ax1.axvline(x=tick, color='gray', linestyle='--', linewidth=0.5)
        ax1.axhline(y=tick, color='gray', linestyle='--', linewidth=0.5)
    ax1.axvline(x=126, color="gray", linestyle="--", linewidth=0.5)
    ax1.axvline(x=198, color="gray", linestyle="--", linewidth=0.5)
    ax1.axhline(y=110, color='gray', linestyle='--', linewidth=0.5)
    ax1.set_ylabel('Z', fontsize=14)
    ax1.tick_params(labelsize=14)
    ax1.set_title("Datos de test y entrenamiento $80\%$ de los datos y $\\eta=0.0001$ con optimizador Adam", fontsize=12)
    # ax1.set_title("Datos de test y entrenamiento $80\%$ de los datos y $\\eta=0.0001$", fontsize=16)
    legend_elements = [
        plt.Line2D([0], [0], marker='D', color='w', markerfacecolor='black', label='Datos de test', markersize=8),
        plt.Line2D([0], [0], marker='*', color='w', markerfacecolor='black', label='Datos de entrenamiento', markersize=15)
    ]
    ax1.legend(handles=legend_elements)

    # Second subplot
    ax2 = plt.subplot(1, 2, 2)
    # sc1 = ax2.scatter(x_test2[:, 1], x_test2[:, 0], c=error_test2, marker='D', label="", vmin=vmin, vmax=vmax, cmap="RdBu", s=18)
    # sc2 = ax2.scatter(x_train2[:, 1], x_train2[:, 0], c=error_train2, marker='*', label="", vmin=vmin, vmax=vmax, cmap="RdBu", s=18)

    sc1 = ax2.scatter(x_test[:, 1], x_test[:, 0], c=error_test1, marker='D', label="", vmin=vmin, vmax=vmax, s=18)
    sc2 = ax2.scatter(x_train[:, 1], x_train[:, 0], c=error_train1, marker='*', label="", vmin=vmin, vmax=vmax, s=18)
    ax2.set_xlabel('N', fontsize=14)
    ax2.set_xticks([8, 20, 28, 50, 82, 126, 198], fontsize=16)
    ax2.set_yticks([8, 20, 28, 50, 82, 110], fontsize=16)
    ax2.tick_params(labelsize=14)
    for tick in [8, 20, 28, 50, 82]:
        ax2.axvline(x=tick, color='gray', linestyle='--', linewidth=0.5)
        ax2.axhline(y=tick, color='gray', linestyle='--', linewidth=0.5)
    ax2.axvline(x=126, color="gray", linestyle="--", linewidth=0.5)
    ax2.axvline(x=198, color="gray", linestyle="--", linewidth=0.5)
    ax2.axhline(y=110, color='gray', linestyle='--', linewidth=0.5)
    ax2.set_ylabel('Z', fontsize=14)
    ax2.set_title("Datos de test y entrenamiento $50\%$ de los datos y $\\eta=0.001$ con optimizador SGD", fontsize=12)
    # ax2.set_title("Datos de test y entrenamiento $50\%$ de los datos y $\\eta=0.001$", fontsize=16)
    legend_elements = [
        plt.Line2D([0], [0], marker='D', color='w', markerfacecolor='black', label='Datos de test', markersize=8),
        plt.Line2D([0], [0], marker='*', color='w', markerfacecolor='black', label='Datos de entrenamiento', markersize=15)
    ]
    ax2.legend(handles=legend_elements)

    plt.tight_layout()
    # Common colorbar
    # cbar = plt.colorbar(sc1, ax=[ax1, ax2], label="Diferencia entre la predicción $difE$ y su valor real (MeV)", orientation='horizontal', pad=0.1)
    # cbar.set_label("Diferencia entre la predicción $difE$ y su valor real (MeV)", fontsize=16)
    cbar = plt.colorbar(sc1, ax=[ax1, ax2], label="Error relativo (%)", orientation='horizontal', pad=0.1)
    cbar.set_label("Error relativo (%)", fontsize=16)
    plt.show()

def calculate_relative_error(y_train, data_train, y_test, data_test):
    dif_values_test = np.abs(data_test - y_test)
    dif_values_train = np.abs(data_train - y_train)
    error_train = dif_values_train/y_train*100
    error_test = dif_values_test/y_test*100
    return error_train, error_test


data_train = pd.read_csv("difE_adam_train_with_1e_4_80_no_drop.csv", sep=",")
data_test = pd.read_csv("difE_adam_test_with_1e_4_80_no_drop.csv", sep=",")
# data_train2 = pd.read_csv("difE_adam_train_with_1e_2_40_no_drop.csv", sep=",")
# data_test2 = pd.read_csv("difE_adam_test_with_1e_2_40_no_drop.csv", sep=",")
# data_train = pd.read_csv("difE_sgd_train_with_1e_4_80_no_drop.csv", sep=",")
# data_test = pd.read_csv("difE_sgd_test_with_1e_4_80_no_drop.csv", sep=",")
data_train2 = pd.read_csv("difE_sgd_train_with_1e_3_50_no_drop.csv", sep=",")
data_test2 = pd.read_csv("difE_sgd_test_with_1e_3_50_no_drop.csv", sep=",")


list_zn_train = data_train.iloc[:,0:2].values
difE_real_train = data_train.iloc[:,2].values
difE_pred_train = data_train.iloc[:,3].values

list_zn_test = data_test.iloc[:,0:2].values
difE_real_test = data_test.iloc[:,2].values
difE_pred_test = data_test.iloc[:,3].values

list_zn_train2 = data_train2.iloc[:,0:2].values
difE_real_train2= data_train2.iloc[:,2].values
difE_pred_train2 = data_train2.iloc[:,3].values

list_zn_test2 = data_test2.iloc[:,0:2].values
difE_real_test2 = data_test2.iloc[:,2].values
difE_pred_test2 = data_test2.iloc[:,3].values
# marker_plot(x_train=list_zn_train, y_train=difE_real_train, data_train=difE_pred_train,
#             x_test= list_zn_test, y_test=difE_real_test, data_test=difE_pred_test)


marker_plot_subplot(x_train=list_zn_train, y_train=difE_real_train, data_train=difE_pred_train,
            x_test= list_zn_test, y_test=difE_real_test, data_test=difE_pred_test,
            x_train2=list_zn_train2, y_train2=difE_real_train2, data_train2=difE_pred_train2,
            x_test2= list_zn_test2, y_test2=difE_real_test2, data_test2=difE_pred_test2)