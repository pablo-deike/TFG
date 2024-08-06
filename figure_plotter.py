import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def plot_data(x,y,data, title, funcion, max_value_painted):

    # plot results
    fontsize=12

    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(data[:,0:4], interpolation='nearest', vmin=0, vmax=max_value_painted, cmap="cool")
    tick_values = np.linspace(0, max_value_painted, 6)
    tick_values_parsed = ["%.2f" % values_tostr for values_tostr in tick_values]
    cbar=fig.colorbar(cax)
    cbar.ax.set_ylabel(funcion ,rotation=90,fontsize=fontsize)
    cbar.set_ticks(tick_values)
    cbar.set_ticklabels(tick_values_parsed)

    # put text on matrix elements
    for i, x_val in enumerate(np.arange(len(x))):
        for j, y_val in enumerate(np.arange(len(y))):
            if data[j,i]>1000:
                c = "{0:.1e}".format(data[j,i]) 
            else:
                c = "{0:.2f}".format(data[j,i])  
            ax.text(x_val, y_val, c, va='center', ha='center')

    # convert axis values to to string labels
    x=[str(i) for i in x]
    y=[str(i) for i in y*100]


    ax.set_xticklabels(['']+x)
    ax.set_yticklabels([""]+y)
    
    ax.xaxis.set_ticks_position('bottom')
    ax.xaxis.set_label_position('bottom')

    ax.set_xlabel('Tasa de aprendizaje',fontsize=fontsize)
    ax.set_ylabel("Tamaño de datos de entrenamiento (%)",fontsize=fontsize)
    ax.set_title(title)

    plt.tight_layout()

    plt.show()


def plot_data_epochs(x ,y ,data , title: str):

    # plot results
    fontsize=12

    max_value_painted = data.max()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(data[:,0:4], interpolation='nearest', vmin=0, vmax=max_value_painted, cmap="cool")
    tick_values = np.linspace(0, max_value_painted, 6)
    tick_values_int = [int(values_to_int) for values_to_int in tick_values]
    tick_values_parsed = [str(values_to_str) for values_to_str in tick_values_int]
    cbar=fig.colorbar(cax)
    cbar.ax.set_ylabel('número iteraciones',rotation=90,fontsize=fontsize)
    cbar.set_ticks(tick_values_int)
    cbar.set_ticklabels(tick_values_parsed)

    # put text on matrix elements
    for i, x_val in enumerate(np.arange(len(x))):
        for j, y_val in enumerate(np.arange(len(y))):
            c = "{0:.0f}".format( data[j,i])  
            ax.text(x_val, y_val, c, va='center', ha='center')

    # convert axis vaues to to string labels
    x=[str(i) for i in x]
    y=[str(i) for i in y*100]


    ax.set_xticklabels(['']+x)
    ax.set_yticklabels([""]+y)
        
    ax.xaxis.set_ticks_position('bottom')
    ax.xaxis.set_label_position('bottom')

    ax.set_xlabel('Tasa de aprendizaje',fontsize=fontsize)
    ax.set_ylabel("Tamaño de datos de entrenamiento (%)",fontsize=fontsize)
    ax.set_title(title)

    plt.tight_layout()

    plt.show()
    
    
# data_test=pd.read_csv('C:/Users/pablo/Desktop/scripts tfg/entrenamientos adam tres capas/Adam_test_loss.csv', index_col='Unnamed: 0')
# data_loss=pd.read_csv('C:/Users/pablo/Desktop/scripts tfg/entrenamientos adam tres capas/Adam_train_loss.csv',index_col='Unnamed: 0')
# data_nepochs=pd.read_csv('C:/Users/pablo/Desktop/scripts tfg/entrenamientos adam tres capas/Adam_nepochs.csv', index_col='Unnamed: 0')
# data_test=pd.read_csv('C:/Users/pablo/Desktop/scripts tfg/Adam_batches_test_loss.csv', index_col='Unnamed: 0')
# data_loss=pd.read_csv('C:/Users/pablo/Desktop/scripts tfg/Adam_batches_train_loss.csv',index_col='Unnamed: 0')
# data_nepochs=pd.read_csv('C:/Users/pablo/Desktop/scripts tfg/Adam_batches_nepochs.csv', index_col='Unnamed: 0')
data_test=pd.read_csv('C:/Users/pablo/Desktop/scripts tfg/SGD_batches128_test_loss.csv', index_col='Unnamed: 0')
data_loss=pd.read_csv('C:/Users/pablo/Desktop/scripts tfg/SGD_batches128_train_loss.csv',index_col='Unnamed: 0')
data_nepochs=pd.read_csv('C:/Users/pablo/Desktop/scripts tfg/SGD_batches128_nepochs.csv', index_col='Unnamed: 0')
test_loss=data_test.values
train_loss=data_loss.values
test_error = np.sqrt(test_loss)
train_error = np.sqrt(train_loss)
nepochs=data_nepochs.values

learning_rates=np.logspace(-5,-2,4)
data_sizes=np.array([0.3,0.4,0.5,0.6,0.7,0.8])

plot_data(learning_rates, data_sizes, test_error, "Error de los datos de testeo", "Error (MeV)", 3)
plot_data(learning_rates, data_sizes, train_error, "Error de los datos de entrenamiento", "Error (MeV)",3)
plot_data_epochs(learning_rates, data_sizes, nepochs, "Número de iteraciones hasta llegar al mínimo")