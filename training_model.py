import torch
import numpy as np
import pandas as pd
from neural_networks.four_layer_nn import FourLayerNN
from data_loaders import create_dataloaders


def training_model_SGD(
    x: list,
    y: list,
    data_size: float,
    learning_rate: float,
    batch_size: int,
):
    train_loader, test_loader, train_size = create_dataloaders(
        x, y, split_ratio=data_size, batch_size=batch_size
    )
    net_slice = FourLayerNN(2, 1)
    net_slice = net_slice.float()
    criterion = torch.nn.MSELoss(reduction="mean")
    optimizer = torch.optim.SGD(net_slice.parameters(), lr=learning_rate, momentum=0.9)
    dloss = 1
    nepoch = 0
    net_slice.train()
    hatY = torch.zeros(train_size, 1)
    n_batches = int(np.ceil(train_size / batch_size))
    new_loss = np.zeros(n_batches)
    previous_loss = np.zeros(n_batches)
    print(
        f"training with {n_batches} batches in {train_size} points of data with learning_rate={learning_rate}"
    )

    while dloss > 1e-3 and nepoch < 5000:
        for batch_idx, (x_train_slice, y_train_slice) in enumerate(train_loader):
            x_train_slice = x_train_slice.float()
            y_train_slice = y_train_slice.float()
            hatY_slice = net_slice.forward(x_train_slice)
            optimizer.zero_grad()
            loss_slice = criterion(hatY_slice, y_train_slice)
            loss_slice.backward()
            optimizer.step()
            # x_train.append(x_train_slice)
            # y_train.append(y_train_slice)
            if nepoch % 50 == 0:
                new_loss[batch_idx] = loss_slice.item()
                print(
                    f"The loss function for {nepoch} is {loss_slice.item()} and batch numb={batch_idx}"
                )
                if batch_idx == n_batches - 1:
                    dloss = np.abs(np.mean(new_loss - previous_loss))
                    print(f"current diff{dloss}")
                    previous_loss = new_loss
                    new_loss = np.zeros(n_batches)
        nepoch += 1
    all_x_train = torch.Tensor()
    all_y_train = torch.Tensor()
    for x_train, y_train in train_loader:
        all_x_train = torch.cat((all_x_train, x_train), 0)
        all_y_train = torch.cat((all_y_train, y_train), 0)
    hatY = net_slice.forward(all_x_train)
    # elapsed_time = time.time() - start_time
    with torch.no_grad():
        net_slice.eval()
        for x_test, y_test in test_loader:
            x_test = x_test.float()
            y_test = y_test.float()
            hatY_pred = net_slice.forward(x_test)
            hatY_detach = hatY_pred.detach()
            test_loss = criterion(hatY_detach, y_test)
    return (
        test_loss,
        loss_slice,
        nepoch,
        hatY,
        hatY_pred,
        all_x_train,
        x_test,
        all_y_train,
        y_test,
    )


# df = pd.read_csv("C:/Users/pablo/Desktop/scripts tfg/data_E.csv")
# list_zn = df[["# Z", "N"]].values.tolist()
# difE = df["difE"].values.tolist()
# x = torch.tensor(list_zn)
# y = torch.tensor(np.abs(difE))

# # test_loss, loss, nepoch, hatY, hatY_pred, all_x_train, x_test, all_y_train, y_test= training_model(x,y,data_size=0.8, learning_rate=1e-4,batch_size=128)

# test_loss, loss, nepoch, hatY, hatY_pred, all_x_train, x_test, all_y_train, y_test = (
#     training_model(
#         x, y, data_size=0.8, learning_rate=0.0001, batch_size=128, nepoch=751
#     )
# )
# print("done")
# print(np.sqrt(test_loss.item()), np.sqrt(loss.item()))
# np.savetxt(
#     "difE_sgd_train_with_1e_4_80_no_drop.csv",
#     torch.column_stack((all_x_train, all_y_train, hatY)).detach().numpy(),
#     delimiter=",",
#     header="Z,N,difE real, difE pred",
#     comments="",
# )
# np.savetxt(
#     "difE_sgd_test_with_1e_4_80_no_drop.csv",
#     torch.column_stack((x_test, y_test, hatY_pred)).detach().numpy(),
#     delimiter=",",
#     header="Z,N,difE real, difE pred",
#     comments="",
# )
