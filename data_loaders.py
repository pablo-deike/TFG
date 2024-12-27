from torch.utils.data import TensorDataset, DataLoader
import torch


def create_dataloaders(x, y, split_ratio):
    dataset = TensorDataset(x, y)
    train_size = int(len(dataset) * split_ratio)
    test_size = len(dataset) - train_size

    train_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, test_size]
    )
    train_loader = DataLoader(
        train_dataset, batch_size=train_size, shuffle=True, drop_last=True
    )
    test_loader = DataLoader(test_dataset, batch_size=test_size, shuffle=False)

    return train_loader, test_loader, train_size
