# Repo of neural networks in nuclear physics

## How it works
This repo will have neural network classes. This neural network classes should have training and evaluation. This will make them easy and reusable

It will also have one parent class of the gridsearch. To apply grid search to any type of dataset, we should inherit from this parent class. Again to make everything easier 

For code editing we will use pep-8 and we can use any linting tool such as black: [Black documentation](https://black.readthedocs.io/en/stable/index.html "Black documentation"). This is recommended to avoid merge conflicts and better readability

## Activating conda environment

To use the conda environment in Linux, after installing miniconda and adding it to the path run the following command on the repository folder
```
conda env create -f environment.yml
```
This will create the environment with the necessary dependencies. Then run the following:
```
source ~/.bashrc
conda activate myenv
```
## How to upload content
For now I recommend creating one branch for each of the ongoing updates. This way we won't have thousands of commits that would make the commit history unreadable plus all the changes are now easy and straightforward, so anyone can approve their own pr 

## Ongoing updates + branch(es) if needed
- Create neural network classes: initial/create_dnn_classes
- Create grid search parent class: 
- Create dataloaders: initial/create_loaders

## Future updates not started
- Implement GPU

## Finished updates + branch(es)
