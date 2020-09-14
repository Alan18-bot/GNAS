# Game-based Neural Architecture Search

This is the code for the paper "Game-based Neural Architecture Search"

# Requirements

Python >= 3.6.11

Pytorch >= 1.6.0

# Architecture Search

To search CNN architectures by GNAS and the improved random search, run
```
bash pytorch_GNAS/train_search.sh
```
Additionally, we provide the random seeds in ``` train_search.sh ``` to reproduce the distribution figures in our paper.




# Architecture Evaluation

After searching, the architectures returned by GNAS are given in ```GNAS_Arcs/```, along with their validation accuracies. We choose the architectures with high validation accuracies and train them further.

To train a found architecture from scratch, run
```
bash pytorch_GNAS/train_final.sh
```
The default architecture is our best found architecture. The expected result is 2.59% test error.

# Acknowledgements
We thank Liu et al. for the discussion on some training details in [`DARTS`](https://github.com/quark0/darts) implementation. 
We furthermore thank the anonymous reviewers for their constructive comments.

