# Game-based Neural Architecture Search

These files have been deprecated. The latest implementation of GNAS is shown in pytorch_GNAS.

This is the code for the paper "Game-based Neural Architecture Search"

# Requirements

Python >= 3.7.3

Tensorflow >= 1.14.0

# To search Architecture

To search CNN architectures for CIFAR-10 by GNAS, run
```
bash scripts/train_search_GNAS.sh
```
To use random search and the improved random search, respectively, run
```
bash scripts/train_search_RS.sh
```
and
```
bash scripts/train_search_GNAS_and_RS_plus.sh
```



After searching, the architectures returned by GNAS will be in ```Out_models/```, along with their validation accuracies. Your can choose several architectures with relatively high validation accuracies and train them further.

To train a fixed CNN architecture on CIFAR-10 from scratch, run
```
bash scripts/train_final.sh
```
The best found cells are shown in "best_cells.pdf".
Moreover, we show several well-performing architectures found by our algorithms in ```train_final.sh```. All of them can obtain competitive test accuracies. Their training logs are also provided.

To train a fixed CNN architecture on ImageNet from scratch, run
```
bash scripts/train_final_imagenet.sh
```


# Acknowledgements
We thank Hieu Pham for the discussion on some details of the weight-sharing model in [`ENAS`](https://github.com/melodyguan/enas) implementation. 
We furthermore thank the anonymous reviewers for their constructive comments.

