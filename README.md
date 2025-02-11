# DiKGRec

This is the pytorch implementation of our paper "DiKGRec: Generative Recommender Model with Diffusion and Knowledge Graph-based Reasoning"


## Environment
- Anaconda 3
- python 3.7.16
- pytorch 1.12.0
- numpy 1.22.3


## Usage
The experimental data are in './Datasets' folder, including Last-FM, Yelp2018 and Amazon-book

```
python Main.py --data lastfm --kg_norm 2 --epoch 500 --lr2 5e-3 --layer 4 --head 2 --kg_loss_ratio 0.8
```

```
python Main.py --data lastfm --kg_norm 2 --epoch 500 --lr2 5e-3 --layer 4 --kg_loss_ratio 0.8
```

```
python Main.py --data yelp2018 --noise_ratio 0.3 --lr2 5e-3 --kg_norm 2 --head 2 --kg_loss_ratio 0.2 
```


## Acknowledgement: The code is developed based on parts of the codes in the following papers:
```linux
[1] Jiang, Y., Yang, Y., Xia, L., & Huang, C. (2024, March). Diffkg: Knowledge graph diffusion model for recommendation. 
[2] Wang, W., Xu, Y., Feng, F., Lin, X., He, X., & Chua, T. S. (2023, July). Diffusion recommender model. 
```
