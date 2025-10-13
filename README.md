# DiKGRec

This is the pytorch implementation of our paper "DiKGRec: Generative Recommender Model with Diffusion and Knowledge Graph-based Reasoning"


## Environment
- Anaconda 3
- python 3.7.16
- pytorch 1.12.0
- numpy 1.22.3


## Usage
The experimental data are in './Datasets' folder, including Last-FM, Yelp2018, Amazon-book and MovieLen

```
python Main.py --data lastfm --lr2 5e-4 --kg_loss_ratio 0.8 --updateW 1 --oriW 1 
```

```
python Main.py --data yelp2018 --lr2 1e-3 --kg_loss_ratio 0.4 --updateW 2 --oriW 0 --layer 2
```

```
python Main.py --data amazon-book --lr 5e-5 --lr2 5e-3 --kg_loss_ratio 0.6 --updateW 1 --oriW 1 --batch 200 --layer 4 
```


## Acknowledgement: The code is developed based on parts of the codes in the following papers:
```linux
[1] Jiang, Y., Yang, Y., Xia, L., & Huang, C. (2024, March). Diffkg: Knowledge graph diffusion model for recommendation. 
[2] Wang, W., Xu, Y., Feng, F., Lin, X., He, X., & Chua, T. S. (2023, July). Diffusion recommender model. 
```
## Acknowledgement: Data for Last-FM, Yelp2018, and Amazon-Book are referenced from the following paper:
```linux
[3] Wang, X., He, X., Cao, Y., Liu, M., & Chua, T. S. (2019, July). Kgat: Knowledge graph attention network for recommendation. 
```
## Acknowledgement: Data for MovieLen is referenced from the following paper:
```linux
[4] Mancino, A. C. M., Ferrara, A., Bufi, S., Malitesta, D., Di Noia, T., & Di Sciascio, E. (2023, September). Kgtore: tailored recommendations through knowledge-aware GNN models. 
```