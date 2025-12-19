# DiKGRec

This is the PyTorch implementation of our paper **"DiKGRec: Generative Recommender Model with Diffusion and Knowledge Graph-based Reasoning"**, accepted at KDD 2026.

## Abstract

DiKGRec is a novel generative recommender system that integrates diffusion models with knowledge graph-based reasoning to enhance recommendation accuracy and interpretability. By leveraging diffusion processes for generative modeling and knowledge graphs for relational reasoning, DiKGRec addresses cold-start problems and improves user-item interaction predictions. This repository provides the complete codebase for reproducing the experiments and results presented in the paper.

For more details, please refer to the full paper: [DOI: 10.1145/3770854.3780190]

## Environment

- Anaconda 3
- Python 3.7.16
- PyTorch 1.12.0
- NumPy 1.22.3
- Other dependencies: scipy, tqdm, torch-scatter (install via `pip install torch-scatter -f https://data.pyg.org/whl/torch-1.12.0+cu116.html`)

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/DiKGRec.git
   cd DiKGRec
   ```

2. Create a conda environment:
   ```bash
   conda create -n dikgrec python=3.7.16
   conda activate dikgrec
   ```

3. Install dependencies:
   ```bash
   pip install torch==1.12.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu116
   pip install numpy==1.22.3 scipy tqdm
   pip install torch-scatter -f https://data.pyg.org/whl/torch-1.12.0+cu116.html
   ```

## Datasets

The experimental datasets are located in the `./Datasets` folder and include:
- Last-FM
- Yelp2018
- Amazon-Book
- MovieLen

Each dataset contains:
- `train.txt`: Training user-item interactions
- `test.txt`: Test user-item interactions
- `kg.txt`: Knowledge graph triples

## Usage

To run the model, use the following commands for different datasets. Adjust hyperparameters as needed.

### Last-FM
```bash
python Main.py --data lastfm --lr2 5e-4 --kg_loss_ratio 0.8 --updateW 1 --oriW 1
```

### Yelp2018
```bash
python Main.py --data yelp2018 --lr2 1e-3 --kg_loss_ratio 0.4 --updateW 2 --oriW 0 --layer 2
```

### Amazon-Book
```bash
python Main.py --data amazon-book --lr 5e-5 --lr2 5e-3 --kg_loss_ratio 0.6 --updateW 1 --oriW 1 --batch 200 --layer 4
```

### MovieLen
```bash
python Main.py --data movielen --lr 5e-5 --lr2 5e-4 --kg_loss_ratio 0.5 --updateW 1 --oriW 1 --batch 100 --layer 3
```

Key parameters:
- `--data`: Dataset name (lastfm, yelp2018, amazon-book, movielen)
- `--lr` / `--lr2`: Learning rates for different components
- `--kg_loss_ratio`: Weight for knowledge graph loss
- `--updateW` / `--oriW`: Weights for KG aggregation
- `--layer`: Number of KG aggregation layers
- `--batch`: Batch size
- `--cold_start_num`: Number of cold-start items (default: 0)

For cold-start evaluation, set `--cold_start_num` to a positive integer (e.g., 100).

## Results

The results from the experiments are saved in `.stdout` files (e.g., `result_amazon_bbest_combined.stdout`). To reproduce the results, run the commands above and check the output logs. Expected metrics include Recall@20 and NDCG@20.

Example output snippet:
```
Best epoch: 150, Recall: 0.0456, NDCG: 0.0321
```

For detailed hyperparameter tuning and ablation studies, refer to the paper.

## Citation

If you use this code or find our work helpful, please cite our paper:

```bibtex
@inproceedings{zheng2026dikgrec,
  title={DiKGRec: Generative Recommender Model with Diffusion and Knowledge Graph-based Reasoning},
  author={Zheng, Your Name and Co-authors},
  booktitle={Proceedings of the 32nd ACM SIGKDD Conference on Knowledge Discovery and Data Mining (KDD '26)},
  year={2026},
  pages={TBD},
  doi={TBD}
}
```

## Acknowledgements

The code is developed based on parts of the codes in the following papers:
- [1] Jiang, Y., Yang, Y., Xia, L., & Huang, C. (2024, March). DiffKG: Knowledge graph diffusion model for recommendation.
- [2] Wang, W., Xu, Y., Feng, F., Lin, X., He, X., & Chua, T. S. (2023, July). Diffusion recommender model.

Data for Last-FM, Yelp2018, and Amazon-Book are referenced from:
- [3] Wang, X., He, X., Cao, Y., Liu, M., & Chua, T. S. (2019, July). KGAT: Knowledge graph attention network for recommendation.

Data for MovieLen is referenced from:
- [4] Mancino, A. C. M., Ferrara, A., Bufi, S., Malitesta, D., Di Noia, T., & Di Sciascio, E. (2023, September). KGToRe: Tailored recommendations through knowledge-aware GNN models.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

For questions or issues, please open an issue on GitHub or contact.
