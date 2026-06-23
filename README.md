# MSCA-PLA: Multi-Scale Cross-Attention with Differentiable Pooling for Protein--Ligand Binding Affinity Prediction

Note:
Some function names in the codebase (e.g., GeoHDS-related naming)
were retained from the first implementation version for compatibility
and historical reasons.

The current algorithm proposed in this repository is referred to as
MSCA-PLA in the paper/documentation.

## Architecture

- **GeoBlock**: Geometric interaction network block with RBF distance encoding for intra- and inter-molecular edges
- **Cluster Channel**: DiffPool hierarchical pooling + cross-attention between ligand and protein clusters
- **Atom Channel**: Bidirectional protein-ligand cross-attention at atom level
- **Robust Gating Fusion**: Staged training strategy (forced → progressive → free) with anti-collapse regularization

## Requirements

```
Python >= 3.8
PyTorch >= 1.12
PyTorch Geometric >= 2.0
RDKit
scikit-learn
scipy
numpy
pandas
tqdm
lmdb
```

Install via conda:

```bash
conda env create -f GeoHDS.yaml
conda activate geohds
```

Or install manually:

```bash
pip install torch torch-geometric rdkit scikit-learn scipy numpy pandas tqdm lmdb
```

## Data

All preprocessed datasets are available on Zenodo:

**[https://zenodo.org/records/18937880](https://zenodo.org/records/18937880)**

| File | Task | Size |
|------|------|------|
| `cross.zip` | Cross-dataset binding affinity (PDBbind / CASF) | 112.7 MB |
| `diverse_protein.zip` | Diverse protein evaluation (LBA-30 / LBA-60) | 280.1 MB |
| `LEP.zip` | Ligand efficacy prediction | 36.9 MB |

Download and place data as follows:

```bash
# Task 1
unzip cross.zip -d cross_dataset/data/

# Task 2
unzip diverse_protein.zip -d diverse_protein/dataset/

# Task 3
unzip LEP.zip -d lep/dataset/
```

## Usage

### Task 1: Cross-Dataset Binding Affinity (PDBbind / CASF)

```bash
cd cross_dataset
python train.py
```


### Task 2: Diverse Protein Evaluation (LBA-30 / LBA-60)

```bash
cd diverse_protein

# LBA-30
python train.py --mode test --seqid 30 --num_epochs 15 --learning_rate 1.5e-3 \
    --data_dir dataset/split-by-sequence-identity-30/data

# LBA-60
python train.py --mode test --seqid 60 --num_epochs 500 --learning_rate 1e-3 \
    --data_dir dataset/split-by-sequence-identity-60/data
```

### Task 3: Ligand Efficacy Prediction (LEP)

```bash
cd lep
python train.py --mode test --learning_rate 15e-4 --data_dir dataset/
```

## Project Structure

```
GeoHDS/
├── cross_dataset/          # Task 1: cross-dataset binding affinity
│   ├── GeoHDS.py           #   base GNN layers
│   ├── DualStreamGeoHDS.py #   dual-stream model
│   ├── dataset_GeoHDS.py   #   data loading and graph construction
│   ├── train.py            #   training script
│   ├── split_dataset.py    #   dataset splitting utility
│   ├── config/             #   training configuration
│   └── log/                #   logging utilities
├── diverse_protein/        # Task 2: diverse protein LBA-30/60
│   ├── model.py            #   base GNN layers
│   ├── DualStreamGeoHDS.py #   dual-stream model
│   ├── data.py             #   ATOM3D data transforms
│   └── train.py            #   training script
├── lep/                    # Task 3: ligand efficacy prediction
│   ├── model.py            #   model + MLP_LEP head
│   ├── data.py             #   ATOM3D data transforms
│   └── train.py            #   training script
└── atom3d/                 # ATOM3D library (third-party)
```


```

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
