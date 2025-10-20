# STGNNs and other ML models for Migratory Bird Trajectory Prediction and Species Identification

Course: COMPSCI760
Research Project: Spatiotemporal Graph Neural Networks for Predicting Migratory Bird Movements and Species Identification Using Historical Trajectory Data

### Installation

```
conda create -n cs760 python=3.11.4
conda activate cs760
conda install -c conda-forge -y \
  numpy=1.26.4 \
  pandas=2.2.3 \
  scipy=1.14.1 \
  scikit-learn=1.5.2 \
  joblib=1.2.0 \
  threadpoolctl=3.1.0 \
  openpyxl=3.1.2 \
  matplotlib=3.8.0 \
  nbformat=5.9.2 \
  jupyterlab=4.3.0 \
  lightgbm=4.0.0 \
  optuna==3.6.1 \
  plotly==5.24.1 \
  python-kaleido==0.2.1
conda install -c pytorch -y pytorch torchvision torchaudio cpuonly
```

```
# use python version 3.11.4
# using venv // [venv] is the venv name

python3 -m venv [venv]
source [venv]/bin/activate  # macOS
[venv]\Scripts\activate     # Windows

pip install -r requirements.txt
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# run the program
python3 baseline_rnn_birds.py
```

## Repo Structure
```
./data/                                     # contains all the datasets
./src/preprocessing.py                      # initial preprocessing
./preprocessing.py                          # second preprocessing

baseline_hyperparameter_optimization.py     # baseline hyperparamter optimization
baseline_lightgbm_birds.py                  # baseline lightgbm model
baseline_rnn_birds.py                       # baseline rnn model

stgnn.py                                    # stgnn (GCN version)
stgnn_gat.py                                # stgnn (GAT version)
stgnn_gat_validation.py                     # stgnn (GAT version) validation
stgnn_validation.py                         # stgnn (GCN version) validation
```
