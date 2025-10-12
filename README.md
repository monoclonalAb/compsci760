# compsci760
Spatiotemporal Graph Neural Networks for Predicting Migratory Bird Movements and Species Identification Using Historical Trajectory Data

### Installation

```
conda create -n 760 python=3.11.4
conda activate 760
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
  lightgbm=4.0.0
conda install -c pytorch -y pytorch torchvision torchaudio cpuonly
```

```
# using venv // [venv] is the venv name

python3 -m venv [venv]
source [venv]/bin/activate  # macOS
[venv]\Scripts\activate     # Windows

pip install -r requirements.txt
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# run the program
python3 baseline_rnn_birds.py
```
