# ConcealedThreatDetection

## Install dependencies

```commandline
conda create -n <myenv> python=3.8
conda activate <myenv>
sh install_libraries.sh
```

## Prepare for training
- create a `.env` file in root directory
- Paste DagsHub MLFlow credentials for running experiments

## Training
- Update `src/config/atz.yaml` as per your need.
- Issue `python src/train.py`
