# Cytotoxicity Prediction using Machine Learning

Prediction of compound cytotoxicity using molecular descriptors
and machine learning models.

## Features
- RDKit molecular descriptors
- Data preprocessing pipeline
- Random Forest / XGBoost / SVM models
- Cross-validation
- Model evaluation
- GUI app, docker implementation

## Tech stack
- Python
- RDKit
- scikit-learn
- pandas
- matplotlib
- gradio

## Results

- High classification indicators

## Installation

### For direct use
```
conda create -n cytotox_env
conda activate cytotox_env

pip install -r requirements.txt
```

### For docker use
```
docker build . -t cytotox
```

## Usage

### For direct use
```
python project_folder/src/frontend/main.py
```

### For docker use
```
docker run [-it/-d] -p 5000:5000 cytotox
```

### Then you can use it for prediction on ```localhost:5000``` or wherever you specify in the file main.py

## Example

SMILES: CC(=O)Oc1ccccc1C(=O)O

Prediction:
Low cytotoxicity