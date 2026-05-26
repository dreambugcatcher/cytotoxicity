# Cytotoxicity Prediction using Machine Learning

Prediction of compound cytotoxicity using molecular descriptors
and machine learning models.

## Features
- RDKit molecular descriptors
- Data preprocessing pipeline
- Random Forest / XGBoost models
- Cross-validation
- Model evaluation
- SHAP interpretability

## Tech stack
- Python
- RDKit
- scikit-learn
- pandas
- matplotlib

## Results

| Model | ROC-AUC |
|---|---|
| Random Forest | 0.84 |
| XGBoost | 0.87 |

## Installation

pip install -r requirements.txt

## Usage

python train.py

## Example

SMILES: CC(=O)Oc1ccccc1C(=O)O

Prediction:
Low cytotoxicity