from pickle import load
import numpy as np
from pandas import concat

import torch
from torch.utils.data import DataLoader
from torch.nn import functional as F
from torch import nn
from pytorch_lightning.core import LightningModule
import pytorch_lightning as pyl

from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split

from rdkit.Chem import MolFromSmiles

from src.data_preparation.descriptors import calc_morgan


class PotencyModel(LightningModule):
    def __init__(self):
        super().__init__()
        self.test_predictions = []
        self.targets = []
        
        self.fingerprint_fc = nn.Sequential(
            nn.Linear(2048, 1012),
            nn.LeakyReLU(),
            nn.Linear(1012, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 1)
        )

        self.loss_fn = nn.MSELoss()

    def forward(self, fingerprint):
        return self.fingerprint_fc(fingerprint)

    def training_step(self, batch, batch_idx):
        fingerprint, y = batch
        y_pred = self(fingerprint)

        loss = self.loss_fn(y_pred, y)
        self.log('Train MSE', loss, on_step=False, on_epoch=True, prog_bar=True)

        r2 = r2_score(y.detach().cpu().numpy(), y_pred.detach().cpu().numpy())
        self.log('Train R²', r2, on_step=False, on_epoch=True, prog_bar=True)

        return loss
    
    def validation_step(self, batch, batch_idx):
        fingerprint, y = batch
        y_pred = self(fingerprint)

        loss = self.loss_fn(y_pred, y)
        self.log('Validation MSE', loss, on_step=False, on_epoch=True, prog_bar=True)

        r2 = r2_score(y.cpu().numpy(), y_pred.cpu().numpy())
        self.log('Validation R²', r2, on_step=False, on_epoch=True, prog_bar=True)

        return loss
    
    def test_step(self, batch, batch_idx):
        fingerprint, y = batch
        y_pred = self(fingerprint)

        loss = self.loss_fn(y_pred, y)
        self.log('Test MSE', loss, on_step=False, on_epoch=True, prog_bar=True)
    
        self.test_predictions.extend(y_pred.cpu().numpy())
        self.targets.extend(y.cpu().numpy())
        
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)


with open('../data/classifier/no_dubl_potency.pickle', 'rb') as file:
    potency = load(file)

molecules = [
    MolFromSmiles(mol) for mol in potency['SMILES']
]

X = calc_morgan(molecules)
Y = potency['Potency']
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

train_loader = DataLoader(concat((x_train, y_train), axis=1).to_numpy(), batch_size=2)
test_loader = DataLoader(concat((x_test, y_test), axis=1).to_numpy(), batch_size=2)

model = PotencyModel()

trainer = pyl.Trainer(max_epochs=5, accelerator="auto")
trainer.fit(model, train_loader)

trainer.test(model, test_loader)

y_pred_test = model.test_predictions
y_true = model.targets
mse = mean_squared_error(y_true, y_pred_test)
q2 = r2_score(y_true, y_pred_test)
rmse = np.sqrt(mse)
print(f'test MSE: {mse:.4f}, test PRMSE: {rmse:.4f}, test Q²: {q2:.4f}')
