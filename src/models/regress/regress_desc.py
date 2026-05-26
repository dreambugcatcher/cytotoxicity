from pickle import load

from pandas import concat

from sklearn.preprocessing import FunctionTransformer, MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, RepeatedKFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, root_mean_squared_error

from rdkit.Chem import MolFromSmiles

from src.data_preparation.descriptors import mol_dsc_calc

with open('../data/classifier/no_dubl_potency.pickle', 'rb') as file:
    potency = load(file)

descriptors_transformer = FunctionTransformer(mol_dsc_calc, validate=False)

molecules = [
    MolFromSmiles(mol) for mol in potency['SMILES']
]

X = descriptors_transformer.transform(molecules)
Y = potency['Potency']
XY = concat((X, Y), axis=1)

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.15)

scaler_x = StandardScaler().fit(x_train)
x_train_scal = scaler_x.transform(x_train)
x_test_scal = scaler_x.transform(x_test)

scaler_y = MinMaxScaler().fit(y_train.to_numpy().reshape(-1, 1))
y_train_scal = scaler_y.transform(y_train.to_numpy().reshape(-1, 1))
y_test_scal = scaler_y.transform(y_test.to_numpy().reshape(-1, 1))

rf = RandomForestRegressor()
pg = {'n_estimators': [50, 100, 200, 300, 400, 500],
      'max_depth': [None, 1, 3, 5, 7, 10],
      }
cv = RepeatedKFold(n_splits=5, n_repeats=5)
gs = GridSearchCV(rf, pg, verbose=3, cv=cv)

gs.fit(x_train_scal, y_train_scal.ravel())

gs.best_estimator_

y_pred = gs.predict(x_test_scal)
print(f'R^2 = {r2_score(y_test_scal, y_pred)}')
print(f'RMSE = {root_mean_squared_error(y_test_scal, y_pred)}')
