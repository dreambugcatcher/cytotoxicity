from pickle import load

from pandas import DataFrame, concat

from sklearn.preprocessing import FunctionTransformer, StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV, RepeatedKFold
from sklearn.svm import SVR
from sklearn.metrics import r2_score, root_mean_squared_error

from rdkit.Chem import Descriptors, MolFromSmiles

from src.data_preparation.descriptors import mol_dsc_calc

with open('../data/classifier/potency.pickle', 'rb') as file:
    potency = load(file)

descriptors_transformer = FunctionTransformer(mol_dsc_calc, validate=False)

molecules = [
    MolFromSmiles(mol) for mol in potency['SMILES']
]

X = descriptors_transformer.transform(molecules)
Y = potency['Potency']
XY = concat((X, Y), axis=1)

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

scaler_x = StandardScaler().fit(x_train)
x_train_scal = scaler_x.transform(x_train)
x_test_scal = scaler_x.transform(x_test)

scaler_y = MinMaxScaler().fit(y_train.to_numpy().reshape(-1, 1))
y_train_scal = scaler_y.transform(y_train.to_numpy().reshape(-1, 1))
y_test_scal = scaler_y.transform(y_test.to_numpy().reshape(-1, 1))

cv = RepeatedKFold(n_splits=5, n_repeats=5, random_state=42)
c = [1, 10, 50]
g = [0.1, 0.01, 0.001]
cf = [0, 0.5, 1]
epsilon = [0.01, 0.2, 0.3]
pg = [
    {
        'kernel': ['poly'],
        'gamma': g,
        'C': c, 
        'degree': [2, 3],
        'coef0' : cf,
        'epsilon': epsilon

    },
    {
        'kernel': ['sigmoid'],
        'gamma': g,
        'C' : c,
        'coef0' : cf,        
        'epsilon': epsilon
    },
    {
        'kernel':  ['rbf'],
        'gamma': g,
        'C' : c,
        'epsilon': epsilon

    }]
grid = GridSearchCV(SVR(verbose=3, shrinking=False), pg, cv=cv, verbose=3, scoring='r2', n_jobs=12)
grid.fit(x_train_scal, y_train_scal.ravel())

grid.best_estimator_

y_pred = grid.predict(x_test_scal)
print(f'R^2 = {r2_score(y_test_scal, y_pred)}')
print(f'RMSE = {root_mean_squared_error(y_test_scal, y_pred)}')
