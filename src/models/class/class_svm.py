from rdkit.Chem import MolFromSmiles
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, RepeatedKFold
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, balanced_accuracy_score, matthews_corrcoef, roc_auc_score, f1_score, confusion_matrix
from pandas import concat
from pickle import load

from src.data_preparation.descriptors import mol_dsc_calc

with open('../data/classifier/modeling_data.pickle', 'rb') as file:
    data = load(file)

descriptors_transformer = FunctionTransformer(mol_dsc_calc, validate=False)

molecules = [
    MolFromSmiles(mol) for mol in data['SMILES']
]

X = descriptors_transformer.transform(molecules)
Y = data['Activity']
XY = concat((X, Y), axis=1)

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.15)

scaler_x = StandardScaler().fit(x_train)
x_train_scal = scaler_x.transform(x_train)
x_test_scal = scaler_x.transform(x_test)

cv = RepeatedKFold(n_splits=5, n_repeats=5, random_state=42)
c = [1, 10, 50]
g = [0.1, 0.01, 0.001]
cf = [0, 0.5, 1]
pg = [
    {
        'kernel': ['poly'],
        'gamma': g,
        'C': c, 
        'degree': [2, 3],
        'coef0' : cf,
    },
    {
        'kernel': ['sigmoid'],
        'gamma': g,
        'C' : c,
        'coef0' : cf,
    },
    {
        'kernel':  ['rbf'],
        'gamma': g,
        'C' : c,
    }
]

grid = GridSearchCV(SVC(verbose=3, shrinking=False), pg, cv=cv, verbose=3, refit='balanced_accuracy', scoring=('f1', 'balanced_accuracy'), n_jobs=12)
grid.fit(x_train_scal, y_train)

y_pred = grid.predict(x_test_scal)

tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel().tolist()
precision = tp / (tp + fp)
recall = tp / (tp + fn)

print(f'ACC = {accuracy_score(y_test, y_pred):.3f}')
print(f'BA = {balanced_accuracy_score(y_test, y_pred):.3f}')
print(f'MCC = {matthews_corrcoef(y_test, y_pred):.3f}')
print(f'ROC AUC = {roc_auc_score(y_test, y_pred):.3f}')
print(f'F1 = {f1_score(y_test, y_pred):.3f}')
print(f'Precision = {precision:.3f}')
print(f'Recall = {recall:.3f}')
