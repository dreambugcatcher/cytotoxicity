from rdkit.Chem import MolFromSmiles
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, RepeatedKFold
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix, balanced_accuracy_score, matthews_corrcoef, f1_score
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

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.20)
scaler_x = StandardScaler().fit(x_train)
x_train_scal = scaler_x.transform(x_train)
x_test_scal = scaler_x.transform(x_test)

gb = HistGradientBoostingClassifier(class_weight='balanced')
pg = {
    'learning_rate': [0.01, 0.1, 1],
    'max_depth': [None, 1, 3, 5, 7, 10],
    'max_iter': [100, 200, 300],
    'max_leaf_nodes': [30, 40, 50],
    'min_samples_leaf': [10, 20, 30],
    'l2_regularization': [0, 0.3, 0.6]
}

cv = RepeatedKFold(n_splits=5, n_repeats=5)
gs = GridSearchCV(gb, pg, verbose=3, cv=cv, refit='balanced_accuracy', scoring=('f1', 'balanced_accuracy'))

gs.fit(x_train_scal, y_train)

y_pred = gs.predict(x_test_scal)
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
