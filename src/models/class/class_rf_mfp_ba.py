from rdkit.Chem import MolFromSmiles
from sklearn.model_selection import train_test_split, GridSearchCV, RepeatedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix, balanced_accuracy_score, matthews_corrcoef, f1_score
from pandas import concat
from pickle import load

from src.data_preparation.descriptors import calc_morgan

with open('../data/classifier/modeling_data.pickle', 'rb') as file:
    data = load(file)

molecules = [
    MolFromSmiles(mol) for mol in data['SMILES']
]

X = calc_morgan(molecules)
Y = data['Activity']
XY = concat((X, Y), axis=1)

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.20)

rf = RandomForestClassifier(class_weight='balanced')
pg = {'n_estimators': [50, 100, 200, 300, 400, 500],
      'max_depth': [None, 1, 3, 5, 7, 10],
      }
cv = RepeatedKFold(n_splits=5, n_repeats=5)
gs = GridSearchCV(rf, pg, verbose=3, cv=cv, refit='balanced_accuracy', scoring=('f1', 'balanced_accuracy'))

gs.fit(x_train, y_train)

y_pred = gs.predict(x_test)
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
