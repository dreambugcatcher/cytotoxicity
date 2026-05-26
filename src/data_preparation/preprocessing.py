from pandas import DataFrame, read_csv, concat
from sklearn.preprocessing import LabelEncoder
from pickle import dump
import chython as ch
from tqdm import tqdm


data = concat((read_csv('../../data/classifier/AID_1345082_datatable_all.csv'),
               read_csv('../../data/classifier/AID_1345083_datatable_all.csv')))

data = data.dropna(subset='PUBCHEM_EXT_DATASOURCE_SMILES')
data = data[data['PUBCHEM_ACTIVITY_OUTCOME'] != 'Inconclusive']
modeling_data = concat((data['PUBCHEM_EXT_DATASOURCE_SMILES'],
                        data['PUBCHEM_ACTIVITY_OUTCOME']), axis=1)
modeling_data.columns = ['SMILES', 'ACTIVITY']
modeling_data = modeling_data.reset_index()
del modeling_data['index']

y = LabelEncoder().fit_transform(modeling_data['ACTIVITY'])

for index, _smiles in enumerate(modeling_data['SMILES']):
    mol = ch.smiles(_smiles)
    mol.canonicalize()
    mol.clean_isotopes()
    if mol.check_valence():
        raise Exception('Valence ERROR!')
    
    modeling_data.loc[index, 'SMILES'] = str(mol)

modeling_data = concat((modeling_data['SMILES'], DataFrame(y, columns=['ACTIVITY_bin'])), axis=1)

_smiles = modeling_data['SMILES'].value_counts().index

no_dubl_data = DataFrame(columns=['SMILES', 'Activity'])

for smi in tqdm(_smiles):
    _table = modeling_data[modeling_data['SMILES'] == smi]
    act = _table['ACTIVITY_bin'].iloc[0]
    no_dubl_data.loc[len(no_dubl_data)] = [smi, act]

test = {}
for smi, act in zip(modeling_data.to_dict()['SMILES'].values(), modeling_data.to_dict()['ACTIVITY_bin'].values()):
    test[smi] = act

test = DataFrame.from_dict({'SMILES': [smi for smi in test.keys()], 'Activity': [act for act in test.values()]})
with open('../../data/classifier/modeling_data.pickle', 'wb') as file:
    dump(test, file)

potency = concat((
    data['PUBCHEM_EXT_DATASOURCE_SMILES'],
    data['Potency']
), axis=1).dropna(subset='PUBCHEM_EXT_DATASOURCE_SMILES').dropna(subset='Potency')
potency.columns = ['SMILES', 'Potency']
save_dubl = potency.reset_index().drop('index', axis=1).drop_duplicates()
save_dubl['Potency'] = save_dubl['Potency'].astype(float)

save_min = DataFrame(columns=['SMILES', 'Potency'])
for i, smi in tqdm(enumerate(save_dubl['SMILES'].value_counts().index)):
    mid = save_dubl[save_dubl['SMILES'] == smi]['Potency'].min()
    save_min.loc[i] = [smi, mid]

for index, _smiles in tqdm(enumerate(save_min['SMILES'])):
    mol = ch.smiles(_smiles)
    mol.canonicalize()
    mol.clean_isotopes()
    if mol.check_valence():
        raise Exception('Valence ERROR!')
    
    save_min.loc[index, 'SMILES'] = str(mol)

with open('../../data/classifier/potency_min.pickle', 'wb') as file:
    dump(save_min, file)

potency = data.dropna(subset='Potency')
potency = potency.reset_index()
del potency['index']
potency = concat((
    potency['PUBCHEM_EXT_DATASOURCE_SMILES'],
    potency['Potency']
), axis=1)
potency.columns = ['SMILES', 'Potency']

for index, _smiles in zip(range(len(potency)), potency['SMILES']):
    mol = ch.smiles(_smiles)
    mol.canonicalize()
    mol.clean_isotopes()
    if mol.check_valence():
        raise Exception('Valence ERROR!')
    
    potency.loc[index, 'SMILES'] = str(mol)
else:
    print('potency prepare DONE!')

_smiles = potency['SMILES'].value_counts().index

no_dubl_potency = DataFrame(columns=['SMILES', 'Potency'])

for smi in _smiles:
    _table = potency[potency['SMILES'] == smi]
    mid = _table['Potency'].astype(float).mean()
    no_dubl_potency.loc[len(no_dubl_potency)] = [smi, mid]

with open('no_dubl_potency.pickle', 'wb') as file:
    dump(no_dubl_potency, file)
