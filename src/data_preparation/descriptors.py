from typing import List

from pandas import DataFrame
import numpy as np
from numpy.typing import NDArray

from rdkit.Chem import AllChem, Descriptors
from rdkit import DataStructs
from rdkit.Chem import Mol


def calc_morgan(mols: List[Mol]) -> DataFrame[NDArray]:
    """ генерация молекулярных отпечатков по методу Моргана с радиусом 2 и длиной 2048
    """
    mfp_gen = AllChem.GetMorganGenerator(radius=2, ) 
    for_df = []
    for m in mols:
        arr = np.zeros((1,), dtype=int)
        DataStructs.ConvertToNumpyArray(mfp_gen.GetFingerprint(m), arr)
        for_df.append(arr)
    return DataFrame(for_df)


#создаем словарь из дескприторов структуры
ConstDescriptors = {"HeavyAtomCount" : Descriptors.HeavyAtomCount,
                       "NHOHCount" : Descriptors.NHOHCount,
                       "NOCount" : Descriptors.NOCount,
                       "NumHAcceptors" : Descriptors.NumHAcceptors,
                       "NumHDonors" : Descriptors.NumHDonors,
                       "NumHeteroatoms" : Descriptors.NumHeteroatoms,
                       "NumRotatableBonds" : Descriptors.NumRotatableBonds,
                       "NumValenceElectrons" : Descriptors.NumValenceElectrons,
                       "NumAromaticRings" : Descriptors.NumAromaticRings,
                       "NumAliphaticHeterocycles" : Descriptors.NumAliphaticHeterocycles,
                       "RingCount" : Descriptors.RingCount}

#создаем словарь из физико-химических дескрипторов                            
PhisChemDescriptors = {"MW" : Descriptors.MolWt,
                          "LogP" : Descriptors.MolLogP,
                          "MR" : Descriptors.MolMR,
                          "TPSA" : Descriptors.TPSA}

#объединяем все дескрипторы в один словарь
descriptors = {}
descriptors.update(ConstDescriptors)
descriptors.update(PhisChemDescriptors)

#функция для генерации дескрипторов из молекул
def mol_dsc_calc(mols: List[Mol]) -> DataFrame:

    return DataFrame({k: f(m) for k, f in descriptors.items()} 
             for m in mols)
