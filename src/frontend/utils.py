from typing import List

from pandas import DataFrame
from numpy.typing import NDArray
from numpy import zeros

from rdkit.Chem import AllChem
from rdkit import DataStructs
from rdkit.Chem import Mol


def calc_morgan(mol: Mol) -> NDArray:
    """ генерация молекулярных отпечатков по методу Моргана с радиусом 2 и длиной 4096
    """
    mfp_gen = AllChem.GetMorganGenerator(radius=2, fpSize=4096)
    arr = zeros((1,), dtype=int)
    DataStructs.ConvertToNumpyArray(mfp_gen.GetFingerprint(mol), arr)
    return arr