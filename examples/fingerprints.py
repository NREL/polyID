import pandas as pd
from typing import Dict, List, Union
import shortuuid
import warnings

import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem, MolFromSmiles as mfs
from rdkit.Chem import MolToSmiles as m2s
from rdkit.Chem import rdFingerprintGenerator

from mordred import Calculator, descriptors

warnings.filterwarnings('ignore')

class HierarchticalFingerprints():
    def __init__(self):
        
        try:
            self.dfmordred_description = pd.read_csv('../data/mordred_fp_descripts.csv')
        except:
            import os
            print(os.getcwd())
        
        self.fpcols_QM = self.dfmordred_description['name'].tolist()
        self.fpcols_A = []
        self.radius = 2
        self.fingerprint_col = 'smiles_polymer'
        
        self.df_atomic = pd.DataFrame()
        self.df_molecular_morphological = pd.DataFrame()
        self.df_atomic_molecular_morphological = pd.DataFrame()
        
    def _get_fp(self, smiles: str) -> pd.Series:
        """Gets a the fingerprint hashes for a single smiles string.

        Args:
            smiles (str): singe smiles string

        Returns:
            pd.Series: Series containing the hashes for the each fingerprint and a count of their occurance in the molecule.
        """

        mol = Chem.MolFromSmiles(smiles)
        fp = AllChem.GetMorganFingerprint(mol, self.radius, useFeatures=False)
        fp = pd.Series(fp.GetNonzeroElements(), name=smiles)
        return fp

    def _get_fps(self, df: pd.DataFrame) -> pd.DataFrame:
        """Gets a the fingerprint hashes for a set smiles strings in a dataframe.

        Args:
            df (pd.DataFrame): Dataframe containing a column of smiles strings for which fingerprints should be generated. The column containing the fingerprints should match DoV().fingerprint_col.

        Returns:
            pd.DataFrame: Dataframe containing the hashes for the each fingerprint and a count of their occurance in the molecule.
        """

        return df.progress_apply(
            lambda row: self._get_fp(row[self.fingerprint_col]), axis=1
        ).fillna(0)
    
    def _parse_smiles_or_mol(self,smiles_or_mol):
        if type(smiles_or_mol) ==rdkit.Chem.rdchem.Mol:
            smiles = m2s(smiles_or_mol)
            mol = smiles_or_mol
        elif type(smiles_or_mol)==str:
            smiles = smiles_or_mol
            mol = mfs(smiles_or_mol)
        else:
            assert False, "smiles_or_mol must be smiles string or rdkit.mol"
        return smiles,mol
    
    def _fingerprint_mordred(self,smiles_or_mol: Union[str,rdkit.Chem.rdchem.Mol])->pd.Series():
        smiles,mol = self._parse_smiles_or_mol(smiles_or_mol)
        calc = Calculator(descriptors, ignore_3D=True)
        df = calc.pandas([mol],quiet=True)
        ds = pd.Series(df.iloc[0].to_dict(),name=smiles)
        return ds
        
    def gen_fp_atomic(self,df):
        df = self._hash_index(df)
        self.df_atomic = self._get_fps(df)
        self.fpcols_A = self.df_atomic.keys()
        
    def gen_fp_molecular_morphological(self,df):
        df = self._hash_index(df)
        self.df_molecular_morphological = df.progress_apply(lambda row: self._fingerprint_mordred(row[self.fingerprint_col]),axis=1)
    
    def _hash_index(self,df):
        hash_cols = [self.fingerprint_col]
        ps_hash = df[hash_cols[0]].astype(str)
        for col in hash_cols[1:]:
            ps_hash = ps_hash+df[col].astype(str)
        df.index = ps_hash.apply(shortuuid.uuid)
        df.index.name = 'hash-{}'.format('-'.join(hash_cols))

        return df

    def gen_fp_atomic_molecular_morphological(self,df):
        if self.df_atomic.shape[0]==0:
            self.gen_fp_atomic(df)
        if self.df_molecular_morphological.shape[0]==0:
            self.gen_fp_molecular_morphological(df)
        
        self.df_atomic_molecular_morphological = pd.merge(self.df_atomic,self.df_molecular_morphological,left_index=True,right_index=True,how='inner')
        
        assert self.df_atomic.shape[1]+self.df_molecular_morphological.shape[1]==self.df_atomic_molecular_morphological.shape[1],"Shape mismatch."
