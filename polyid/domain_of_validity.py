import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from tqdm import tqdm

tqdm.pandas()


class DoV:
    def __init__(self):
        self.fingerprint_col = "smiles_polymer"
        self.radius = 2

    def get_fp(self, smiles: str) -> pd.Series:
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

    def get_fps(self, df: pd.DataFrame) -> pd.DataFrame:
        """Gets a the fingerprint hashes for a set smiles strings in a dataframe.

        Args:
            df (pd.DataFrame): Dataframe containing a column of smiles strings for which fingerprints should be generated. The column containing the fingerprints should match DoV().fingerprint_col.

        Returns:
            pd.DataFrame: Dataframe containing the hashes for the each fingerprint and a count of their occurance in the molecule.
        """

        return df.progress_apply(
            lambda row: self.get_fp(row[self.fingerprint_col]), axis=1
        ).fillna(0)

    def __count_fps_overlap(self, row, alltraincount):
        row.index = row.index.astype(int)
        testcount = row[row != 0]

        dfcount = pd.DataFrame(
            {"testcount": testcount, "alltraincount": alltraincount}
        ).fillna(0)
        dfcount = dfcount[dfcount.testcount > 0]
        returnvalue = sum(dfcount.alltraincount == 0)

        return pd.Series(returnvalue, name="min_occ")

    def get_fps_overlap(
        self, dfpredict: pd.DataFrame, dftrain_fps: pd.DataFrame
    ) -> pd.DataFrame:
        """Finds the overlaping fingerprints between the structures in the prediction dataframe and the training datafrmae.

        Args:
            dfpredict (pd.DataFrame): dataframe containing polymers for prediction. Polymer smiles should located in self. fingerprint_col
            dftrain_fps (pd.DataFrame): dataframe containing polymers that were used for training. Polymer smiles should located in self.fingerprint_col

        Returns:
            pd.DataFrame: dfpredict_fps which is the dfpredict dataframe with the count of the fingerprints outside of the training dataframe, which is located in 'fps_notin_train'.
        """
        if type(dftrain_fps) != pd.DataFrame:
            dftrain_fps = self.dftrain_fps
        alltraincount = dftrain_fps.sum(0)
        alltraincount.index = alltraincount.index.astype(int)

        dfpredict_fps = self.get_fps(dfpredict)
        dfoccur = dfpredict_fps.apply(
            lambda row: self.__count_fps_overlap(row, alltraincount), axis=1
        )
        dfoccur.columns = ["fps_notin_train"]
        return pd.concat([dfpredict, dfoccur], axis=1)

    @property
    def radius(self):
        """The radius for taht will be used for Morgan fingerprinting."""
        return self._radius

    @radius.setter
    def radius(self, radius):
        self._radius = radius

    @property
    def fingerprint_col(self):
        """The column which should contain canonnical smiles for which fingerprint hashes will be generated."""
        return self._fingerprint_col

    @fingerprint_col.setter
    def fingerprint_col(self, fingerprint_col):
        self._fingerprint_col = fingerprint_col
