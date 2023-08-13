import nfp
import numpy as np
import pandas as pd

# Define how to featurize the input molecules
def atom_featurizer(atom):
    """ Return an string representing the atom type
    """
    return str(
        (
            atom.GetSymbol(),
            atom.GetIsAromatic(),
            nfp.get_ring_size(atom, max_size=6),
            atom.GetDegree(),
            atom.GetTotalNumHs(includeNeighbors=True),
        )
    )


def bond_featurizer(bond, flipped=False):
    """ Get a similar classification of the bond type.
    Flipped indicates which 'direction' the bond edge is pointing. """

    if not flipped:
        atoms = "{}-{}".format(
            *tuple((bond.GetBeginAtom().GetSymbol(), bond.GetEndAtom().GetSymbol()))
        )
    else:
        atoms = "{}-{}".format(
            *tuple((bond.GetEndAtom().GetSymbol(), bond.GetBeginAtom().GetSymbol()))
        )

    btype = str(bond.GetBondType())
    ring = "R{}".format(nfp.get_ring_size(bond, max_size=6)) if bond.IsInRing() else ""

    return " ".join([atoms, btype, ring]).strip()

def evaluate_error(df):
    # inverse log transform
    cols = list(df.keys()[df.keys().str.contains('log10')])
    df[cols] = df[cols].apply(lambda col: 10**col)
    df = df.rename({col:col.replace('log10_','') for col in cols},axis=1)

    # stack into prediction
    cols_meta = list(df.keys()[(~df.keys().str.contains('_mean'))&(~df.keys().str.contains('_pred'))])
    dfpred = df[cols_meta+list(df.keys()[df.keys().str.contains('_pred_mean')])]
    dfpred = dfpred.melt(id_vars=cols_meta)
    dfpred.variable = dfpred.variable.str.replace('_pred_mean','')
    dfpred.rename({'value':'y'},axis=1,inplace=True)

    dfobv = df[df.keys()[~df.keys().str.contains('_pred_mean')]]
    dfobv = dfobv.melt(id_vars=cols_meta)
    dfobv.variable = dfobv.variable.str.replace('_mean','')
    dfobv.rename({'value':'yhat'},axis=1,inplace=True)

    # calculate absolule error
    cols_merge = ['distribution','smiles_monomer','variable']
    dfreturn = pd.merge(dfobv,dfpred,left_on=cols_merge,right_on=cols_merge,how='inner')
    dfreturn['abserr'] = np.abs(dfreturn.y-dfreturn.yhat)

    # log transform absolute error
    def logtransformgroup(g):
        gname = g['variable'].iloc[0]
        if 'Perm' in gname:
            g['abserr'] = np.log10(g['abserr'])
            g['variable'] = 'log10_'+gname
        return g
        
    dfreturn = dfreturn.groupby('variable').apply(logtransformgroup)
    return dfreturn