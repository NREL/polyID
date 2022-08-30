"""Featurizers to use with the preprocessors"""
from collections import defaultdict

import rdkit
from rdkit.Chem import AllChem


# nfp features copied to remove dependence on nfp if these are updated
def get_ring_size(obj, max_size=12):
    if not obj.IsInRing():
        return 0
    else:
        for i in range(max_size):
            if obj.IsInRingSize(i):
                return i
        else:
            return "max"


def atom_features_v1(atom):
    """Return an integer hash representing the atom type"""

    return str(
        (
            atom.GetSymbol(),
            atom.GetDegree(),
            atom.GetTotalNumHs(),
            atom.GetImplicitValence(),
            atom.GetIsAromatic(),
        )
    )


def atom_features_v2(atom):
    props = [
        "GetChiralTag",
        "GetDegree",
        "GetExplicitValence",
        "GetFormalCharge",
        "GetHybridization",
        "GetImplicitValence",
        "GetIsAromatic",
        "GetNoImplicit",
        "GetNumExplicitHs",
        "GetNumImplicitHs",
        "GetNumRadicalElectrons",
        "GetSymbol",
        "GetTotalDegree",
        "GetTotalNumHs",
        "GetTotalValence",
    ]

    atom_type = [getattr(atom, prop)() for prop in props]
    atom_type += [get_ring_size(atom)]

    return str(tuple(atom_type))


def bond_features_v1(bond, **kwargs):
    """Return an integer hash representing the bond type.
    flipped : bool
        Only valid for 'v3' version, whether to swap the begin and end atom
        types
    """

    return str(
        (
            bond.GetBondType(),
            bond.GetIsConjugated(),
            bond.IsInRing(),
            sorted([bond.GetBeginAtom().GetSymbol(), bond.GetEndAtom().GetSymbol()]),
        )
    )


def bond_features_v2(bond, **kwargs):
    return str(
        (
            bond.GetBondType(),
            bond.GetIsConjugated(),
            bond.GetStereo(),
            get_ring_size(bond),
            sorted([bond.GetBeginAtom().GetSymbol(), bond.GetEndAtom().GetSymbol()]),
        )
    )


def bond_features_v3(bond, flipped=False):
    if not flipped:
        start_atom = atom_features_v1(bond.GetBeginAtom())
        end_atom = atom_features_v1(bond.GetEndAtom())

    else:
        start_atom = atom_features_v1(bond.GetEndAtom())
        end_atom = atom_features_v1(bond.GetBeginAtom())

    return str(
        (
            bond.GetBondType(),
            bond.GetIsConjugated(),
            bond.GetStereo(),
            get_ring_size(bond),
            bond.GetEndAtom().GetSymbol(),
            start_atom,
            end_atom,
        )
    )


def bond_features_wbo(start_atom, end_atom, bondatoms):

    start_atom_symbol = bondatoms[0].GetSymbol()
    end_atom_symbol = bondatoms[1].GetSymbol()

    return str((start_atom_symbol, end_atom_symbol))


def atom_features_meso(atom: rdkit.Chem.rdchem.Atom) -> str:
    neighbor_stereo = _PHA_meso(atom)
    neighbor_stereo = [neighbor_stereo.get("left", 2), neighbor_stereo.get("right", 2)]

    return str(
        (
            atom.GetSymbol(),
            atom.GetDegree(),
            atom.GetTotalNumHs(),
            atom.GetImplicitValence(),
            atom.GetIsAromatic(),
            atom.GetChiralTag(),
            neighbor_stereo,
        )
    )


def _PHA_meso(atom: rdkit.Chem.rdchem.Atom) -> dict:
    # This method will only work for PHAs with one chiral center per monomer,
    # which attaches it to the backbone
    # Shortest path is calculated every single time, which is not efficient
    atom_id = atom.GetIdx()
    mol = atom.GetOwningMol()

    try:
        acid = AllChem.MolFromSmarts("[C;$(C[OH]);$(C=O)]")
        ol = AllChem.MolFromSmarts("[C;$(C[OH]);!$(C=O)]")
        start_atom = mol.GetSubstructMatches(acid)[0][0]
        finish_atom = mol.GetSubstructMatches(ol)[0][0]

        shortest_path = AllChem.rdmolops.GetShortestPath(mol, start_atom, finish_atom)
        rs_list = [
            (idx, cc)
            for idx, cc in AllChem.FindMolChiralCenters(mol)
            if idx in shortest_path
        ]

        neighbor_list = list(zip(*(rs_list, rs_list[1:])))

        neighbor_meso = defaultdict(dict)
        # No stereo
        if len(rs_list) < 2:
            return {}

        for diad in neighbor_list:
            is_meso = diad[0][1] == diad[1][1]
            neighbor_meso[diad[0][0]]["right"] = 1 if is_meso else 0
            neighbor_meso[diad[1][0]]["left"] = 1 if is_meso else 0

        neighbor_meso = neighbor_meso.get(atom_id, {"right": 2, "left": 2})

        return neighbor_meso
    except BaseException:
        return {}
