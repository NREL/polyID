import nfp


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
