import selfies as sf

from rdkit import Chem


def get_selfies(smi, **kwargs):
    try:
        return sf.encoder(smi, **kwargs)
    except:
        mol = Chem.MolFromSmiles(smi, **kwargs)
        smiles = Chem.MolToSmiles(mol, canonical=True, kekuleSmiles=True)
        return sf.encoder(smiles)