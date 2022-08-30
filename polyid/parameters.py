from typing import Dict, List


class Parameters:
    def __init__(
        self,
        kfolds: int = 3,
        batch_size: int = 64,
        epochs: int = 5,
        decay: float = 1e-5,
        learning_rate: float = 0.0005,
        atom_features: int = 32,
        bond_features: int = 32,
        mol_features: int = 8,
        num_messages: int = 2,
        dropout: float = 0.05,
        prediction_columns: List[str] = None,
        **kwargs
    ):
        """These are all default parameters. They do not guarantee a good model
        generation."""
        self.kfolds = kfolds
        self.prediction_columns = prediction_columns
        self.atom_features = atom_features
        self.mol_features = mol_features
        self.num_messages = num_messages
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.dropout = dropout
        self.decay = decay
        self.bond_features = bond_features

        # Assign any non-default key val pairs
        for key, val in kwargs.items():
            self.__setitem__(key, val)

    @property
    def training_params(self) -> Dict:
        """The training parameter inputs for generating a model.

        Returns
        -------
        Dict
            A dictionary containing the batch size, kfolds, epochs, dropout, decay,
            and learning_rate
        """
        # TODO KMS -- why are batch size and things in brackets?
        training_params = {
            "batch_size": [self.batch_size],
            "kfolds": list(range(self.kfolds)),
            "epochs": [self.epochs],
            "learning_rate": [self.learning_rate],
            "dropout": [self.dropout],
            "decay": [self.decay],
        }
        return training_params

    def to_dict(self) -> Dict:
        """Returns a dictionary with all the parameters.

        Returns
        -------
        Dict
            Dictionary containing all of the key/val pairs for parameters.
        """
        return self.__dict__

    @classmethod
    def from_dict(cls, param_dict):
        """Generate a Parameters object from a dictionary.

        Parameters
        ----------
        param_dict : Dict
            A dictionary containing key/val pairs of parameters.

        Returns
        -------
        Parameters
            A Parameters class instance.
        """
        return cls(**param_dict)

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, val):
        setattr(self, key, val)
