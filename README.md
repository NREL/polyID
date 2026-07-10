
<p align="center">
  <img src="https://raw.githubusercontent.com/NREL/polyID/master/images/polyID-logo_color-full.svg" alt="PolyID Logo" width="400"/>
</p>

PolyID<sup>TM</sup> provides a framework for building, training, and predicting polymer properities using graph neural networks. The codes leverages [nfp](https://pypi.org/project/nfp/), for building tensorflow-based message-passing neural networ, and [m2p](https://pypi.org/project/m2p/), for building polymer structures. The example notebooks demonstrate how to build polymer structures, train a message-passing neural network ensemble, and evaluate its predictions, following the methodology used in the publication.

1. [Quick train and predict](https://github.com/NREL/polyID/blob/master/examples/quick_train_and_predict.ipynb): `examples/quick_train_and_predict.ipynb` — an end-to-end example that loads a polymer dataset, splits it into train/holdout, trains a k-fold ensemble of message-passing neural networks, saves/loads the models (`.keras` format), and evaluates predictions on the holdout set.
2. [Checking domain of validity](https://github.com/NREL/polyID/blob/master/examples/domain_of_validity.ipynb): `examples/domain_of_validity.ipynb` — determine the domain of validity for a set of predictions.

For more details, see the manuscript [PolyID: Artificial Intelligence for Discovering Performance-Advantaged and Sustainable Polymers](https://doi.org/10.1021/acs.macromol.3c00994), _Macromolecules_ 2023.

## Cite 
If you use PolyID in your work, please cite
```
@article{wilson2023polyid,
  title={PolyID: Artificial Intelligence for Discovering Performance-Advantaged and Sustainable Polymers},
  author={Wilson, A Nolan and St John, Peter C and Marin, Daniela H and Hoyt, Caroline B and Rognerud, Erik G and Nimlos, Mark R and Cywar, Robin M and Rorrer, Nicholas A and Shebek, Kevin M and Broadbelt, Linda J and Beckham, Gregg T and Crowley, Michael F},
  journal={Macromolecules},
  volume={56},
  number={21},
  pages={8547--8557},
  year={2023},
  publisher={ACS Publications}
}
```
