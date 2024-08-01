
<p align="center">
  <img src="https://raw.githubusercontent.com/NREL/polyID/master/images/polyID-logo_color-full.svg" alt="PolyID Logo" width="400"/>
</p>

PolyID<sup>TM</sup> provides a framework for building, training, and predicting polymer properities using graph neural networks. The codes leverages [nfp](https://pypi.org/project/nfp/), for building tensorflow-based message-passing neural networ, and [m2p](https://pypi.org/project/m2p/), for building polymer structures.  The notebooks have been provided that demonstrate how to: (1) build polymer structures from a polymer database and split into a training/validation and test set, (2) train a message passing neural network from using the trainining/validation set, and (3) evaluate the trained network on the test set. These three notebooks follow the methodology used in the forthcoming publication.

1. [Building polymer structures](https://github.com/NREL/polyID/blob/master/examples/1_generate_polymer_structures.ipynb): `examples/1_generate_polymer_structures.ipynb`
2. [Training a message passing neural network](https://github.com/NREL/polyID/blob/master/examples/2_generate_and_train_models.ipynb): `examples/2_generate_and_train_models.ipynb`
3. [Predicting and evaluating a trained network](https://github.com/NREL/polyID/blob/master/examples/3_evaluate_model_performance_and_DoV.ipynb): `examples/3_evaluate_model_performance_and_DoV.ipynb`

Additional notebooks have been provided to provide more examples and capabilities of the PolyID code base.

4. [Checking domain of validity](https://github.com/NREL/polyID/blob/master/examples/example_determine_domain-of-validity.ipynb): `examples/example_determine_domain-of-validity.ipynb`
5. [Generating hierarchical fingerprints for performance comparison](https://github.com/NREL/polyID/blob/master/examples/example_hierarchical_fingerprints.ipynb): `examples/example_hierarchical_fingerprints.ipynb`
6. [Predicting with the trained model](https://github.com/NREL/polyID/blob/master/examples/example_predict_with_trained_models.ipynb): `examples/example_predict_with_trained_models.ipynb` 

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
