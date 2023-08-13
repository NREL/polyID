
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

Details for the methods are forthcoming in an upcoming manuscript.
