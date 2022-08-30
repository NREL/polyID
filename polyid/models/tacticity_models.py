"""Models that incorporate tacticity in one way or another"""
# TODO document better

import nfp
import tensorflow as tf
from keras import layers


# TODO switch order of model_summary
def pm_model(preprocessor, model_summary, prediction_columns, params):
    # Define the keras model
    # Input layers
    atom = layers.Input(shape=[None], dtype=tf.int64, name="atom")
    bond = layers.Input(shape=[None], dtype=tf.int64, name="bond")
    connectivity = layers.Input(shape=[None, 2], dtype=tf.int64, name="connectivity")
    global_features = layers.Input(shape=[None], dtype=tf.float32, name="pm")

    num_features = params["num_features"]  # Controls the size of the model

    # Convert from a single integer defining the atom state to a vector
    # of weights associated with that class
    atom_state = layers.Embedding(
        preprocessor.atom_classes, num_features, name="atom_embedding", mask_zero=True
    )(atom)

    # Ditto with the bond state
    bond_state = layers.Embedding(
        preprocessor.bond_classes, num_features, name="bond_embedding", mask_zero=True
    )(bond)

    # Reshape the pm input
    global_features_state = layers.Reshape((1,))(global_features)

    global_features_state = layers.Dense(
        units=params["mol_features"], name="global_features_state"
    )(global_features_state)

    # Here we use our first nfp layer. This is an attention layer that looks at
    # the atom and bond states and reduces them to a single, graph-level vector.
    # mum_heads * units has to be the same dimension as the atom / bond dimension
    global_state = nfp.GlobalUpdate(units=params["mol_features"], num_heads=1)(
        [atom_state, bond_state, connectivity, global_features_state]
    )
    global_state = layers.Add()([global_state, global_features_state])

    for _ in range(params["num_messages"]):  # Do the message passing
        new_bond_state = nfp.EdgeUpdate()(
            [atom_state, bond_state, connectivity, global_state]
        )
        bond_state = layers.Add()([bond_state, new_bond_state])

        new_atom_state = nfp.NodeUpdate()(
            [atom_state, bond_state, connectivity, global_state]
        )
        atom_state = layers.Add()([atom_state, new_atom_state])

        new_global_state = nfp.GlobalUpdate(units=params["mol_features"], num_heads=1)(
            [atom_state, bond_state, connectivity, global_state]
        )
        global_state = layers.Add()([global_state, new_global_state])

    # Get prediction layers to concat together
    prediction_layers = []
    for col in prediction_columns:
        # From polyID predicting with bond_states
        prediction_layer = layers.Dense(1, name=f"{col}_dense")(bond_state)
        prediction_layer = layers.GlobalAveragePooling1D(name=col)(prediction_layer)
        prediction_layers.append(prediction_layer)

    output = layers.Concatenate(name="predictions")(prediction_layers)

    model = tf.keras.Model([atom, bond, connectivity, global_features], output)

    if model_summary:
        print(model.summary())

    return model
