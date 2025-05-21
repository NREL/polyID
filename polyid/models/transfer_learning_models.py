import nfp
import tensorflow as tf
import numpy as np
from nfp import masked_mean_absolute_error
#from tensorflow.keras import layers
from keras import layers
from .base_models import message_block


def get_last_graph_layers(md_model, num_messages):
    # get the last atom, bond, and global feature vectors (after the residual connection)
    # TODO automatically extract the last bond, atom, and global layers
    last_bond_layer = md_model.get_layer(f"edge_update_{num_messages-1}")
    last_bond_res_layer = md_model.layers[md_model.layers.index(last_bond_layer) + 1]
    print(f"{last_bond_layer.name = }, {last_bond_res_layer.name = }")

    last_atom_layer = md_model.get_layer(f"node_update_{num_messages-2}")
    last_atom_res_layer = md_model.layers[md_model.layers.index(last_atom_layer) + 1]
    print(f"{last_atom_layer.name = }, {last_atom_res_layer.name = }")

    last_mol_layer = md_model.get_layer(f"global_update_{num_messages}")
    last_mol_res_layer = md_model.layers[md_model.layers.index(last_mol_layer) + 1]
    print(f"{last_mol_layer.name = }, {last_mol_res_layer.name = }")

    return last_atom_res_layer.output, last_bond_res_layer.output, last_mol_res_layer.output


def transfer_learning_model(md_model, preprocessor, model_summary=False, prediction_columns=None, params=None):
    """ Starting from a model trained on MD data, freeze and/or add layers and train on the polymer data
    """

    # keep the inputs the same 
    connectivity_layer = md_model.get_layer("connectivity")
    connectivity = connectivity_layer.output

    if params.get("freeze_to_message_block") is not None:
        # freeze all layers up to the nth message block
        print(f"Freezing layers up to {params['freeze_to_message_block']} message blocks")
        # TODO is the edge, node, or global update layer the last layer of the message block?
        bond_layer = md_model.get_layer(f"edge_update_{params['freeze_to_message_block'] - 1}")
        print(f"{params['freeze_to_message_block'] = }, {bond_layer.name = }, "
              f"{md_model.layers.index(bond_layer) = }")
        for i in range(md_model.layers.index(bond_layer) + 1):
            md_model.layers[i].trainable = False
        
        print(f"Number of trainable parameters: {np.sum([np.prod(v.get_shape()) for v in md_model.trainable_weights])}")
        print(f"Number of frozen parameters: {np.sum([np.prod(v.get_shape()) for v in md_model.non_trainable_weights])}")


    atom_state, bond_state, global_state = get_last_graph_layers(md_model, params["num_messages"])
    if params.get("add_messages") is not None:
        # first rename the original layers to avoid name collisions
        for layer in md_model.layers:
            if layer.name in md_model.input_names:
                continue
            layer._name = f"md_{layer.name}"

        print(f"Adding {params['add_messages']} message passing layers")
        # add message passing layers
        for i in range(params["add_messages"]):
            message_idx = params["num_messages"] + i
            atom_state, bond_state, global_state = message_block(
                params, atom_state, bond_state, global_state, connectivity, message_idx,
            )

    # Add the new prediction layers
    num_predictions = len(prediction_columns)
    outputs = []

    for i in range(num_predictions):
        bond_state_i = layers.Dropout(params["dropout"])(bond_state)
        bond_values = layers.Dense(1, name="bondwise_values_{}".format(i))(bond_state_i)
        output = layers.GlobalAveragePooling1D(name=prediction_columns[i])(bond_values)
        outputs.append(output)

    if len(outputs) > 1:
        outputs = layers.Concatenate(name="all_predictions")(outputs)

    # compile model
    model = tf.keras.Model(md_model.inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=params["learning_rate"], weight_decay=params["decay"]
        ),
        loss=[masked_mean_absolute_error],
    )

    print(f"Total number of trainable parameters: {np.sum([np.prod(v.get_shape()) for v in model.trainable_weights])}")
    # if modelsummary:model.summary()
    return model
