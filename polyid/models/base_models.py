import nfp
import tensorflow as tf
from nfp import masked_mean_absolute_error
from tensorflow.keras import layers


def global100(preprocessor, model_summary=False, prediction_columns=None, params=None):
    "this is the global state where the output is the bonds_states"

    # for k,v in params.items():
    #     if k in model_params.keys():
    #         params[k] = model_params[k]

    # Raw (integer) graph inputs
    atom = layers.Input(shape=[None], dtype=tf.int64, name="atom")
    bond = layers.Input(shape=[None], dtype=tf.int64, name="bond")
    connectivity = layers.Input(shape=[None, 2], dtype=tf.int64, name="connectivity")

    # Initialize the atom states
    atom_state = layers.Embedding(
        preprocessor.atom_classes,
        params["atom_features"],
        name="atom_embedding",
        mask_zero=True,
    )(atom)

    # Initialize the bond states
    bond_state = layers.Embedding(
        preprocessor.bond_classes,
        params["atom_features"],
        name="bond_embedding",
        mask_zero=True,
    )(bond)

    # Here we use our first nfp layer. This is an attention layer that looks at
    # the atom and bond states and reduces them to a single, graph-level vector.
    # mum_heads * units has to be the same dimension as the atom / bond dimension
    global_state = nfp.GlobalUpdate(units=params["mol_features"], num_heads=1)(
        [atom_state, bond_state, connectivity]
    )

    def message_block(atom_state, bond_state, global_state, connectivity, i):

        # Global update
        global_update = nfp.GlobalUpdate(units=params["mol_features"], num_heads=1)(
            [atom_state, bond_state, connectivity, global_state]
        )
        global_state = layers.Add()(
            [global_state, global_update]
        )  # global difference calculation

        # Bond update
        bond_update = nfp.EdgeUpdate()(
            [atom_state, bond_state, connectivity, global_state]
        )
        bond_state = layers.Add()([bond_state, bond_update])

        # Atom update
        new_atom_state = nfp.NodeUpdate()(
            [atom_state, bond_state, connectivity, global_state]
        )
        atom_state = layers.Add()([atom_state, new_atom_state])

        return atom_state, bond_state, global_state

    for j in range(params["num_messages"]):
        atom_state, bond_state, global_state = message_block(
            atom_state, bond_state, global_state, connectivity, j
        )

    # outputs
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
    model = tf.keras.Model([atom, bond, connectivity], outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=params["learning_rate"], decay=params["decay"]
        ),
        loss=[masked_mean_absolute_error],
    )

    # if modelsummary:model.summary()
    return model
