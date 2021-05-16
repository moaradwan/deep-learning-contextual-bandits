import tensorflow as tf

from deepctr.layers.core import DNN
from deepctr.layers.interaction import CrossNet

from tf_agents.networks import network


class FunctionalEncodingNetwork(network.Network):
    """Feed Forward network with CNN and FNN layers."""

    def __init__(
        self,
        model,
        input_tensor_spec,
        name="FunctionalEncodingNetwork",
    ):

        self.model = model
        super(FunctionalEncodingNetwork, self).__init__(
            input_tensor_spec=input_tensor_spec, state_spec=(), name=name
        )

        self.built = True  # Allow access to self.variables

    def call(self, inputs, network_state=(), **kwargs):
        if "step_type" in kwargs:
            kwargs.pop("step_type")
        return self.model(inputs, **kwargs), network_state


def DCN(
    cross_num,
    input_shape,
    cross_parameterization="vector",
    dnn_hidden_units=(
        128,
        128,
    ),
    l2_reg_cross=1e-5,
    l2_reg_dnn=0,
    seed=1024,
    dnn_dropout=0,
    dnn_use_bn=False,
    dnn_activation="relu",
):

    if len(dnn_hidden_units) == 0 and cross_num == 0:
        raise ValueError("Either hidden_layer or cross layer must > 0")

    inputs = tf.keras.layers.Input(shape=(input_shape,), name="input", dtype=tf.float32)
    linear_logit = tf.keras.layers.Dense(input_shape)(inputs)
    dnn_input = inputs

    if len(dnn_hidden_units) > 0 and cross_num > 0:  # Deep & Cross
        deep_out = DNN(
            dnn_hidden_units,
            dnn_activation,
            l2_reg_dnn,
            dnn_dropout,
            dnn_use_bn,
            seed=seed,
        )(dnn_input)
        cross_out = CrossNet(
            cross_num, parameterization=cross_parameterization, l2_reg=l2_reg_cross
        )(dnn_input)
        final_logit = tf.keras.layers.Concatenate()([cross_out, deep_out])
    elif len(dnn_hidden_units) > 0:  # Only Deep
        final_logit = DNN(
            dnn_hidden_units,
            dnn_activation,
            l2_reg_dnn,
            dnn_dropout,
            dnn_use_bn,
            seed=seed,
        )(dnn_input)
    elif cross_num > 0:  # Only Cross
        final_logit = CrossNet(
            cross_num, parameterization=cross_parameterization, l2_reg=l2_reg_cross
        )(dnn_input)
    else:  # Error
        raise NotImplementedError

    final_logit = tf.keras.layers.Concatenate()([final_logit, linear_logit])
    model = tf.keras.models.Model(inputs=inputs, outputs=final_logit)

    return model


def get_dcn_network(input_shape, cross_num=3):
    dcn = DCN(cross_num=cross_num, input_shape=input_shape)
    dcn_network = FunctionalEncodingNetwork(dcn, tf.TensorSpec((input_shape)))
    return dcn_network


def get_wide_deep_network(input_shape, layers):
    dcn = DCN(cross_num=0, dnn_hidden_units=layers, input_shape=input_shape)
    wdn_network = FunctionalEncodingNetwork(dcn, tf.TensorSpec((input_shape)))
    return wdn_network
