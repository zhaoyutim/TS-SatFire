# Licensed under the Apache License, Version 2.0 (the 'License');
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an 'AS IS' BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import tensorflow as tf
import typing_extensions as tx

from temporal_models.t4fire.patch_encoder import PatchEncoder
from . import layers

ConfigDict = tx.TypedDict(
    "ConfigDict",
    {
        "dropout": float,
        "mlp_dim": int,
        "num_heads": int,
        "num_layers": int,
        "hidden_size": int,
    },
)
CONFIG_Ti: ConfigDict = {
    "dropout": 0.1,
    "mlp_dim": 768,
    "num_heads": 3,
    "num_layers": 12,
    "hidden_size": 192,
}

CONFIG_S: ConfigDict = {
    "dropout": 0.1,
    "mlp_dim": 1664,
    "num_heads": 6,
    "num_layers": 12,
    "hidden_size": 384,
}

CONFIG_B: ConfigDict = {
    "dropout": 0.1,
    "mlp_dim": 3072,
    "num_heads": 12,
    "num_layers": 12,
    "hidden_size": 768,
}

CONFIG_L: ConfigDict = {
    "dropout": 0.1,
    "mlp_dim": 4096,
    "num_heads": 16,
    "num_layers": 24,
    "hidden_size": 768,
}

def build_model(
    input_shape: tuple,
    num_layers: int,
    hidden_size: int,
    num_heads: int,
    name: str,
    mlp_dim: int,
    classes: int,
    dropout=0.1,
    activation="linear",
    include_top=True,
    representation_size=None,
    return_sequence=True,
    is_masked=True
):
    """Build a ViT model.

    Args:
        image_size: The size of input images.
        patch_size: The size of each patch (must fit evenly in image_size)
        classes: optional number of classes to classify images
            into, only to be specified if `include_top` is True, and
            if no `weights` argument is specified.
        num_layers: The number of transformer layers to use.
        hidden_size: The number of filters to use
        num_heads: The number of transformer heads
        mlp_dim: The number of dimensions for the MLP output in the transformers.
        dropout_rate: fraction of the units to drop for dense layers.
        activation: The activation to use for the final layer.
        include_top: Whether to include the final classification layer. If not,
            the output will have dimensions (batch_size, hidden_size).
        representation_size: The size of the representation prior to the
            classification layer. If None, no Dense layer is inserted.
        return_sequence: Return Sequence or not.
    """
    x = tf.keras.layers.Input(shape=input_shape)
    proj = tf.keras.layers.Dense(units=hidden_size)(x)
    y = PatchEncoder(input_shape[0], hidden_size)(x)
    y = y + proj
    for n in range(num_layers):
        y, _ = layers.TransformerBlock(
            num_heads=num_heads,
            mlp_dim=mlp_dim,
            dropout=dropout,
            is_masked=is_masked,
            name=f"Transformer/encoderblock_{n}",
        )(y)
    y = tf.keras.layers.LayerNormalization(
        epsilon=1e-6, name="Transformer/encoder_norm"
    )(y)
    if representation_size is not None:
        y = tf.keras.layers.Dense(
            representation_size, name="pre_logits", activation="tanh"
        )(y)
    if not return_sequence:
        y = tf.keras.layers.Flatten()(y)
    if include_top:
        y = tf.keras.layers.Dense(classes, name="head", activation=activation)(y)
    return tf.keras.models.Model(inputs=x, outputs=y, name=name)


def vit_base(
    input_shape = (10,45),
    classes=2,
    activation="linear",
    include_top=True,
    weights="imagenet21k+imagenet2012",
):
    model = build_model(
        **CONFIG_B,
        name="vit-b16",
        input_shape=input_shape,
        classes=classes,
        activation=activation,
        include_top=include_top,
        representation_size=768 if weights == "imagenet21k" else None,
    )
    return model

def vit_tiny(
    input_shape = (10,45),
    classes=2,
    activation="linear",
    include_top=True,
    weights="imagenet21k+imagenet2012",
):
    model = build_model(
        **CONFIG_Ti,
        name="vit-ti",
        input_shape=input_shape,
        classes=classes,
        activation=activation,
        include_top=include_top,
        representation_size=768 if weights == "imagenet21k" else None,
    )
    return model

def vit_small(
        input_shape=(10, 45),
        classes=2,
        activation="linear",
        include_top=True,
        weights="imagenet21k+imagenet2012",

):
    model = build_model(
        **CONFIG_S,
        name="vit-small",
        input_shape=input_shape,
        classes=classes,
        activation=activation,
        include_top=include_top,
        representation_size=768 if weights == "imagenet21k" else None,
    )
    return model

def vit_tiny_custom(
        input_shape = (10,45),
        classes=2,
        activation="linear",
        include_top=True,
        weights="imagenet21k+imagenet2012",
        num_heads=3,
        mlp_dim=768,
        num_layers=12,
        hidden_size=192,
        return_sequence=True,
        is_masked=True
):
    CONFIG_Ti_CUSTOM: ConfigDict = {
        "dropout": 0.1,
        "mlp_dim": mlp_dim,
        "num_heads": num_heads,
        "num_layers": num_layers,
        "hidden_size": hidden_size,
    }
    model = build_model(
        **CONFIG_Ti_CUSTOM,
        name="vit-ti",
        input_shape=input_shape,
        classes=classes,
        activation=activation,
        include_top=include_top,
        representation_size=768 if weights == "imagenet21k" else None,
        return_sequence=return_sequence,
        is_masked=is_masked
    )
    return model


