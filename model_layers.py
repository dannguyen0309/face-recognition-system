# model_layers.py
import tensorflow as tf
import keras
from keras import layers

# ====== Functions used in Lambda when training (CBAM + L2) ======
@keras.saving.register_keras_serializable(package="custom")
def avg_fn(x):
    """Spatial average over channels, keepdims=True"""
    return tf.reduce_mean(x, axis=-1, keepdims=True)

@keras.saving.register_keras_serializable(package="custom")
def max_fn(x):
    """Spatial max over channels, keepdims=True"""
    return tf.reduce_max(x, axis=-1, keepdims=True)

@keras.saving.register_keras_serializable(package="custom")
def sp_out_shape(input_shape):
    """(B, H, W, C) -> (B, H, W, 1) cho spatial attention."""
    return (input_shape[0], input_shape[1], input_shape[2], 1)

@keras.saving.register_keras_serializable(package="custom")
def l2_fn(t):
    """L2-normalize embedding along the last axis."""
    return tf.math.l2_normalize(t, axis=-1)

@keras.saving.register_keras_serializable(package="custom")
def l2_out_shape(input_shape):
    """L2 does not change shape."""
    return input_shape


# ====== TransformerBlock (Just like during training) ======

@keras.saving.register_keras_serializable(package="custom")
class TransformerBlock(layers.Layer):
    """Tiny ViT-like encoder block over CNN tokens (matching training code)."""
    def __init__(
        self,
        d_model,
        num_heads,
        mlp_dim,
        attn_drop=0.0,
        proj_drop=0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.num_heads = num_heads
        self.mlp_dim = mlp_dim
        self.attn_drop = attn_drop
        self.proj_drop = proj_drop

        self.norm1 = layers.LayerNormalization(epsilon=1e-6, dtype="float32")
        self.mha = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=d_model // num_heads,
            dropout=attn_drop,
            dtype="float32",
        )
        self.drop1 = layers.Dropout(proj_drop)
        self.norm2 = layers.LayerNormalization(epsilon=1e-6, dtype="float32")
        self.ffn = keras.Sequential(
            [
                layers.Dense(mlp_dim, activation="gelu", dtype="float32"),
                layers.Dropout(proj_drop),
                layers.Dense(d_model, dtype="float32"),
                layers.Dropout(proj_drop),
            ]
        )

    def call(self, x, training=False):
        h = self.norm1(x)
        h = self.mha(h, h, training=training)
        h = self.drop1(h, training=training)
        x = layers.Add()([x, h])

        h2 = self.norm2(x)
        h2 = self.ffn(h2, training=training)
        return layers.Add()([x, h2])

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "d_model": self.d_model,
                "num_heads": self.num_heads,
                "mlp_dim": self.mlp_dim,
                "attn_drop": self.attn_drop,
                "proj_drop": self.proj_drop,
            }
        )
        return config
