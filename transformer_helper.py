import tensorflow as tf
import numpy as np
from os import environ
environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


def get_angles(pos, k, d: int):
    """
    Get angles to be used in the positional encoding vectors

    Arguments:
        pos -- Column vector containing the positions [[0], [1], ...,[N-1]]
        k --   Row vector containing the dimension span [[0, 1, 2, ..., d-1]]
        d -- Encoding size

    Returns:
        angles -- (pos, d) np.array
    """
    # Get i from dimension span k
    i = k // 2
    # Calculate the angles using pos, i and d
    angles = pos / (10000 ** (2 * i / d))
    # END CODE HERE

    return angles


def positional_encoding(positions: int, d: int):
    """
    Precomputes a matrix with all the positional encodings

    Arguments:
        positions - Maximum number of positions to be encoded
        d - Encoding size

    Returns:
        pos_encoding - (1, position, d_model) matrix with the positional encodings
    """
    angle_rads = get_angles(np.arange(positions)[:, np.newaxis],
                            np.arange(d)[np.newaxis, :],
                            d)

    # apply sin to even indices 2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

    # apply cos to odd indices 2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    pos_encoding = angle_rads[np.newaxis, :, :].reshape(1, positions, d)

    # casts tensor to float dtype
    return tf.cast(pos_encoding, dtype=tf.float32)

#####
# Two types of masks that are useful when building your Transformer network: the *padding mask* and the *look-ahead mask*
#####


def create_padding_mask(decoder_token_ids):
    """
    Creates a matrix mask for the padding cells
    replaces zeros with negative infinities to not cause problems when applying softmax

    Arguments:
        decoder_token_ids -- (n, m) matrix

    Returns:
        mask -- (n, 1, m) binary tensor
    """
    seq = 1 - tf.cast(tf.math.equal(decoder_token_ids, 0), tf.float32)
    # tf.math.equal(x,y) Returns the truth value of (x == y) element-wise.
    # True has numerical value 1, False 0 so all the zeros turn into 1 here, and everything else becomes 0

    # add extra dimensions to add the padding to the attention logits.

    return seq[:, tf.newaxis, :]


def create_look_ahead_mask(sequence_length):
    """
    Returns an upper triangular matrix filled with ones.
    Lets the training model check if it got predictions right by having access to the actual output

    Arguments:
        sequence_length -- matrix size (sequence length is the number of time steps per input
                           input.shape = [batch_size, sequence_length, num_features])

    Returns:
        mask -- (size, size) tensor

    >>>create_look_ahead_mask(5)
    <tf.Tensor: shape=(1, 5, 5), dtype=float32, numpy=
    array([[[1., 0., 0., 0., 0.],
            [1., 1., 0., 0., 0.],
            [1., 1., 1., 0., 0.],
            [1., 1., 1., 1., 0.],
            [1., 1., 1., 1., 1.]]], dtype=float32)>
    """
    mask = tf.linalg.band_part(
        tf.ones((1, sequence_length, sequence_length)), -1, 0)
    return mask


def scaled_dot_product_attention(q, k, v, mask=None):
    """
    Calculate the attention weights.
      q, k, v must have matching leading dimensions.
      k, v must have matching penultimate dimension, i.e.: seq_len_k = seq_len_v.
      The mask has different shapes depending on its type(padding or look ahead)
      but it must be broadcastable for addition.

    Arguments:
        q -- query shape == (..., seq_len_q, depth)
        k -- key shape == (..., seq_len_k, depth)
        v -- value shape == (..., seq_len_v, depth_v)
        mask: Float tensor with shape broadcastable
              to (..., seq_len_q, seq_len_k). Defaults to None.

    Returns:
        output -- attention_weights
    """
    assert k.shape[-2] == v.shape[-2]
    matmul_qk = tf.matmul(q, k.T)  # (..., seq_len_q, seq_len_k)

    # scale matmul_qk
    dk = k.shape[-2]
    scaled_attention_logits = matmul_qk / np.power(dk, 0.5)

    if mask is not None:
        scaled_attention_logits += (1. - mask) * -1e9

    # softmax is normalized on the last axis (seq_len_k) so that the scores
    # add up to 1.
    attention_weights = tf.exp(scaled_attention_logits) / np.sum(np.exp(scaled_attention_logits),
                                                                 axis=-1,
                                                                 keepdims=True)  # (..., seq_len_q, seq_len_k)

    output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)

    return output, attention_weights
