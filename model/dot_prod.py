
import tensorflow as tf



class DotProductAttention(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    def call(self, queries, keys, values, d_k, mask=None):
        scores = tf.matmul(queries, keys, transpose_b=True) / tf.sqrt(tf.cast(d_k, tf.float32))        
        if mask is not None:
            inverse_mask = (mask == False)
            scores += -1e9 * tf.cast(inverse_mask,tf.float32)
        weights = tf.keras.backend.softmax(scores)
        
        #below sets to zero if mask had a row of zeros (softmax would give data leakage)
        if mask is not None:
            weights = tf.math.minimum(tf.math.abs(tf.cast(mask,tf.float32)),tf.math.abs(weights))
        
        return tf.matmul(weights, values)


class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, num_heads, output_shape, projection_shape):
        super().__init__()
        self.attention = DotProductAttention()  # Scaled dot product attention
        self.heads = num_heads  # Number of attention heads to use
        self.projection_shape = projection_shape  # Dimensionality of the linearly projected queries, keys and values
        self.W_q = tf.keras.layers.Dense(projection_shape)  # Learned projection matrix for the queries
        self.W_k = tf.keras.layers.Dense(projection_shape)  # Learned projection matrix for the keys
        self.W_v = tf.keras.layers.Dense(projection_shape)  # Learned projection matrix for the values
        self.W_o = tf.keras.layers.Dense(output_shape)  # Learned projection matrix for the multi-head output
        assert projection_shape % self.heads == 0

        #heads must be a factor of projection_shape

    def reshape_tensor(self, x, heads, flag):
        if flag:
            # Tensor shape after reshaping and transposing: (batch_size, seq_length, heads,-1)
            x = tf.reshape(x, shape=(tf.shape(x)[0], tf.shape(x)[1], heads, -1))
            x = tf.transpose(x, perm=(0, 2, 1, 3))
        else:
            # Reverting the reshaping and transposing operations: (batch_size, seq_length, projection_shape)
            x = tf.transpose(x, perm=(0, 2, 1, 3))
            x = tf.reshape(x, shape=(tf.shape(x)[0], tf.shape(x)[1], self.projection_shape))
        return x

    def call(self, queries, keys, values, mask=None):
        # Rearrange the queries to be able to compute all heads in parallel
        q_reshaped = self.reshape_tensor(self.W_q(queries), self.heads, True)
        # Resulting tensor shape: (batch_size, heads, input_seq_length, -1)

        # Rearrange the keys to be able to compute all heads in parallel
        k_reshaped = self.reshape_tensor(self.W_k(keys), self.heads, True)
        # Resulting tensor shape: (batch_size, heads, input_seq_length, -1)

        # Rearrange the values to be able to compute all heads in parallel
        v_reshaped = self.reshape_tensor(self.W_v(values), self.heads, True)
        # Resulting tensor shape: (batch_size, heads, input_seq_length, -1)

        # Compute the multi-head attention output using the reshaped queries, keys and values
        o_reshaped = self.attention(q_reshaped, k_reshaped, v_reshaped, self.projection_shape, mask)
        # Resulting tensor shape: (batch_size, heads, input_seq_length, -1)

        # Rearrange back the output into concatenated form
        output = self.reshape_tensor(o_reshaped, self.heads, False)
        # Resulting tensor shape: (batch_size, input_seq_length, d_v)

        # Apply one final linear projection to the output to generate the multi-head attention
        # Resulting tensor shape: (batch_size, input_seq_length, d_model)
        return self.W_o(output)


