import tensorflow as tf
from model import dot_prod


class FFN_1(tf.keras.layers.Layer):
    def __init__(self, output_shape, dropout_rate=0.1):
        super(FFN_1, self).__init__()
      
        self.dense_a = tf.keras.layers.Dense(output_shape)
        self.dense_b = tf.keras.layers.Dense(output_shape)
        self.dense_c = tf.keras.layers.Dense(output_shape)
        self.layernorm = [tf.keras.layers.LayerNormalization() for _ in range(2)]        
        self.dropout = tf.keras.layers.Dropout(dropout_rate)

    def call(self, x, query, training = True):
        ## call layer after first MHA_X
        ## x is the output of MHA_X_1
        ## query is query input to MHA_X_1 

        query = self.dense_a(query)
        x += query
        x = self.layernorm[0](x)
        x_skip = tf.identity(x)
        x = self.dense_b(x)
        x = tf.nn.gelu(x)
        x = self.dropout(x, training=training)
        x = self.dense_c(x)
        x += x_skip
        return self.layernorm[1](x)


class FFN_o(tf.keras.layers.Layer):
    def __init__(self, output_shape, dropout_rate=0.1):
        super(FFN_o, self).__init__()

        self.dense_b = tf.keras.layers.Dense(output_shape)
        self.dense_c = tf.keras.layers.Dense(output_shape)
        self.layernorm = [tf.keras.layers.LayerNormalization() for _ in range(2)]        
        self.dropout = tf.keras.layers.Dropout(dropout_rate)

    def call(self, x, query, training = True):

      ## query is the output of previous MHA_X layer
      ## x is query input to MHA_X_o 

        x += query
        x = self.layernorm[0](x)
        x_skip = tf.identity(x)
        x = self.dense_b(x)
        x = tf.nn.gelu(x)
        x = self.dropout(x, training=training)
        x = self.dense_c(x)
        x += x_skip
        return self.layernorm[1](x)

class MHA_X_a(tf.keras.layers.Layer):
    def __init__(self,
                  num_heads,
                  projection_shape,
                  output_shape,
                  dropout_rate=0.1):
        super(MHA_X_a, self).__init__()
        self.mha = dot_prod.MultiHeadAttention(num_heads, output_shape, projection_shape)
        self.ffn = FFN_1(output_shape, dropout_rate)

    def call(self, query, key, value, mask, training = True):
        x = self.mha(query, key, value, mask)
        x = self.ffn(x, query, training=training)  # Shape `(batch_size, seq_len, output_shape)`.
        return x

class MHA_XY_a(tf.keras.layers.Layer):
    def __init__(self,
                 num_heads,
                  projection_shape,
                  output_shape,
                  dropout_rate=0.1):
        super(MHA_XY_a, self).__init__()
        self.mha = dot_prod.MultiHeadAttention(num_heads, output_shape, projection_shape)
        self.ffn = FFN_1(output_shape, dropout_rate)

    def call(self, query, key, value, mask, training=True):
        x = self.mha(query, key, value, mask)
        x = self.ffn(x, query, training=training)  # Shape `(batch_size, seq_len, output_shape)`.

        return x


class MHA_X_b(tf.keras.layers.Layer):
    def __init__(self,
                  num_heads,
                  projection_shape,
                  output_shape,
                  dropout_rate=0.1):
        super(MHA_X_b, self).__init__()
        self.mha = dot_prod.MultiHeadAttention(num_heads, output_shape, projection_shape)
        self.ffn = FFN_o(output_shape, dropout_rate)

    def call(self, query, key, value, mask, training = True):
        x = self.mha(query, key, value, mask)
        x = self.ffn(x, query, training = training)  # Shape `(batch_size, seq_len, output_shape)`.

        return x


class MHA_XY_b(tf.keras.layers.Layer):
    def __init__(self,
                  num_heads,
                  projection_shape,
                  output_shape,
                  dropout_rate=0.1):
        super(MHA_XY_b, self).__init__()
        self.mha = dot_prod.MultiHeadAttention(num_heads, output_shape, projection_shape)
        self.ffn = FFN_o(output_shape, dropout_rate)

    def call(self, query, key, value, mask, training=True):
        x = self.mha(query, key, value, mask)
        x = self.ffn(x, query, training=training)  # Shape `(batch_size, seq_len, output_shape)`.

        return x


class taylorformer(tf.keras.Model):
    def __init__(self, num_heads,
                  projection_shape,
                  output_shape,
                  num_layers,
                  dropout_rate=0.1, target_y_dim=1,
                  bound_std = False
                  ):
        super().__init__()

        self.num_layers = num_layers
        
        self.mha_x_a = MHA_X_a(num_heads,
                  projection_shape,
                  output_shape,
                  dropout_rate=dropout_rate)
      
        self.mha_x_b = [MHA_X_b(num_heads,
                  projection_shape,
                  output_shape,
                  dropout_rate=dropout_rate) for _ in range(num_layers-1)]

        self.mha_xy_a = MHA_XY_a(num_heads,
                  projection_shape,
                  output_shape, dropout_rate=dropout_rate)
        
        self.mha_xy_b = [MHA_XY_b(num_heads,
                  projection_shape,
                  output_shape,
                  dropout_rate=dropout_rate) for _ in range(num_layers-1)]

        self.linear_layer = tf.keras.layers.Dense(output_shape)

        self.dense_sigma = tf.keras.layers.Dense(target_y_dim)
        self.dense_last = tf.keras.layers.Dense(target_y_dim)
        self.bound_std = bound_std

    def call(self, input, training=True):
        query_x, key_x, value_x, query_xy, key_xy, value_xy, mask, y_n = input

        x = self.mha_x_a(query_x,query_x, query_x, mask,training=training)
        xy = self.mha_xy_a(query_xy, key_xy, value_xy, mask,training=training)

        for i in range(self.num_layers - 2):

            xy  = self.mha_xy_b[i](xy, xy, xy, mask,training=training)
            x  = self.mha_x_b[i](x, x, x, mask,training=training)

        xy = self.mha_xy_b[-1](xy, xy, xy, mask,training=training)
        x = self.mha_x_b[-1](x, x, value_x, mask,training=training)

        combo = tf.concat([x,xy], axis = 2)
        z = self.linear_layer(combo)
        
        log_σ = self.dense_sigma(z)

        μ = self.dense_last(z) + y_n

        σ = tf.exp(log_σ)
        if self.bound_std:

            σ = 0.01 + 0.99 * tf.math.softplus(log_σ)

        log_σ = tf.math.log(σ)
    
        return μ, log_σ
