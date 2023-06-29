import tensorflow as tf
import sys
sys.path.append("../../")
from model import dot_prod


class FFN(tf.keras.layers.Layer):
    def __init__(self, output_shape, dropout_rate=0.1):
        super().__init__()

        self.dense_b = tf.keras.layers.Dense(output_shape)
        self.dense_c = tf.keras.layers.Dense(output_shape)
        self.layernorm = [tf.keras.layers.LayerNormalization() for _ in range(2)]        
        self.dropout = tf.keras.layers.Dropout(dropout_rate)

    def call(self, x, query):

      ## query is the output of previous MHA_X layer
      ## x is query input to MHA_X_o 

        x += query
        x = self.layernorm[0](x)
        x_skip = tf.identity(x)
        x = self.dense_b(x)
        x = tf.nn.gelu(x)
        x = self.dropout(x)
        x = self.dense_c(x)
        x += x_skip
        return self.layernorm[1](x)

class MHA_XY(tf.keras.layers.Layer):
    def __init__(self,
                 num_heads,
                  projection_shape,
                  output_shape,
                  dropout_rate=0.1):
        super().__init__()
        self.mha = dot_prod.MultiHeadAttention(num_heads, output_shape, projection_shape)
        self.ffn = FFN(output_shape, dropout_rate)

    def call(self, query, key, value, mask):
        x = self.mha(query, key, value, mask)
        x = self.ffn(x, query)  # Shape `(batch_size, seq_len, output_shape)`.
        return x
    

class embed_layers(tf.keras.layers.Layer):
    def __init__(self,output_shape,num_layers_embed=4):
        super().__init__()
        self.num_layers = num_layers_embed
        self.embed = [tf.keras.layers.Dense(output_shape,activation="relu") for _ in range(self.num_layers-1)]
        self.embed.append(tf.keras.layers.Dense(output_shape))

    def call(self,inputs):
        x = inputs
        for i in range(self.num_layers):
            x = self.embed[i](x)
        return x

class TNP_Decoder(tf.keras.models.Model):
    def __init__(self,output_shape=64,num_layers=6,projection_shape=16,
                 num_heads=4,dropout_rate=0.0,target_y_dim=1,bound_std=False):
        super().__init__()

        self.num_layers = num_layers

        self.mha_xy = [MHA_XY(num_heads,projection_shape,
                              output_shape,dropout_rate) for _ in range(num_layers)]

        self.embed = embed_layers(output_shape,num_layers_embed=4)

        self.dense = tf.keras.layers.Dense(output_shape,activation="relu")
        self.linear = tf.keras.layers.Dense(2*target_y_dim)
        self.target_y_dim = target_y_dim
        self.bound_std = bound_std
        
    def call(self,inputs,training=True):
        
        context_target_pairs,target_masked_pairs,mask = inputs
        input_for_mha = tf.concat([context_target_pairs,target_masked_pairs],axis=1)

        embed = self.embed(input_for_mha)
        
        v = embed
        k = tf.identity(v)
        q = tf.identity(v)

        for i in range(self.num_layers):
            x = self.mha_xy[i](q,k,v,mask)
            q = tf.identity(x)
            k = tf.identity(x)
            v = tf.identity(x)
      
        L = self.dense(x)
        L = self.linear(L)

        mean,log_sigma = L[:,:,:self.target_y_dim],L[:,:,self.target_y_dim:]

        if self.bound_std:
            sigma = 0.05 + 0.95 * tf.math.softplus(log_sigma)
        else:
            sigma = tf.exp(log_sigma)
        
        log_sigma = tf.math.log(sigma)
        return mean,log_sigma          