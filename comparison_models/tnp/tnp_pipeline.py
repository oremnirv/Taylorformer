import tensorflow as tf
from tensorflow import keras
from comparison_models.tnp.tnp import TNP_Decoder
import sys
sys.path.append("../../")
from data_wrangler.feature_extractor import feature_wrapper


class tnp_pipeline(keras.models.Model):

    def __init__(self,num_heads=4,projection_shape_for_head=4,output_shape=64, dropout_rate=0.0, 
                 permutation_repeats=1,bound_std=False, num_layers=6,target_y_dim=1):
        super().__init__()
        self._permutation_repeats = permutation_repeats
        self._feature_wrapper = feature_wrapper()
        self._tnp = TNP_Decoder(output_shape=output_shape,num_layers=num_layers,projection_shape=int(projection_shape_for_head*num_heads),
                num_heads=num_heads,dropout_rate=dropout_rate,target_y_dim=target_y_dim,bound_std=bound_std)


    def call(self, inputs):

        x, y, n_C, n_T, training = inputs
        #x and y have shape batch size x length x dim

        x = x[:,:n_C+n_T,:]
        y = y[:,:n_C+n_T,:]

        if training == True:    
            x,y = self._feature_wrapper.permute([x, y, n_C, n_T, self._permutation_repeats]) 

        ######## make mask #######

        context_part = tf.concat([tf.ones((n_C,n_C),tf.bool),tf.zeros((n_C,2*n_T),tf.bool)],
                         axis=-1)
        first_part = tf.linalg.band_part(tf.ones((n_T,n_C+2*n_T),tf.bool),-1,n_C)
        second_part = tf.linalg.band_part(tf.ones((n_T,n_C+2*n_T),tf.bool),-1,n_C-1)
        mask = tf.concat([context_part,first_part,second_part],axis=0)
        
        ###### mask appropriate inputs ######

        batch_s = tf.shape(x)[0]

        context_target_pairs = tf.concat([x,y],axis=2)
        
        y_masked = tf.zeros((batch_s,n_T,y.shape[-1]))
        target_masked_pairs = tf.concat([x[:,n_C:],y_masked],axis=2)

        μ, log_σ  = self._tnp([context_target_pairs,target_masked_pairs,mask],training)
        return μ[:,-n_T:], log_σ[:, -n_T:]



            

def instantiate_tnp(dataset,training=True):
            

    if dataset == "exchange":

        return tnp_pipeline(num_heads=6,projection_shape_for_head=8,output_shape=48, dropout_rate=0.1, 
                 permutation_repeats=0,bound_std=False, num_layers=6,target_y_dim=1)

    if dataset == "ETT":

        return tnp_pipeline(num_heads=7,projection_shape_for_head=12,output_shape=48, dropout_rate=0.05, 
                 permutation_repeats=0,bound_std=False, num_layers=4,target_y_dim=1)