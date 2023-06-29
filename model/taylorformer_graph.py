
import tensorflow as tf
from model import losses


def build_graph():
    
    @tf.function(experimental_relax_shapes=True)
    def train_step(taylorformer_model, optimizer, x, y, n_C, n_T, training=True):

        with tf.GradientTape(persistent=True) as tape:

            μ, log_σ = taylorformer_model([x, y, n_C, n_T, training]) 
            _, _, _, likpp, mse = losses.nll(y[:, n_C:n_T+n_C], μ, log_σ)
        
        gradients = tape.gradient(likpp, taylorformer_model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, taylorformer_model.trainable_variables))
        return μ, log_σ, likpp, mse

    tf.keras.backend.set_floatx('float32')
    return train_step
