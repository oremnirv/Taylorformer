from data_wrangler import dataset_preparer
from model import taylorformer_pipeline
import numpy as np
import tensorflow as tf
from data_wrangler.batcher import batcher
import argparse



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run",type=int,help = "run number", default=0)  # change this to the run number you want to use [0, 1, 2, 3, 4] 
    parser.add_argument("step",type=int,help = "step number", default=37)  # change this to the step number given in the checkpoint file end e.g. ckpt-37 is inside weights_/forecasting/ETT/taylorformer/96/ckpt/check_run_0

    args = parser.parse_args()

    n_C = 96
    n_T = 96
    model = 'taylorformer'
    x_train, y_train, x_val, y_val, x_test, y_test = dataset_preparer.dataset_processor(path_to_data="datasets/ETTm2.csv") 
    save_dir = "weights/forecasting/ETT"
    save_dir = save_dir + "/" + model + '/' + str(n_T)
    
    model = taylorformer_pipeline.instantiate_taylorformer('ETT')

    name_comp = 'run_' + str(args.run)
    folder = save_dir + '/ckpt/check_' + name_comp
    opt = tf.keras.optimizers.Adam(3e-4)
    
    
    ### LOAD THE MODEL
    ckpt = tf.train.Checkpoint(step=tf.Variable(args.step), optimizer=opt, net=model)
    manager = tf.train.CheckpointManager(ckpt, folder, max_to_keep=1)
    ckpt.restore(manager.latest_checkpoint) 

    ## Run pre-trained model on batch of test data 
    test_batch_s = 16
    idx_list = list(range(x_test.shape[0] - (n_C + n_T)))
    t_te, y_te, idx_list = batcher(x_test, y_test, idx_list, batch_s = test_batch_s, window=n_C+n_T)
    t_te = np.repeat(np.linspace(-1,1,(n_C+n_T))[np.newaxis,:,np.newaxis],axis=0,repeats=y_te.shape[0])
    μ, log_σ = model([t_te, y_te, n_C, n_T, False])
