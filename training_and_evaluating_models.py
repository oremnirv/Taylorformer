#!/usr/bin/env python

from model import taylorformer_graph, losses
from data_wrangler import feature_extractor
import keras
import numpy as np
import tensorflow as tf
from model import taylorformer_pipeline
from comparison_models.tnp import tnp_pipeline
from data_wrangler import dataset_preparer
import argparse
from data_wrangler.batcher import batcher, batcher_np
import os

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("dataset", type=str, help="dataset")
    parser.add_argument("model", type=str, help="model")
    parser.add_argument("iterations", type=int, help="number of iterations for training")
    parser.add_argument("num_repeats", type=int, help="number of random seed repeats")
    parser.add_argument("n_C",type=int,help = "context")
    parser.add_argument("n_T",type=int,help = "target")
    parser.add_argument("run",type=int,help = "run number")
    

    args = parser.parse_args()


    if args.dataset == "exchange":
        x_train, y_train, x_val, y_val, x_test, y_test = dataset_preparer.dataset_processor(path_to_data="datasets/exchange.csv") 
        save_dir = "weights/forecasting/exchange"
        print('make sure to create the exchange folder in weights/forecasting/')
    
    elif args.dataset == "ETT":
        x_train, y_train, x_val, y_val, x_test, y_test = dataset_preparer.dataset_processor(path_to_data="datasets/ETTm2.csv") 
        save_dir = "weights/forecasting/ETT"
        print('make sure to create the ETT folder in weights/forecasting/')

    elif args.dataset == "rbf":
        x_train, y_train, x_val, y_val, x_test, y_test,n_context_test = dataset_preparer.gp_data_processor(path_to_data_folder="datasets/rbf/")
        save_dir = "weights/gp/rbf"
        print('make sure to create the gp folder in weights/gp')

    elif args.dataset == "electricity":
        x_train, y_train, x_val, y_val, x_test, y_test = dataset_preparer.electricity_processor(path_to_data="datasets/electricity_training.npy")
        save_dir = "weights/forecasting/electricity"
        print('make sure to create the electricity folder in weights/forecasting/')

    else: 
        raise ValueError("Dataset not found")
    

    if args.dataset == "rbf":
        save_dir = save_dir + '/' + args.model

    else:
        #### for forecasting inlude the n_T in the save directory
        save_dir = save_dir + "/" + args.model + '/' + str(args.n_T)

    n_C = args.n_C
    n_T = args.n_T

    batch_size = 32
    test_batch_s = 100
    valid_batch_size = 100
    if args.dataset != "rbf":
        if n_T > 700 :
            batch_size = 16
            test_batch_s = 16
            valid_batch_size = 16

    

    nll_list = []
    mse_list = []

    for repeat in range(args.num_repeats):

        step = 1
        run= args.run + repeat
        tf.random.set_seed(run)

        if args.model == "taylorformer":
            model = taylorformer_pipeline.instantiate_taylorformer(args.dataset)
        
        if args.model == "tnp":
            model = tnp_pipeline.instantiate_tnp(args.dataset)

    
        tr_step = taylorformer_graph.build_graph()
        
        ###### can we put the name of the model into the folder name #########?

        name_comp = 'run_' + str(run)
        folder = save_dir + '/ckpt/check_' + name_comp
        if not os.path.exists(folder): os.mkdir(folder)
        opt = tf.keras.optimizers.Adam(3e-4)
        ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=opt, net=model)
        manager = tf.train.CheckpointManager(ckpt, folder, max_to_keep=1)
        ckpt.restore(manager.latest_checkpoint) 
        sum_mse_tot = 0; sum_nll_tot = 0
        mini = 50000

        validation_losses = []

        for i in range(args.iterations):

            if args.dataset != "rbf":
                idx_list = list(range(x_train.shape[0] - (n_C+n_T)))
                x,y,_ = batcher(x_train,y_train,idx_list,window=n_C+n_T,batch_s=batch_size) ####### generalise for not just forecasting
                x = np.repeat(np.linspace(-1,1,(n_C+n_T))[np.newaxis,:, np.newaxis], axis=0, repeats=batch_size) # it doesnt matter what the time is, just the relation between the times.
                #### edit batcher to fix this

            #### batcher for NP data (which is already in sequences)
            if args.dataset == "rbf":
                x,y = batcher_np(x_train,y_train,batch_s=batch_size)
                #### nc nT need specification
                n_C = tf.constant(int(np.random.choice(np.linspace(3,99,97))))
                n_T = 100 - n_C

            _,_, _, _ = tr_step(model, opt, x,y,n_C,n_T, training=True)

            if i % 100 == 0:

                if args.dataset != "rbf":
                    idx_list = list(range(x_val.shape[0] - (n_C+n_T)))
                    ##### if val set is empty - increase val set size.  ########
                    t_te,y_te,_ = batcher(x_val,y_val,idx_list,batch_s = valid_batch_size,window=n_C+n_T)
                    t_te = np.repeat(np.linspace(-1,1,(n_C+n_T))[np.newaxis,:,np.newaxis],axis=0,repeats=valid_batch_size)

                if args.dataset == "rbf":
                    t_te,y_te = batcher_np(x_val,y_val,batch_s=valid_batch_size)
                    ####nc nt need to be specified without interfering with the nc nt above
                    n_C = 20
                    n_T = 80
                μ, log_σ = model([t_te, y_te, n_C, n_T, False])
                _,_,_, nll_pp_te, msex_te = losses.nll(y_te[:, n_C:n_C+n_T], μ, log_σ)

                validation_losses.append(nll_pp_te)

                np.save(folder + "/validation_losses_iteration",np.array(validation_losses))

                if nll_pp_te < mini:
                    mini = nll_pp_te
                    manager.save()
                    step += 1
                    ckpt.step.assign_add(1)

        ###################################################################
        ########### Evaluation code #######################################
        ###################################################################


        ckpt = tf.train.Checkpoint(step=tf.Variable(step), optimizer=opt, net=model)
        manager = tf.train.CheckpointManager(ckpt, folder, max_to_keep=1)
        ckpt.restore(manager.latest_checkpoint) 

        if args.dataset != "rbf":
   
            test_batch_s = 100 #need to specify this as it gets changed in the loop below
            if n_T > 700 :
                test_batch_s = 16
            idx_list = list(range(x_test.shape[0] - (n_C+n_T)))
            num_batches = len(idx_list)//test_batch_s

            for _ in range(num_batches): #### specify correct number of batches for the batcher #####
                if(_ == (num_batches-1)): test_batch_s = len(idx_list)        
                t_te,y_te,idx_list = batcher(x_test, y_test, idx_list,batch_s = test_batch_s, window=n_C+n_T)
                t_te = np.repeat(np.linspace(-1,1,(n_C+n_T))[np.newaxis,:,np.newaxis],axis=0,repeats=y_te.shape[0])
                μ, log_σ = model([t_te, y_te, n_C, n_T, False])
                _, sum_mse, sum_nll, _, _ = losses.nll(y_te[:, n_C:n_C+n_T], μ, log_σ)
                sum_nll_tot += sum_nll / n_T
                sum_mse_tot += sum_mse / n_T

            nllx =  sum_nll_tot / (test_batch_s * x_test.shape[0]//test_batch_s)
            msex =  sum_mse_tot / (test_batch_s * x_test.shape[0]//test_batch_s)


        if args.dataset == "rbf":

            for i in range(x_test.shape[0]):
                n_C = tf.constant(n_context_test[i],"int32")
                n_T = 100 - n_C
                μ, log_σ = model([x_test[i:i+1], y_test[i:i+1], n_C, n_T, False])
                _, sum_mse, sum_nll, _, _ = losses.nll(y_test[i:i+1, n_C:n_C+n_T], μ, log_σ)
                sum_nll_tot += sum_nll / n_T
                sum_mse_tot += sum_mse / n_T

            nllx =  sum_nll_tot / x_test.shape[0]
            msex =  sum_mse_tot / x_test.shape[0]



        nll_list.append(nllx.numpy())
        mse_list.append(msex.numpy())
                
            
        np.save(save_dir + '/nll_list.npy', nll_list)    
        np.save(save_dir + '/mse_list.npy', mse_list)  


