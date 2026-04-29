import numpy as np

def batcher(t, y, idx_list, batch_s = 32, window = 288):
    '''
    cutting one long array to sequences of length 'window'.
    'batch_s' must be ≤ full array - window length

    input to forecast: (None, 1, 1) for t,y.
    input to NP tasks: (None, seq_len, 1) for t,y. window = 1.
    idx_list: list of indices, must be ≤ full array - window length.
    '''
    
    if len(idx_list) < 1:
        print("warning- you didn't loop over the correct range")
        
    
    batch_s = min(batch_s, y.shape[0]-window)    
    idx = np.random.choice(len(idx_list), batch_s, replace = False)

    y = np.array([np.array(y)[idx_list[i]:idx_list[i]+window, :, :] for i in idx])
    t = np.array([np.array(t)[idx_list[i]:idx_list[i]+window, :, :] for i in idx])
    for i in sorted(idx, reverse=True): del idx_list[i]
        
    t = t.squeeze()
    y = y.squeeze()
    
    if len(t.shape) == 2:
        t = t[:,:,np.newaxis]
        y = y[:,:,np.newaxis]
        
    return t,y, idx_list

def batcher_np(t,y,batch_s=32):

    idx = np.random.choice(y.shape[0], batch_s, replace = False)

    y = y[idx, :, :]
    t = t[idx, :, :]

    return t,y