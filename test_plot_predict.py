#%%
import os
import torch
import random
import numpy as np
from utils.tools import dotdict
from exp.exp_main import Exp_Main
import matplotlib.pyplot as plt
import json
#%%
# Set seeds for reproducibility
fix_seed = 2020 #2021
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)
# Get arguments from function
get_args_func = False

# Create directories
for dir_path in ["./logs", "./logs/LongForecasting", "./results"]:
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


if get_args_func:

    # Model configurations
    #models = ['DLinear', 'NLinear', 'Linear','Autoformer', 'Transformer', 'Informer']
    models = [ 'DLinear']

    #seq_len_vals = [96, 360, 1200]
    seq_len_vals = [380]
    label_len = 96#25

    # Basic configurations
    pred_len = 12#12         # Prediction horizon (1 second)
    n_features = 1       # Number of features in dataset
else:
    # Read json for the arguments
    # Before loading , check the data_path in a text editor
    import json

    # Read JSON file back as dictionary
    with open('args.json', 'r') as f:
        args = json.load(f)
    args = dotdict(args)
    print("Arguments loaded from JSON file")


#%% Run function to get arguments
def get_args(model_name, seq_len, label_len, pred_len, n_features):
    args = dotdict()
    freq = 't'
    # Basic configuration
    args.is_training = 1
    args.model_id = f'PCA_MA2_{seq_len}_{pred_len}_{label_len}_{freq}'
    args.model = model_name
    args.train_only = False
    
    # Data loader configuration
    args.data = 'custom'
    args.root_path = './DATA_MA'
    args.data_path = 'normalized_dataF_PCA_s.csv'#'xNormPCA_s.csv' # 'normalized_dataPCA_s.csv'
    args.features = 'S'
    args.target = 'F1'
    args.freq = freq
    args.checkpoints = f'./{model_name.lower()}_checkpoints'
    args.scale = False
    
    # Forecasting task
    args.seq_len = seq_len
    args.label_len = label_len # 250#25
    args.pred_len = pred_len
    
    # Model parameters
    args.individual = True
    args.enc_in = n_features
    args.dec_in = n_features
    args.c_out = n_features
    args.moving_avg = 1024
    
    # Training parameters
    args.num_workers = 0
    args.itr = 1
    args.train_epochs = 1000
    args.batch_size = 32
    args.patience = 10
    args.learning_rate = 0.0001
    args.loss = 'mse'
    args.lradj = '5' #'type1' #3
    args.use_amp = False
    args.des = 'Exp'
    
    # GPU configuration
    args.use_gpu = True #if torch.cuda.is_available() else False
    args.gpu = 0
    args.use_multi_gpu = False
    args.devices = '0'
    args.test_flop = False
    
    # Additional common parameters
    args.dropout = 0.05
    args.embed = 'timeF'
    args.activation = 'gelu'
    args.output_attention = False
    args.do_predict = True
    
    # Model specific parameters
    #if model_name in ['Autoformer', 'Transformer', 'Informer']:
        # Parameters for transformer-based models
    args.embed_type = 3 # 0. '0: default 1: value embedding + temporal embedding + positional embedding 2: value embedding + temporal embedding 3: value embedding + positional embedding 4: value embedding'
    args.d_model = 512
    args.n_heads = 1 #8
    args.e_layers = 1 #2,3
    args.d_layers = 1 #1
    args.d_ff = 2048
    args.factor = 1
    args.distil = False

    #elif model_name in ['DLinear', 'NLinear', 'Linear']:
    # Parameters for linear models
    args.individual = True
    
    return args

#%% get EXP and setting
# Run experiments for each model and sequence length
for model_name in models:
    for seq_len in seq_len_vals:
        if get_args_func:
            # Get the args from function
            args = get_args(model_name, seq_len, label_len, pred_len, n_features)
        
        log_file_name = f"{model_name}_I_MA_{seq_len}_{pred_len}.log"
        
        # Run experiment with logging
        with open(f"./logs/LongForecasting/{log_file_name}", 'w') as f:
            print(f"\nStarting experiment with {model_name}, seq_len={seq_len}")
            print('Args in experiment:', args)
            
            Exp = Exp_Main
            setting = '{}_TEST-PREDICT_{}_{}_{}'.format(args.data_path, args.model_name, args.seq_len, args.pred_len)
            '''
            setting = '{}_ma5_{}_{}_ft{}_sl{}_pl{}_in{}_it{}_lr{}_bs{}_movag{}_lablen{}_freq{}'.format(
                args.model_id,
                args.model,
                args.data,
                args.features,
                args.seq_len,
                args.pred_len,
                args.individual,
                args.itr,
                args.learning_rate,
                args.batch_size,
                args.moving_avg,
                args.label_len,
                args.freq
            
            )
            '''
#%%            
exp = Exp(args)
#%%
#print(f'>>>>>>> Training: {setting} >>>>>>>>>>>>>>>>>>>>>>>>>>')
#exp.train(setting)

#if not args.train_only:
#    print(f'>>>>>>> Testing: {setting} <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
exp.test(setting, test=1)
#%%
# Prediction and visualization
exp.predict(setting, True)
preds = np.load(f'./results/{setting}/pred.npy')
trues = np.load(f'./results/{setting}/true.npy')
#%%
# Plot results
#plt.figure(figsize=(10, 6))
import matplotlib.pyplot as plt
plt.plot(trues[0,:,0], label='GroundTruth')
plt.plot(preds[0,:,0], label='Prediction')
plt.title(f'{model_name} - Sequence Length: {seq_len}')
plt.legend()
#plt.savefig(f'./results/{setting}/prediction_plot.png')
plt.show()
#plt.close()
            
torch.cuda.empty_cache()
