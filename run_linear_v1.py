#%%
import os
import torch
import random
import numpy as np
from utils.tools import dotdict
from exp.exp_main import Exp_Main
import numpy as np
import matplotlib.pyplot as plt
#%%
# Set seeds for reproducibility
fix_seed = 2021
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)

# Create log directories if they don't exist
if not os.path.exists("./logs"):
    os.makedirs("./logs")
if not os.path.exists("./logs/LongForecasting"):
    os.makedirs("./logs/LongForecasting")
#%%
# Configuration parameters
seq_len = 96          # Input window size (~7.7 seconds at 12.5Hz)
pred_len = 12         # Prediction horizon (1 second)
n_features = 66       # Number of features in your dataset
model_name = "DLinear"

# Create args as dotdict
args = dotdict()

# Basic configuration
args.is_training = 1
args.model_id = f'MA_{seq_len}_{pred_len}'
args.model = model_name
args.train_only = False

# Data loader configuration
args.data = 'custom'
args.root_path = './DATA_MA'
args.data_path = 'normalized_data.csv'
args.features = 'M'
args.target = 'F66'
args.freq = 'ms'
args.checkpoints = './linear_checkpoints'

# Forecasting task
args.seq_len = seq_len
args.label_len = 48  # Not used for linear models
args.pred_len = pred_len

# Model parameters
args.individual = True
args.enc_in = n_features
args.dec_in = n_features
args.c_out = n_features
args.moving_avg = 5

# Training parameters
args.num_workers = 0
args.itr = 1
args.train_epochs = 10
args.batch_size = 32
args.patience = 3
args.learning_rate = 0.0001
args.loss = 'mse'
args.lradj = 'type1'
args.use_amp = False
args.des = 'Exp'

# GPU configuration
args.use_gpu = True if torch.cuda.is_available() else False
args.gpu = 0
args.use_multi_gpu = False

# Post-processing
log_file_name = f"{model_name}_I_MA_{seq_len}_{pred_len}.log"
#%%
# Redirect output to log file
with open(f"./logs/LongForecasting/{log_file_name}", 'w') as f:
    print('Args in experiment:')
    print(args)
    
    Exp = Exp_Main
    setting = '{}_{}_{}_ft{}_sl{}_pl{}_in{}_it{}_lr{}_bs{}'.format(
        args.model_id,
        args.model,
        args.data,
        args.features,
        args.seq_len,
        args.pred_len,
        args.individual,
        args.itr,
        args.learning_rate,
        args.batch_size
    )
    
    exp = Exp(args)
    print(f'>>>>>>> Start training: {setting} >>>>>>>>>>>>>>>>>>>>>>>>>>')
    exp.train(setting)
    
    if not args.train_only:
        print(f'>>>>>>> Testing: {setting} <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
        exp.test(setting)
    
    torch.cuda.empty_cache()

#%%
exp.predict(setting, True)
# Load data
preds = np.load('./results/'+setting+'/pred.npy')
trues = np.load('./results/'+setting+'/true.npy')
#%%
# draw prediction
plt.figure()
plt.plot(trues[0,:,5], label='GroundTruth')
plt.plot(preds[0,:,5], label='Prediction')
plt.legend()
plt.show()

