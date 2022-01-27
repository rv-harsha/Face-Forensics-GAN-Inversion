# imports are always needed
import torch

# get index of currently selected device
torch.cuda.current_device() # returns 0 in my case

# get number of GPUs available
torch.cuda.device_count() # returns 1 in my case

# get the name of the device
torch.cuda.get_device_name() # good old Tesla K80

# setting device on GPU if available, else CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Current device:', torch.cuda.device_count(), flush=True)
print('Using device:', device, flush=True)
print('Device count', torch.cuda.device_count(), flush=True)

#Additional Info when using cuda
if device.type == 'cuda':
    print(torch.cuda.get_device_name(), flush=True)
    print('\nMemory Usage:\n', flush=True)
    print('Allocated:', round(torch.cuda.memory_allocated()/1024**3,1), 'GB', flush=True)
    print('Cached:   ', round(torch.cuda.memory_reserved()/1024**3,1), 'GB', flush=True)

print(torch.cuda.memory_summary(device=device, abbreviated=True), flush=True)

import os.path as osp

outdir = "/nas/vista-hdd01/users/raidurga/outputs/ai2ai_run_004/expts/batch-001"
sheet = osp.join(outdir, "master-report-new.csv")
reg_noise_wgt = outdir.rsplit("/", 2)[0]

print(reg_noise_wgt)
