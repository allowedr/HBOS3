import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import torch
if torch.cuda.is_available():
    device = torch.device("cuda")
    print('cuda')
else :
    device = torch.device("cpu")
    print('cpu')