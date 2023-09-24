import torch

N_EPOCHS_GAN = 301  # number of epochs of training
N_EPOCHS_POL = 301  # number of epochs of training
SAVE_PER_EPO = 50  # save weights per n epochs

BATCH_SIZE = 32  # size of the batches
LR = 0.0001  # learning rate
B1 = 0.99  # adam: decay of first order momentum of gradient
B2 = 0.999  # adam: decay of first order momentum of gradient

P_MAX = 1  # max peak power, use to normalize
W_ADVS = 0.04  # weight of adversarial loss
W_CONT = 1.0  # weight of content loss
W_FEAT = 0.5  # weight of feature loss
W_SWIT = 1.0  # weight of switch loss
W_OUTL = 1.0  # weight of outline loss
W_ADVS_POL = 0.001  # weight of peak loss

DROPRATE = 0.4  # dropout rate of neural network
NF_GEN = 64  # feature number of generator
NF_DIS = 8  # feature number of discriminator
N_RES_GEN = 5  # number of residual block for generator
N_RES_POL = 10  # number of residual block for polish network
OUTL_WS = 2  # max pooling kernel window size
OUTL_ST = 1  # max pooling stride size
SWIT_WS = 4  # average pooling kernel window size
SWIT_ST = 2  # average pooling stride size

DIM_LR = 48  # dim of low resolution profile
DIM_HR = 288  # dim of high resolution profile

WEATHER_DIM = 5  # dim of weather data
INPUT_CH = 1 + WEATHER_DIM  # number of input channels
SAVE_CYCLE = 10  # epoch interval between saving models
TAG = ""  # task tag
LOADING = False  # load previous trained model

train_set_path = "../dataset/your_trainset.npy"
test_set_path = "../dataset/your_testset.npy"

CUDA = True if torch.cuda.is_available() else False
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
Tensor = torch.cuda.FloatTensor if CUDA else torch.FloatTensor
