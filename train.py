import os
import time
from torch.autograd import Variable
import torch
import config

from dataset import trainloader
from model import PolishLoss, Disc, Gen5min_POL_norm, Gen60to5_SRGAN_norm, Gen30to5_SRGAN_norm, Gen15to5_SRGAN_norm


DIR = "../plot/" + config.TAG
if not os.path.isdir(DIR):
    os.makedirs(DIR)
DIR = "../checkpoint/" + config.TAG
if not os.path.isdir(DIR):
    os.makedirs(DIR)

advs_loss = torch.nn.BCELoss()
cont_loss = torch.nn.MSELoss()

# Initialize generator and disc
if config.DIM_LR == 24:
    G_gan = Gen60to5_SRGAN_norm(in_channels=config.INPUT_CH,
                                num_channels=config.NF_GEN,
                                num_blocks=config.N_RES_GEN,
                                name='G_GAN_60to5')
    G_pol = Gen5min_POL_norm(in_channels=1,
                             num_channels=config.NF_GEN,
                             num_blocks=config.N_RES_POL,
                             name='G_pol_5min')
    disc = Disc(num_fea=config.NF_DIS, name='Disc_5min')
elif config.DIM_LR == 48:
    G_gan = Gen30to5_SRGAN_norm(in_channels=config.INPUT_CH,
                                num_channels=config.NF_GEN,
                                num_blocks=config.N_RES_GEN,
                                name='G_GAN_30to5')
    G_pol = Gen5min_POL_norm(in_channels=1,
                             num_channels=config.NF_GEN,
                             num_blocks=config.N_RES_POL,
                             name='G_pol_5min')
elif config.DIM_LR == 96:
    G_gan = Gen15to5_SRGAN_norm(in_channels=config.INPUT_CH,
                                num_channels=config.NF_GEN,
                                num_blocks=config.N_RES_GEN,
                                name='G_GAN_15to5')
    G_pol = Gen5min_POL_norm(in_channels=1,
                             num_channels=config.NF_GEN,
                             num_blocks=config.N_RES_POL,
                             name='G_pol_5min')
disc = Disc(num_fea=config.NF_DIS, name='Disc_5min')

if config.CUDA:
    G_gan.cuda()
    G_pol.cuda()
    disc.cuda()
    advs_loss.cuda()
    cont_loss.cuda()

# optimizers
optimizer_G_gan = torch.optim.Adam(G_gan.parameters(), lr=config.LR, betas=(config.B1, config.B2))
optimizer_G_pol = torch.optim.Adam(G_pol.parameters(), lr=config.LR, betas=(0.9, 0.999))
optimizer_D = torch.optim.Adam(disc.parameters(), lr=config.LR, betas=(config.B1, config.B2))

# plot helper

polish_loss = PolishLoss()
# ------------------------------------Training------------------------------------
start_t = time.time()
for epoch in range(config.N_EPOCHS_GAN + config.N_EPOCHS_POL):

    for i, data in enumerate(trainloader):
        lr_prfl, lr_input, hr_prfl_intp, real_hr = data
        bs = real_hr.size(0)

        real_tag = Variable(config.Tensor(bs, 1).fill_(1.0), requires_grad=False)
        fake_tag = Variable(config.Tensor(bs, 1).fill_(0.0), requires_grad=False)

        # Configure input
        real_hr = Variable(real_hr.type(config.Tensor))
        disc_input_real = disc.build_input(real_hr, bs)
        if epoch < config.N_EPOCHS_GAN:
            G_gan.train()
            disc.train()
            
            # train generator
            optimizer_G_gan.zero_grad()
            fake_hr_gan = G_gan(lr_input)
            disc_input_gan = disc.build_input(fake_hr_gan, bs)
            score_fake_gan, fea_fake = disc(disc_input_gan)
            score_real, fea_real = disc(disc_input_real)
            loss_advs_gan = advs_loss(score_fake_gan, real_tag) * config.W_ADVS
            loss_cont_gan = cont_loss(fake_hr_gan, real_hr) * config.W_CONT
            loss_feat_gan = cont_loss(fea_fake, fea_real) * config.W_FEAT
            loss_G_gan = loss_advs_gan + loss_cont_gan + loss_feat_gan
            loss_G_gan.backward()
            optimizer_G_gan.step()

            # train disc
            optimizer_D.zero_grad()
            score_real, _ = disc(disc_input_real)
            score_fake_gan, _ = disc(disc_input_gan.detach())
            loss_real = advs_loss(score_real, real_tag)
            loss_fake = advs_loss(score_fake_gan, fake_tag)
            loss_D = (loss_real + loss_fake) / 2
            loss_D.backward()
            optimizer_D.step()

            loss_swit_pol = torch.zeros_like(loss_D)
            loss_outl_pol = torch.zeros_like(loss_D)
            loss_G_pol = torch.zeros_like(loss_D)

        else:
            G_gan.eval()
            G_pol.train()

            score_real, fea_real = disc(disc_input_real)

            # train generator
            G_pol.zero_grad()
            fake_hr_gan = G_gan(lr_input)
            fake_hr_pol = G_pol(fake_hr_gan)  # + fake_hr_gan
            disc_input_pol = disc.build_input(fake_hr_pol, bs)
            score_fake_pol, fea_fake_pol = disc(disc_input_pol)
            loss_cont_gan = cont_loss(fake_hr_pol, real_hr) * config.W_CONT
            loss_advs_gan = advs_loss(score_fake_pol, real_tag) * config.W_ADVS
            loss_outl_pol, loss_swit_pol = polish_loss.cal_loss(fake_hr_pol, real_hr)
            loss_G_pol = loss_outl_pol + loss_swit_pol + \
                         advs_loss(score_fake_pol, real_tag) * config.W_ADVS_POL
            loss_G_pol.backward()
            optimizer_G_pol.step()


            score_fake_gan, fea_fake = disc(disc_input_pol.detach())
            loss_feat_gan = cont_loss(fea_fake, fea_real) * config.W_FEAT
            loss_real = advs_loss(score_real, real_tag) * config.W_ADVS
            loss_fake = advs_loss(score_fake_gan, fake_tag) * config.W_ADVS
            loss_D_pol = (loss_real + loss_fake) / 2
            loss_D = loss_D_pol

    msg_tmp = "---epoch:{}".format(epoch)
    msg_tmp += " || Time(s):{}".format(int(time.time() - start_t))
    print(msg_tmp)

    if epoch % config.SAVE_CYCLE == 0:
        if epoch < config.N_EPOCHS_GAN:
            G_gan.save_checkpoint(epoch)
            disc.save_checkpoint(epoch)
        else:
            G_pol.save_checkpoint(epoch - config.N_EPOCHS_GAN)
