import config
import numpy as np
from scipy.interpolate import interp1d
import torch
from torch.utils.data import DataLoader, Dataset

class ProfileDataset(Dataset):
    def __init__(self, path, lr_dim, hr_dim, wethr_dim):
        super().__init__()
        self.dataset = np.load(path)
        self.lr_dim = lr_dim
        self.hr_dim = hr_dim
        self.wethr_dim = wethr_dim
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.x = np.linspace(0, hr_dim, num=lr_dim)
        self.x_new = np.linspace(0, hr_dim, num=hr_dim)

    def __getitem__(self, index):
        lr_prfl = np.expand_dims(self.dataset[index, 0:self.lr_dim], axis=0)
        lr_input = np.zeros(((1 + self.wethr_dim), self.lr_dim))
        for idx in range(1 + self.wethr_dim):
            if idx == 0:
                lr_input[idx, :] = self.dataset[index, idx * self.lr_dim:(idx + 1) * self.lr_dim]/config.P_MAX
            else:
                lr_input[idx, :] = self.dataset[index, idx * self.lr_dim:(idx + 1) * self.lr_dim]

        f = interp1d(self.x, lr_prfl, kind='cubic')
        hr_prfl_intp = f(self.x_new)

        hr_prfl_gt = self.dataset[index, (self.lr_dim * (1 + self.wethr_dim)):\
                                         (self.lr_dim * (1 + self.wethr_dim) + self.hr_dim)]

        lr_prfl = torch.from_numpy(lr_prfl).float().to(self.device)
        lr_input = torch.from_numpy(lr_input).float().to(self.device)
        hr_prfl_intp = torch.from_numpy(hr_prfl_intp).float().to(self.device)
        hr_prfl_gt = torch.from_numpy(hr_prfl_gt).float().to(self.device).unsqueeze(0)

        return lr_prfl, lr_input, hr_prfl_intp, hr_prfl_gt

    def __len__(self):
        return len(self.dataset)


trainset = ProfileDataset(config.train_set_path, lr_dim=config.DIM_LR,
                          hr_dim=config.DIM_HR, wethr_dim=config.WEATHER_DIM)
trainloader = DataLoader(trainset, batch_size=config.BATCH_SIZE, num_workers=0, shuffle=True)
testset = ProfileDataset(config.test_set_path, lr_dim=config.DIM_LR,
                         hr_dim=config.DIM_HR, wethr_dim=config.WEATHER_DIM)
testloader = DataLoader(testset, batch_size=config.BATCH_SIZE, num_workers=0, shuffle=True)
testset_eval = ProfileDataset(config.test_set_path, lr_dim=config.DIM_LR,
                              hr_dim=config.DIM_HR, wethr_dim=config.WEATHER_DIM)
testloader_eval = DataLoader(testset_eval, batch_size=1, num_workers=0, shuffle=False)
