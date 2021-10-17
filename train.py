import os
from Discriminator import Discriminator
from Generator import GeneratorResNet
from Data import ImageDataset
import torch.nn as nn
import torch
import itertools
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
import matplotlib.pyplot as plt


# define Loss
criterion_GAN = nn.MSELoss()
criterion_cycle = nn.L1Loss()
criterion_identity = nn.L1Loss()

# initialize G and D
G_AB = GeneratorResNet(3, num_residual_blocks=9)
D_B = Discriminator(3)

G_BA = GeneratorResNet(3, num_residual_blocks=9)
D_A = Discriminator(3)

# use gpu
os.environ["CUDA_VISIBLE_DEVICES"] = '6'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
G_AB.to(device)
D_B = D_B.to(device)
G_BA = G_BA.to(device)
D_A = D_A.to(device)

criterion_GAN = criterion_GAN.to(device)
criterion_cycle = criterion_cycle.to(device)
criterion_identity = criterion_identity.to(device)

# 初始化优化器
lr = 0.0002
b1 = 0.5
b2 = 0.999

optimizer_G = torch.optim.Adam(
    itertools.chain(G_AB.parameters(), G_BA.parameters()), lr=lr, betas=(b1, b2)
)

optimizer_D_A = torch.optim.Adam(
    D_A.parameters(), lr=lr, betas=(b1, b2)
)

optimizer_D_B = torch.optim.Adam(
    D_B.parameters(), lr=lr, betas=(b1, b2)
)

# 对学习率进行调整
n_epoches = 100
decay_epoch = 20

lambda_func = lambda epoch: 1 - max(0, epoch-decay_epoch)/(n_epoches-decay_epoch)

lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(optimizer_G, lr_lambda=lambda_func)
lr_scheduler_D_A = torch.optim.lr_scheduler.LambdaLR(optimizer_D_A, lr_lambda=lambda_func)
lr_scheduler_D_B = torch.optim.lr_scheduler.LambdaLR(optimizer_D_B, lr_lambda=lambda_func)

# 定义dataLoader 导入数据
data_dir = './datasets/'
batch_size = 5

transforms_ = transforms.Compose([
    # transforms.Resize(int(256*1.12), Image.BICUBIC),
    # #transforms.RandomCrop(256, 256),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),   # 把数据转为Tensor,并Norm到（0， 1）
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))   # 处理之后的图片数值范围在[-1, 1]
])

trainloader = DataLoader(
    ImageDataset(data_dir, mode='train', transforms=transforms_),
    batch_size=batch_size,
    shuffle=True,
    num_workers=0
)

testloader = DataLoader(
    ImageDataset(data_dir, mode='test', transforms=transforms_),
    batch_size=batch_size,
    shuffle=False,
    num_workers=0
)


def sample_images(real_A, real_B, figside=1.5):
    assert real_A.size() == real_B.size(), 'The image size for two domains must be the same'

    G_AB.eval()
    G_BA.eval()

    real_A = real_A.to(device)
    fake_B = G_AB(real_A).detach()
    real_B = real_B.to(device)
    fake_A = G_BA(real_B).detach()

    nrows = real_A.size(0)
    real_A = make_grid(real_A, nrow=nrows, normalize=True)
    fake_B = make_grid(fake_B, nrow=nrows, normalize=True)
    real_B = make_grid(real_B, nrow=nrows, normalize=True)
    fake_A = make_grid(fake_A, nrow=nrows, normalize=True)

    image_grid = torch.cat((real_A, fake_B, real_B, fake_A), 1).cpu().permute(1, 2, 0)

    plt.figure(figsize=(figside * nrows, figside * 4))
    plt.imshow(image_grid)
    plt.axis('off')
    plt.show()


for epoch in range(n_epoches):
    for i, (real_A, real_B) in enumerate(trainloader):
        out_shape = [real_A.size(0), 1, real_A.size(2) // D_A.scale_factor, real_A.size(3) // D_A.scale_factor]
        real_A, real_B = real_A.to(device), real_B.to(device)
        valid = torch.ones(out_shape).to(device)
        fake = torch.zeros(out_shape).to(device)

        '''train Generator'''
        # set to training mode in the begining, beacause sample_images will set it to eval mode
        G_AB.train()
        G_BA.train()

        optimizer_G.zero_grad()

        fake_B = G_AB(real_A)
        fake_A = G_BA(real_B)

        # identity loss
        loss_id_A = criterion_identity(fake_B, real_A)
        loss_id_B = criterion_identity(fake_A, real_B)
        loss_identity = (loss_id_A + loss_id_B) / 2

        # GAN loss, train G to make D think it's true
        loss_GAN_AB = criterion_GAN(D_B(fake_B), valid)
        loss_GAN_BA = criterion_GAN(D_A(fake_A), valid)
        loss_GAN = (loss_GAN_AB + loss_GAN_BA) / 2

        # cycle loss
        recov_A = G_BA(fake_B)
        recov_B = G_AB(fake_A)
        loss_cycle_A = criterion_cycle(recov_A, real_A)
        loss_cycle_B = criterion_cycle(recov_B, real_B)
        loss_cycle = (loss_cycle_A + loss_cycle_B) / 2

        # G totol loss
        loss_G = 5.0 * loss_identity + loss_GAN + 10.0 * loss_cycle

        loss_G.backward()
        optimizer_G.step()

        """Train Discriminator A"""
        optimizer_D_A.zero_grad()

        loss_real = criterion_GAN(D_A(real_A), valid)
        loss_fake = criterion_GAN(D_A(fake_A.detach()), fake)
        loss_D_A = (loss_real + loss_fake) / 2

        loss_D_A.backward()
        optimizer_D_A.step()

        """Train Discriminator B"""
        optimizer_D_B.zero_grad()

        loss_real = criterion_GAN(D_B(real_B), valid)
        loss_fake = criterion_GAN(D_B(fake_B.detach()), fake)
        loss_D_B = (loss_real + loss_fake) / 2

        loss_D_B.backward()
        optimizer_D_B.step()

    lr_scheduler_G.step()
    lr_scheduler_D_A.step()
    lr_scheduler_D_B.step()

    print(epoch)
    if epoch == 1:
        torch.save(G_AB.state_dict(), './models/test.pth')
    # test
    if (epoch + 1) % 10 == 0:
        # test_real_A, test_real_B = next(iter(testloader))
        # sample_images(test_real_A, test_real_B)

        loss_D = (loss_D_A + loss_D_B) / 2
        print(f'[Epoch {epoch + 1}/{n_epoches}]')
        print(
            f'[G loss: {loss_G.item()} | identity: {loss_identity.item()} GAN: {loss_GAN.item()} cycle: {loss_cycle.item()}]')
        print(f'[D loss: {loss_D.item()} | D_A: {loss_D_A.item()} D_B: {loss_D_B.item()}]')

G_AB_PATH = './models/G_AB.pth'
G_BA_PATH = './models/G_BA.pth'
torch.save(G_AB.state_dict(), G_AB_PATH)
torch.save(G_BA.state_dict(), G_BA_PATH)
