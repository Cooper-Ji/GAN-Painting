import torch
from Generator import GeneratorResNet
from torch.utils.data import DataLoader
from Data import ImageDataset
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torchvision.utils import make_grid

AB_PATH = './models/G_AB.pth'
BA_PATH = './models/G_BA.pth'
data_dir = './datasets/'
batch_size = 5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

G_AB = GeneratorResNet(3, num_residual_blocks=9)
G_BA = GeneratorResNet(3, num_residual_blocks=9)
G_AB.load_state_dict(torch.load(AB_PATH))
G_BA.load_state_dict(torch.load(BA_PATH))
G_AB.to(device)
G_BA.to(device)

transforms_ = transforms.Compose([
    # transforms.Resize(int(256*1.12), Image.BICUBIC),
    # #transforms.RandomCrop(256, 256),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),   # 把数据转为Tensor,并Norm到（0， 1）
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))   # 处理之后的图片数值范围在[-1, 1]
])

testloader = DataLoader(
    ImageDataset(data_dir, mode='test', transforms=transforms_),
    batch_size=batch_size,
    shuffle=True,
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


with torch.no_grad():
    G_BA.eval()

    real_A, real_B = next(iter(testloader))
    sample_images(real_A, real_B)


