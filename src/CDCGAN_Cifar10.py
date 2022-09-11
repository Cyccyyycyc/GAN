# Adam没变化 -> RMSProp没变化 
# -> clip_value from 0.01 to 0.1 棋盘状围影严重 
# -> Adam G加通道and加转置卷积3次, G:先激活后bn,leaky 变 relu ,3个转置卷积 384通道

import argparse
import os
import numpy as np
import math

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch

data_save_path = "../data_set/cifar10"
model_save_path = "../net-module/V7W"
img_save_path = "../images/Cifar10-images/CDCGAN-W-V7"


parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=3000, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.00002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
# cat/proc/cpuinfo | grep "processor" | wc -l
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--n_classes", type=int, default=10, help="number of classes for dataset")
parser.add_argument("--img_size", type=int, default=32, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument("--n_critic", type=int, default=5, help="number of training steps for discriminator per iter")
parser.add_argument("--clip_value", type=float, default=0.1, help="lower and upper clip value for disc. weights")
parser.add_argument("--sample_interval", type=int, default=400, help="interval between image sampling")
parser.add_argument("--save_interval", type=int, default=300, help="interval between model saving")
opt = parser.parse_args()
print(opt)

# 1 * 32 * 32
img_shape = (opt.channels, opt.img_size, opt.img_size)

cuda = True if torch.cuda.is_available() else False

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.init_size = opt.img_size // 8  # 除法下取整等于4

        self.label_emb = nn.Embedding(num_embeddings=opt.n_classes, embedding_dim=opt.n_classes)
        self.l1 = nn.Sequential(nn.Linear(opt.latent_dim + opt.n_classes, 384 * self.init_size ** 2))

        self.conv_blocks = nn.Sequential(
            # N * 384 * 4 * 4
            nn.BatchNorm2d(384),
            nn.ConvTranspose2d(in_channels=384, out_channels=192, kernel_size=4, stride=2, padding=1),
            # N * 192 * 8 * 8
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(192),
            nn.ConvTranspose2d(in_channels=192, out_channels=96, kernel_size=4, stride=2, padding=1),
            # N * 96 * 16 * 16
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(96),
            nn.ConvTranspose2d(in_channels=96, out_channels=3, kernel_size=4, stride=2, padding=1),
            nn.Tanh()  # 归一化
        )

    def forward(self, noise, condition):
        gen_input = torch.cat((self.label_emb(condition), noise), dim=-1)
        out = self.l1(gen_input)
        out = out.view(out.shape[0], 384, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.l1 = nn.Sequential(nn.Embedding(num_embeddings=opt.n_classes, embedding_dim=opt.n_classes),
            nn.Linear(in_features=opt.n_classes, out_features=1 * opt.img_size ** 2))

        def discriminator_block(in_filters, out_filters, bn=True):
            block = [nn.Conv2d(in_channels=in_filters, out_channels=out_filters, kernel_size=4, stride=2, padding=1),
                     nn.LeakyReLU(0.2, inplace=True),
                     nn.Dropout2d(0.25)]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block

        self.model = nn.Sequential(
            # N * 2 * 32 * 32
            *discriminator_block(opt.channels + 1, 16, bn=False),
            # N * 16 * 16 * 16
            *discriminator_block(16, 32),
            # N * 32 * 8 * 8
            *discriminator_block(32, 64),
            # N * 64 * 4 * 4
            *discriminator_block(64, 128),
            # N * 128 * 2 * 2
        )

        # The height and width of downsampled image 32//16=2
        ds_size = opt.img_size // 2 ** 4
        self.adv_layer = nn.Sequential(nn.Linear(in_features=128 * ds_size ** 2, out_features=1),
                                       nn.Sigmoid())  # 归一化

    def forward(self, img, condition):
        condition = self.l1(condition)
        condition = condition.view(condition.shape[0], 1, opt.img_size, opt.img_size)
        # Concatenate label embedding and image to produce input
        d_in = torch.cat((img, condition),dim=1)
        out = self.model(d_in)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)
        return validity

# Initialize generator and discriminator
generator = Generator()
discriminator = Discriminator()

if cuda:
    generator.cuda()
    discriminator.cuda()

# Initialize weights
generator.apply(weights_init_normal)
discriminator.apply(weights_init_normal)

# Configure data loader
norm_mean = [0.5, 0.5, 0.5]  # 均值
norm_std = [0.5, 0.5, 0.5]  # 方差
data_transform = transforms.Compose(
            [transforms.Resize(opt.img_size),
             transforms.ToTensor(),
             transforms.Normalize(mean=norm_mean, std=norm_std)])

os.makedirs(data_save_path, exist_ok=True)
os.makedirs(model_save_path + "/Generator", exist_ok=True)
os.makedirs(model_save_path + "/Discriminator", exist_ok=True)

data_set = datasets.CIFAR10(root=data_save_path, train=True, download=True, transform=data_transform)
dataloader = DataLoader(dataset=data_set, batch_size=opt.batch_size, shuffle=True)

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=[opt.b1, opt.b2])
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=[opt.b1, opt.b2])

FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor

def sample_image(n_row, order):
    """Saves a grid of generated digits ranging from 0 to n_classes"""
    # Sample noise
    z = Variable(FloatTensor(np.random.normal(0, 1, (n_row ** 2, opt.latent_dim))))
    # Get labels ranging from 0 to n_classes for n rows
    labels = np.array([num for _ in range(n_row) for num in range(n_row)])
    labels = Variable(LongTensor(labels))
    gen_imgs = generator(z, labels)
    os.makedirs(img_save_path, exist_ok=True)
    save_image(gen_imgs.data, img_save_path + "/%d.png" % order, nrow=n_row, normalize=True)

# ----------
#  Training
# ----------
batches_done = 0
for epoch in range(opt.n_epochs):
    for i, (imgs, labels) in enumerate(dataloader):
        batch_size = imgs.shape[0]

        # Configure input
        real_imgs = Variable(imgs.type(FloatTensor))
        labels = Variable(labels.type(LongTensor))

        # ---------------------
        #  Train Discriminator
        # ---------------------
        optimizer_D.zero_grad()

        # Sample noise and labels as generator input
        z = Variable(FloatTensor(np.random.normal(0, 1, (batch_size, opt.latent_dim))))
        gen_labels = Variable(LongTensor(np.random.randint(low=0, high=opt.n_classes, size=batch_size)))

        # Generate a batch of images
        gen_imgs = generator(z, gen_labels).detach()  #第一次使用加detach()

        # Adversarial loss
        validity_real = discriminator(real_imgs, labels)
        validity_fake = discriminator(gen_imgs, gen_labels)  # 第一次使用加detach()
        loss_D = -torch.mean(validity_real) + torch.mean(validity_fake)

        loss_D.backward()
        optimizer_D.step()

        # # Clip weights of discriminator
        # for p in discriminator.parameters():
        #     p.data.clamp_(-opt.clip_value, opt.clip_value)

        # Train the generator every n_critic iterations
        if i % opt.n_critic == 0:
            # -----------------
            #  Train Generator
            # -----------------

            optimizer_G.zero_grad()

            # Generate a batch of images
            gen_imgs = generator(z, gen_labels)

            # Adversarial loss
            loss_G = -torch.mean(discriminator(gen_imgs, gen_labels))

            loss_G.backward()
            optimizer_G.step()

            print(
                "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                % (epoch, opt.n_epochs, batches_done % len(dataloader), len(dataloader), loss_D.item(), loss_G.item())
            )

        if batches_done % opt.sample_interval == 0:
            sample_image(n_row=10, order=batches_done/opt.sample_interval)
        batches_done += 1
    if (epoch + 1) % opt.save_interval == 0:
        torch.save(generator, model_save_path + "/Generator/G{}".format(epoch // opt.save_interval))
        torch.save(discriminator, model_save_path + "/Discriminator/D{}".format(epoch // opt.save_interval))



