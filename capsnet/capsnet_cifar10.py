import torch
import torchvision
import torch.nn.functional as F
from torch import nn, optim, autograd
from torch.autograd import Variable
from torchvision import transforms, utils 
import os
import numpy as np

from summary import summary

BATCH_SIZE = 100
# NUM_LABELS = 10
# model_path = 'capsnet.pkl'

NUM_LABELS = 10
model_path = 'capsnet_cifar.pkl'

save_img_iter = 0      
def save_img(images):
    global save_img_iter
    if not os.path.exists('capsnet_result_images/'):
        os.makedirs('capsnet_result_images/')
    grid = utils.make_grid(images)
    utils.save_image(grid, 'capsnet_result_images/img_{}.png'.format(str(save_img_iter).zfill(3)))
    save_img_iter += 1

class Mnist:
    def __init__(self, batch_size):
        dataset_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))])

        train_dataset = torchvision.datasets.MNIST('./data', train=True, download=True, transform=dataset_transform)
        test_dataset = torchvision.datasets.MNIST('./data', train=False, download=True, transform=dataset_transform)
        self.batch_size = batch_size
        self.train_loader  = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        self.test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    def before_save_img(self, images):
        images = images.view(images.size(0),1,28,28).data.cpu()[:64]
        return images

class CIFAR10(object):
    def __init__(self, batch_size):
        self.batch_size = batch_size
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        self.train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        self.test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

        self.train_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        self.test_loader = torch.utils.data.DataLoader(self.test_dataset, batch_size=batch_size, shuffle=True)
    def before_save_img(self, images):
        images = images.mul(0.5).add(0.5)
        images = images.view(images.size(0),3,32,32).data.cpu()[:64]
        return images

LATENT_DIM = 160
DIM = 128
        
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        # latent vector LATENT_DIM -> (DIM*4,4,4)
        self.block1 = nn.Sequential(
            nn.Linear(LATENT_DIM, DIM * 64),
            nn.BatchNorm1d(DIM * 64),
            nn.ReLU(True),
        )
        
        # (DIM*4,4,4) -> (DIM*2,8,8)
        self.block2 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=4 * DIM, out_channels=2 * DIM, kernel_size=2, stride=2),
            nn.BatchNorm2d(2 * DIM),
            nn.ReLU(True),
        )
        
        # (DIM*2,8,8) -> (DIM,16,16)
        self.block3 = nn.Sequential(
            nn.ConvTranspose2d(2 * DIM, DIM, 2, stride=2),
            nn.BatchNorm2d(DIM),
            nn.ReLU(True),
        )
        
        # (DIM,16,16) -> (3,32,32)
        self.block4 = nn.Sequential(
            nn.ConvTranspose2d(DIM, 3, 2, stride=2),
            nn.Tanh(),
        )
    def forward(self, input):
        output = self.block1(input)
        output = output.view(output.size(0), DIM * 4, 4, 4)
        output = self.block2(output)
        output = self.block3(output)
        output = self.block4(output)
        return output
        

def squash(tensor):
    squared_norm = (tensor ** 2).sum(dim=-1, keepdim=True)
    scale = squared_norm / (1.0 + squared_norm) / torch.sqrt(squared_norm)
    return scale * tensor

class PrimaryCapsule(nn.Module):
    def __init__(self, in_channels, out_channels, out_cap_dim, kernel_size, stride):
        super(PrimaryCapsule, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.out_cap_dim = out_cap_dim
        self.capsules = nn.ModuleList(
            [nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=0) for _ in range(out_cap_dim)])
    def forward(self, x):
        y = [capsule(x).view(x.size(0), -1, 1) for capsule in self.capsules]
        # (BATCH_SIZE, cap_num, cap_dim)
        y = torch.cat(y, dim=-1)
        y = squash(y).transpose(0,1)
        # y = (cap_num, BATCH_SIZE, cap_dim)
        return y
        

class RouteCapsule(nn.Module):
    def __init__(self, in_cap_num, in_cap_dim, out_cap_num, out_cap_dim, num_iter=3):
        super(RouteCapsule, self).__init__()
        self.in_cap_num = in_cap_num
        self.in_cap_dim = in_cap_dim
        self.out_cap_num = out_cap_num
        self.out_cap_dim = out_cap_dim
        self.num_iter = num_iter
        self.route_w = nn.Parameter(0.1 * torch.randn(out_cap_num, in_cap_num, in_cap_dim, out_cap_dim))
    def forward(self, x):
        # x.size = (in_cap_num, BATCH_SIZE, in_cap_dim)
        batch_size = x.size(1)
        
        # u.size = (out_cap_num, in_cap_num, BATCH_SIZE, out_cap_dim)
        u = x @ self.route_w
        
        # b.size = (out_cap_num, in_cap_num, BATCH_SIZE, 1)
        # b = Variable(torch.zeros(u.size(0), u.size(1), u.size(2), 1)).cuda()
        #b = Variable(torch.zeros(self.out_cap_num, self.in_cap_num, batch_size, 1)).cuda()
        b = Variable(torch.zeros(self.out_cap_num, 1, 1, 1)).cuda()
        for i in range(self.num_iter):
            # c.size = (out_cap_num, in_cap_num, BATCH_SIZE, 1)
            c = F.softmax(b, dim=0)
            # s.size = (out_cap_num, BATCH_SIZE, out_cap_dim)
            s =(c * u).sum(dim=1)
            # v.size = (out_cap_num, BATCH_SIZE, out_cap_dim)
            v = squash(s)
            if i < self.num_iter - 1:
                delta_b = (u * v.view(v.size(0), 1, v.size(1), v.size(2))).sum(dim=-1, keepdim=True)
                #print('delta_b:', delta_b.size(), b.size())
                b = b + delta_b
                #print('delta_b:', delta_b.max(), b.max())

            #else:
            #    print(c[:,:,0,:].sum(dim=1))
        # v.size = (out_cap_num, BATCH_SIZE, out_cap_dim)
        return v
        
        
        
        

class CapsNet(nn.Module):
    def __init__(self):
        super(CapsNet, self).__init__()
        #self.init_mnist()
        self.init_cifar()
        
    def init_mnist(self):        
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=256, kernel_size=9, stride=1)
        
        self.primary_caps = PrimaryCapsule(in_channels=256, out_channels=32, out_cap_dim=8, kernel_size=9, stride=2)
        
        self.digit_caps = RouteCapsule(in_cap_num=32*6*6, in_cap_dim=8, out_cap_num=NUM_LABELS, out_cap_dim=16, num_iter=3)
        
        self.decoder = Generator()
        
        self.reconstruction_loss = nn.MSELoss(reduction='sum')
        #self.reconstruction_loss = nn.MSELoss()
    def init_cifar(self):
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=256, kernel_size=9, stride=1)
        
        self.primary_caps = PrimaryCapsule(in_channels=256, out_channels=64, out_cap_dim=8, kernel_size=9, stride=2)
        
        self.digit_caps = RouteCapsule(in_cap_num=4096, in_cap_dim=8, out_cap_num=NUM_LABELS, out_cap_dim=16, num_iter=3)
        
        self.decoder = nn.Sequential(
            nn.Linear(16 * NUM_LABELS, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 3 * 32 * 32),
            nn.Sigmoid()
        )
        self.reconstruction_loss = nn.MSELoss(reduction='sum')
        #self.reconstruction_loss = nn.MSELoss()
    
    def forward(self, x, y = None):
        x = F.relu(self.conv1(x))
        x = self.primary_caps(x)
        x = self.digit_caps(x)
        
        # x.size = (batch_size, out_cap_num, out_cap_dim)
        x = x.transpose(0,1)
        
        # labels.size = (batch_size, out_cap_num)
        # labels[i,j] = ||v_j|| in batch_i
        labels = (x ** 2).sum(dim=-1) ** 0.5
        #labels = F.softmax(labels, dim=-1)
        if y is None:
            _, max_length_indices = labels.max(dim=-1)
            y = Variable(torch.eye(NUM_LABELS)).cuda().index_select(dim=0, index=max_length_indices.data)
            
        z = (x * y[:, :, None]).view(x.size(0), -1)
        
        reconstructions = self.decoder(z)
        
        return labels, reconstructions
        
    def loss(self, images, labels, pred_labels, reconstructions):
        return self.get_margin_loss(labels, pred_labels) + 0.0005 * self.get_reconstruction_loss(images, reconstructions)
        
    def get_margin_loss(self, y, pred_y):
        left = F.relu(0.9 - pred_y) ** 2
        right = F.relu(pred_y - 0.1) ** 2

        loss = y * left + 0.5 * (1.0 - y) * right 

        return loss.sum(dim = -1).mean()
        
    def get_reconstruction_loss(self, images, reconstructions):
        images = images.view(images.size(0), -1)
        return self.reconstruction_loss(reconstructions, images) 
        
    def save_model(self):
        torch.save(self.state_dict(), model_path)
    def load_model(self):
        self.load_state_dict(torch.load(model_path))
      


     
if __name__ == "__main__":
    #dataset = Mnist(BATCH_SIZE)
    dataset = CIFAR10(BATCH_SIZE)
    print("batch_num:", len(dataset.train_loader), len(dataset.test_loader))
    model = CapsNet()
    if os.path.exists(model_path):
        model.load_model()
    model.cuda()
    #summary(model,input_size=(1,28,28))
    optimizer = optim.Adam(
        model.parameters(), 
        lr = 0.0001, 
        #betas=(0, 0.9), 
        #weight_decay=0.01,
    )
    optimizer_decoder = optim.Adam(
        model.decoder.parameters(), 
        lr = 0.001, 
        #betas=(0, 0.9), 
        #weight_decay=0.01,
    )
    n_epochs = 500
    for epoch in range(n_epochs):
        print("epoch:", epoch)
        
        model.train()
        train_loss = 0.0
        for i, (images, labels) in enumerate(dataset.train_loader):
            labels = torch.eye(NUM_LABELS).index_select(dim=0, index=labels)
            images, labels = Variable(images).cuda(), Variable(labels).cuda()
            
            optimizer.zero_grad()
            pred_labels, reconstructions = model(images, labels)
            loss = model.loss(images, labels, pred_labels, reconstructions)
            loss.backward()
            optimizer_decoder.step()
            optimizer.step()
            
            loss = loss.data.cpu().numpy()
            train_loss += loss
            
            if (i+1) % 125 == 0:
                print("i:", i, "loss:", loss)
                print("pred_labels:", pred_labels.data[0].cpu().numpy())
                print("real_label:", labels.data[0].cpu().numpy())
                print("train accuracy:", sum(np.argmax(pred_labels.data.cpu().numpy(), 1) == np.argmax(labels.data.cpu().numpy(), 1)) / float(labels.size(0)))
                
        save_img(dataset.before_save_img(reconstructions))
        save_img(dataset.before_save_img(images))
        print("train_loss:", train_loss / len(dataset.train_loader))
        print("w.max:", model.digit_caps.route_w.max())
        print("w.min:", model.digit_caps.route_w.min())
        
        model.save_model()
        
        model.eval()
        test_loss = 0.0
        test_acc = 0.0
        for i, (images, labels) in enumerate(dataset.test_loader):
            labels = torch.eye(NUM_LABELS).index_select(dim=0, index=labels)
            images, labels = Variable(images), Variable(labels)
            images, labels = images.cuda(), labels.cuda()

            output, reconstructions = model(images)
            loss = model.loss(images, labels, output, reconstructions)
            test_loss += loss.data.cpu().numpy()
            test_acc += sum(np.argmax(output.data.cpu().numpy(), 1) == np.argmax(labels.data.cpu().numpy(), 1)) / float(labels.size(0))
            if i % 100 == 0:
                save_img(dataset.before_save_img(reconstructions))
                save_img(dataset.before_save_img(images))
                #print("test accuracy:", sum(np.argmax(output.data.cpu().numpy(), 1) == np.argmax(labels.data.cpu().numpy(), 1)) / float(labels.size(0)))

        print("test_loss:", test_loss / len(dataset.test_loader))
        print("test_acc:", test_acc / len(dataset.test_loader))
    
