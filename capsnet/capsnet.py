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
NUM_LABELS = 7
model_path = 'capsnet.pkl'

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
    def train_iter(self):
        def items():
            while True:
                self.train_loader  = torch.utils.data.DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
                for images, labels in self.train_loader():
                    yield images, labels
        return items()

class CIFAR10_iter(object):
    def __init__(self, batch_size):
        self.batch_size = batch_size
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        self.trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        self.iter = self.infinit_items()
    def __iter__(self):
        return self
    def __next__(self):
        return next(self.iter)
    def infinit_items(self):
        while True:
            trainloader = torch.utils.data.DataLoader(self.trainset, batch_size=self.batch_size, shuffle=True, num_workers=0)
            for images, labels in trainloader:
                if labels.size()[0] < self.batch_size:
                    break
                yield images, labels

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
        self.init_mnist()
        
    def init_mnist(self):        
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=256, kernel_size=9, stride=1)
        
        self.primary_caps = PrimaryCapsule(in_channels=256, out_channels=32, out_cap_dim=8, kernel_size=9, stride=2)
        
        self.digit_caps = RouteCapsule(in_cap_num=32*6*6, in_cap_dim=8, out_cap_num=10, out_cap_dim=16, num_iter=3)
        
        self.decoder = nn.Sequential(
            nn.Linear(16 * 10, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 784),
            nn.Sigmoid()
        )
        
        self.reconstruction_loss = nn.MSELoss(reduction='sum')
        #self.reconstruction_loss = nn.MSELoss()
    def init_cifar(self):
        return
        
    
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
            y = Variable(torch.eye(10)).cuda().index_select(dim=0, index=max_length_indices.data)
        
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
      

save_img_iter = 0      
def save_img(images):
    global save_img_iter
    if not os.path.exists('capsnet_result_images/'):
        os.makedirs('capsnet_result_images/')
    images = images.view(images.size(0),1,28,28).data.cpu()[:64]
    #print(images[0])
    grid = utils.make_grid(images)
    utils.save_image(grid, 'capsnet_result_images/img_{}.png'.format(str(save_img_iter).zfill(3)))
    save_img_iter += 1   
     
if __name__ == "__main__":
    mnist = Mnist(BATCH_SIZE)
    model = CapsNet()
    if os.path.exists(model_path):
        model.load_model()
    model.cuda()
    #summary(model,input_size=(1,28,28))
    optimizer = optim.Adam(model.parameters())
    n_epochs = 500
    for epoch in range(n_epochs):
        print("epoch:", epoch)
        
        model.train()
        train_loss = 0.0
        for i, (images, labels) in enumerate(mnist.train_loader):
        
            labels = torch.eye(10).index_select(dim=0, index=labels)
            images, labels = Variable(images).cuda(), Variable(labels).cuda()
            
            optimizer.zero_grad()
            pred_labels, reconstructions = model(images, labels)
            loss = model.loss(images, labels, pred_labels, reconstructions)
            loss.backward()
            optimizer.step()
            
            loss = loss.data.cpu().numpy()
            train_loss += loss
            
            if i % 100 == 0:
                print("i:", i, "loss:", loss)
                print("pred_labels:", pred_labels.data[0].cpu().numpy())
                print("real_label:", labels.data[0].cpu().numpy())
                print("train accuracy:", sum(np.argmax(pred_labels.data.cpu().numpy(), 1) == np.argmax(labels.data.cpu().numpy(), 1)) / float(labels.size(0)))
                print("w.max:", model.digit_caps.route_w.max())
                print("w.min:", model.digit_caps.route_w.min())
        print("train_loss:", train_loss / len(mnist.train_loader))
        save_img(reconstructions)
        save_img(images)
        model.save_model()
        
        model.eval()
        test_loss = 0.0
        for i, (images, labels) in enumerate(mnist.test_loader):
            labels = torch.eye(10).index_select(dim=0, index=labels)
            images, labels = Variable(images), Variable(labels)
            images, labels = images.cuda(), labels.cuda()

            output, reconstructions = model(images)
            loss = model.loss(images, labels, output, reconstructions)
            test_loss += loss.data.cpu().numpy()
            if i % 100 == 0:
                print("test accuracy:", sum(np.argmax(output.data.cpu().numpy(), 1) == np.argmax(labels.data.cpu().numpy(), 1)) / float(labels.size(0)))

        print("test_loss:", test_loss / len(mnist.test_loader))
    
