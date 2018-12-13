import torch
import torchvision
from torch import nn
from torch import optim
from torch.autograd import Variable
from torchvision import transforms, utils 
import os

from summary import summary
LATENT_DIM = 128
DIM = 64
C_ITERS = 5
BATCH_SIZE = 64
LR = 0.00005
WCL = 0.01

class Discriminator(nn.Module):
	def __init__(self):
		super(Discriminator, self).__init__()
		self.model = nn.Sequential(
			# (32 - kernel_size + 2 * padding) / stride + 1
			# (32 - 4 + 2) / 2 + 1 = 16
			# (3, 32, 32) -> (DIM, 16, 16)
			nn.Conv2d(3, DIM, 4, stride=2, padding=1),
			nn.LeakyReLU(),
			# (DIM, 16, 16) -> (DIM * 2, 8, 8)
			nn.Conv2d(DIM, DIM * 2, 4, stride=2, padding=1),
			nn.LeakyReLU(),
			# (DIM * 2, 8, 8) -> (DIM * 4, 4, 4)
			nn.Conv2d(DIM * 2, DIM * 4, 4, stride=2, padding=1),
			nn.LeakyReLU(),
		)
		self.linear = nn.Linear(DIM * 64, 1)
		
	def forward(self, input):
		output = self.model(input)
		output = output.view(output.size(0),-1)
		output = self.linear(output)
		return output
	

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
		#return output.view(-1,3,32,32)
		return output
	

class CIFAR10_iter(object):
	def __init__(self):
		transform = transforms.Compose([
			transforms.ToTensor(),
			transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
		])
		self.trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
		self.iter = self.items()
	def __iter__(self):
		return self
	def __next__(self):
		return next(self.iter)
	def items(self):
		while True:
			trainloader = torch.utils.data.DataLoader(self.trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
			for images, labels in trainloader:
				yield images, labels
			
		

sava_path_G = 'generator.pkl'
sava_path_D = 'discriminator.pkl'

class WGAN(object):
	def __init__(self):
		self.G = Generator()
		self.D = Discriminator()
		if os.path.exists(sava_path_G):
			self.G.load_state_dict(torch.load(sava_path_G))
			print('load', sava_path_G)
		if os.path.exists(sava_path_D):
			self.D.load_state_dict(torch.load(sava_path_D))
			print('load', sava_path_D)
		self.data_iter = CIFAR10_iter()
		self.d_optimizer = optim.RMSprop(self.D.parameters(), lr=LR)
		self.g_optimizer = optim.RMSprop(self.G.parameters(), lr=LR)
		
		self.one = torch.ones(())
		self.neg_one = -self.one
		
		self.use_cuda = torch.cuda.is_available()
		if self.use_cuda:
			self.G.cuda()
			self.D.cuda()
			self.one = self.one.cuda()
			self.neg_one = self.neg_one.cuda()
		for p in self.D.parameters():
			p.data.clamp_(-WCL, WCL)	
			
		self.save_img_iter = 1
			
	def save_model(self):
		torch.save(self.G.state_dict(), sava_path_G)
		torch.save(self.D.state_dict(), sava_path_D)
		print('Models save to', sava_path_G, sava_path_D)
	
	def load_model(self, D_model_filename, G_model_filename):
		D_model_path = os.path.join(os.getcwd(), D_model_filename)
		G_model_path = os.path.join(os.getcwd(), G_model_filename)
		self.D.load_state_dict(torch.load(D_model_path))
		self.G.load_state_dict(torch.load(G_model_path))
		print('Generator model loaded from {}.'.format(G_model_path))
		print('Discriminator model loaded from {}-'.format(D_model_path))
	
	def train_step(self):
	
		for p in self.D.parameters():
			p.requires_grad = True
		for p in self.G.parameters():
			p.requires_grad = False
			
		for i in range(C_ITERS):
			self.D.zero_grad()
			images, labels = next(self.data_iter)
			#print(labels)
			
			z = torch.randn((BATCH_SIZE, LATENT_DIM))
			if self.use_cuda:
				images, z = Variable(images.cuda()), Variable(z.cuda())
			else:
				images, z = Variable(images), Variable(z)
				
			d_loss_real = self.D(images).mean()
			d_loss_real.backward(self.one)
			
			fake_images = self.G(z)
			d_loss_fake = self.D(fake_images).mean()
			d_loss_fake.backward(self.neg_one)
			WD = d_loss_real - d_loss_fake
			self.d_optimizer.step()
			
			for p in self.D.parameters():
				p.data.clamp_(-WCL, WCL)
		
		for p in self.G.parameters():
			p.requires_grad = True
		for p in self.D.parameters():
			p.requires_grad = False
		self.G.zero_grad()
		
		z = torch.randn((BATCH_SIZE, LATENT_DIM))
		if self.use_cuda:
			z = Variable(z.cuda())
		else:
			z = Variable(z)
		
		fake_images = self.G(z)
		g_loss = self.D(fake_images).mean()
		g_loss.backward(self.one)
		self.g_optimizer.step()
		
		return d_loss_real, d_loss_fake, g_loss 
		
	def generate_img(self):
		if not os.path.exists('training_result_images/'):
			os.makedirs('training_result_images/')
		z = torch.rand((64, LATENT_DIM))
		if self.use_cuda:
			z = Variable(z.cuda())
		else:
			z = Variable(z)		
		samples = self.G(z)
		samples = samples.mul(0.5).add(0.5)
		samples = samples.data.cpu()[:64]
		grid = utils.make_grid(samples)
		utils.save_image(grid, 'training_result_images/img_{}.png'.format(str(self.save_img_iter).zfill(3)))
		self.save_img_iter += 1
			
def main():	
	#D = Discriminator().to(device)
	#G = Generator().to(device)
	#print(str(D))
	#summary(D, input_size=(3,32,32))
	#print(str(G))
	#summary(G, input_size=(128,))
	gan = WGAN()
	while True:
		for i in range(1,1001):
			d_loss_real, d_loss_fake, g_loss = gan.train_step()
			if i % 10 == 0:
				print("i =", i, ":", d_loss_real.data.cpu().numpy(), d_loss_fake.data.cpu().numpy(), g_loss.data.cpu().numpy())
		gan.save_model()
		gan.generate_img()
		
if __name__ == '__main__':
	main()
