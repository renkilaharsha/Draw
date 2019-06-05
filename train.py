import torch.optim as optim
import numpy as np
import pandas as pd
from torchvision import datasets,transforms
import torch.utils
from draw_model import DrawModel
from config import *
from utility import Variable,save_image,xrecons_grid
import torch.nn.utils
import matplotlib.pyplot as plt
import torch
import torchvision
from torch import nn
#from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import DataLoader,Dataset
from torchvision import transforms
#from torchvision.utils import save_image
#from torchvision.datasets import MNIST
import os

#torch.set_default_tensor_type('torch.FloatTensor')
cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
torch.set_default_tensor_type('torch.FloatTensor')

img_transform = transforms.Compose([transforms.ToTensor()])
class CustomDatasetFromCSV(Dataset):
    def __init__(self, csv_path, height, width, transform):
        """
        Args:
            csv_path (string): path to csv file
            height (int): image height
            width (int): image width
            transform: pytorch transforms for transforms and tensor conversion
        """
        self.data = pd.read_csv(csv_path)
        self.labels = np.asarray(self.data.iloc[:, -1])
        self.height = height
        self.width = width
        self.transform = transform

    def __getitem__(self, index):
        single_image_label = self.labels[index]
        # Read each 784 pixels and reshape the 1D array ([784]) to 2D array ([28,28])
        img_as_np = np.asarray(self.data.iloc[index][:-1]).reshape(32, 32).astype('uint16')
        # Convert image from numpy array to PIL image, mode 'L' is for grayscale
        img_as_img = Image.fromarray(img_as_np)
        img_as_img = img_as_img.convert('L')
        # Transform image to tensor
        if self.transform is not None:
            img_as_tensor = self.transform(img_as_img)
        # Return image and the label
        return (img_as_tensor, single_image_label)

    def __len__(self):
        return len(self.data.index)

img_transform = transforms.Compose([transforms.ToTensor()])
dataset=CustomDatasetFromCSV('/home/17mcmi06/harsha/project/draw_tel_unbalanced/UHTelPCC.csv',32,32, transform=img_transform)

train_loader = DataLoader(dataset,batch_size=batch_size,shuffle =True)

model = DrawModel(T,A,B,z_size,N,dec_size,enc_size)
optimizer = optim.Adam(model.parameters(),lr=learning_rate,betas=(beta1,0.999))


if USE_CUDA:

    model.cuda()






def generate_image(count):
    x = model.generate(batch_size)
    return x

def save_example_image():
    train_iter = iter(train_loader)
    #print(train_iter.shape())
    data, _ = train_iter.next()
    img = data.cpu().numpy().reshape(batch_size, 32, 32)
    imgs = xrecons_grid(img, B, A)
    plt.matshow(imgs, cmap=plt.cm.gray)
    plt.savefig('image/example.png')

if __name__ == '__main__':
    save_example_image()
    data_ix = np.empty((1,785))
  #  train()

    avg_loss = 0
    count = 0
    for epoch in range(epoch_num):
        for data,lables in train_loader:
            bs = data.size()[0]
            data = Variable(data.float()).view(bs, -1).cuda()
            optimizer.zero_grad()
            loss = model.loss(data)
            avg_loss += loss.cpu().data.numpy()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()
            count += 1
            print ('Epoch-{}; Count-{}; loss: {};'.format(epoch, count, avg_loss / 100))
            #x=generate_image(count)
            #save_image(x,count)
            if count % 100 == 0:
                print ('Epoch-{}; Count-{}; loss: {};'.format(epoch, count, avg_loss / 100))
                torch.save(model.state_dict(),'save/weights_%d.tar'%(count))

                x=generate_image(count)
                save_image(x,count)
                avg_loss = 0
    torch.save(model.state_dict(), 'save/weights_final.tar')
    generate_image(count)

