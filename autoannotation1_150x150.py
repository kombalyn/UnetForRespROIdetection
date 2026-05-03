# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch import optim
import os
import cv2
import random
import torchvision
from PIL import Image
import PIL

#! unzip  TestJPEGImages-20210528T092304Z-001.zip

#!unzip  TestSegmentationClass-20210528T081430Z-001.zip

"""#Load Data"""

import os
from google.colab import drive
drive.mount('/content/drive')
os.chdir("/content/drive/My Drive")

'''
Set load parameters
'''
reading_path_original = "DiffBasedAutoAnnotation/autoAnnonation1/frames0/"
#reading_path_diff = "saved_diffs2/"
reading_path_mask = "DiffBasedAutoAnnotation/autoAnnonation1/masks0/"
szamlalo = 0
batch_length = 3

w_size = 150
h_size = 150

print(reading_path_mask+"frame_002222.npy")

img = cv2.imread(reading_path_original+"frame_2000.png")
#diff = cv2.imread(reading_path_diff+"diff_%04d.png" %  (szamlalo))
#diff = cv2.cvtColor(diff, cv2.COLOR_RGB2GRAY)
mask = cv2.imread(reading_path_mask+"mask_2000.png")

print(reading_path_mask+"mask_2000.png")

mask = mask.astype('uint8')

mask=cv2.resize(mask,(w_size,h_size))

plt.imshow(img)

plt.imshow(mask)
print(mask.shape)

'''
entries = os.listdir(path)
print(entries)

for entry in entries:
    if(entry.isnumeric()):
        print(entry)

for entry in entries:
    if(entry.isnumeric()):
        print(entry)
        files_in_entry = os.listdir(path+"/"+entry)
        for file_in_entry in files_in_entry:
            if(file_in_entry[-4:]==".txt" and file_in_entry[:2]!="S_"):
                print(file_in_entry)
                os.rename(path+"/"+entry+"/"+file_in_entry,path+"/"+entry+"/"+file_in_entry[1:])
'''

entries = os.listdir(reading_path_original)
print(len(entries))

print(entries[0].split('.')[0])

entries[0]

entries = entries[:20]

print(entries)

len(entries)

used_in_batch = random.sample(entries, batch_length)

print(type(used_in_batch))

print(used_in_batch)

used_in_batch_mask = []
for l in range(len(used_in_batch)):
  name = used_in_batch[l].split('.')[0]
  used_in_batch_mask.append(name+".npy")

print(used_in_batch_mask)

for l in range(len(used_in_batch)):
  entries.remove(used_in_batch[l]);

print(entries)

len(entries)

def getMask(mask):
  a = mask[:,:,0]==255
  b = mask[:,:,1]==0
  c = mask[:,:,2]==0
  m = a*b*c*255
  return cv2.merge([m,m,m]).astype(np.uint8)

def increase_brightness(img, value=30):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    lim = 255 - value
    v[v > lim] = 255
    v[v <= lim] += value

    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return img

def get_next_batch(entries, batch_length):
    input_images = []
    mask_images = []
    used_in_batch = random.sample(entries, batch_length)
    used_in_batch_mask = []
    for l in range(len(used_in_batch)):
      name = used_in_batch[l].split('_')[1]
      used_in_batch_mask.append("mask_"+name)
    for i in range(len(used_in_batch)):
      img = cv2.imread(reading_path_original+used_in_batch[i])
      #diff = cv2.imread(used_in_batch_mask[i])
      mask = cv2.imread(reading_path_mask+used_in_batch_mask[i])



      img = cv2.resize(img, (w_size,h_size))
      mask = cv2.resize(mask, (w_size,h_size))
      mask = getMask(mask)

      # gain augmentation
      val = random.randint(10,150)
      img = increase_brightness(img,val)

      if (0.5<random.random()):
        img = cv2.flip(img, 0)
        #diff = cv2.flip(diff, 0)
        mask = cv2.flip(mask, 0)
      if (0.5<random.random()):
        img = cv2.flip(img, 1)
        #diff = cv2.flip(diff, 1)
        mask = cv2.flip(mask, 1)

      #angle = random.randint(-30,30)
      myr = random.random()
      if (myr<0.25):
        img=cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        mask=cv2.rotate(mask, cv2.ROTATE_90_CLOCKWISE)
      elif (0.25<myr and myr<0.5):
        img=cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        mask=cv2.rotate(mask, cv2.ROTATE_90_CLOCKWISE)
        img=cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        mask=cv2.rotate(mask, cv2.ROTATE_90_CLOCKWISE)
      elif (0.5<myr and myr<0.75):
        img=cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        mask=cv2.rotate(mask, cv2.ROTATE_90_CLOCKWISE)
        img=cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        mask=cv2.rotate(mask, cv2.ROTATE_90_CLOCKWISE)
        img=cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        mask=cv2.rotate(mask, cv2.ROTATE_90_CLOCKWISE)

      #angle = random.randint(-30,30)
      #img = rotate_image(img,angle)
      #diff = rotate_image(diff,angle)
      #mask = rotate_image(mask,angle)

      #diff = cv2.cvtColor(diff, cv2.COLOR_RGB2GRAY)
      #image = cv2.merge((img, diff))
      image = img
      input_images.append(image)
      #mask_image = cv2.merge((mask, diff))
      mask_image = mask
      mask_images.append(mask_image)

    imgs = np.asarray(input_images)
    masks = np.asarray(mask_images)
    imgs = np.swapaxes(imgs,1,3)
    masks = np.swapaxes(masks,1,3)
    imgs = torch.from_numpy(imgs)
    masks = torch.from_numpy(masks)


    for l in range(len(used_in_batch)):
      entries.remove(used_in_batch[l]);

    return imgs, masks, entries

entries = os.listdir(reading_path_original)

img_batch, mask_batch, entries = get_next_batch(entries, 2)

print(img_batch.shape)
print(mask_batch.shape)

#plt.imshow(img_batch[0][:,:,3])

import matplotlib.pyplot as plt

plt.imshow(img_batch[0][0,:,:])

plt.imshow(mask_batch[0][0,:,:])

plt.imshow(img_batch[0][0,:,:]*mask_batch[0][0,:,:])

horizontal_flip = torchvision.transforms.RandomHorizontalFlip(p=1)
vertical_flip = torchvision.transforms.RandomVerticalFlip(p=1)

def rotate_image(image, angle):
  image_center = tuple(np.array(image.shape[1::-1]) / 2)
  rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
  result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
  return result

"""# Define Methods"""

#!/usr/bin/env python
# coding: utf-8
#get_ipython().run_line_magic('pylab', 'inline')

def jaccard_distance_loss(y_true, y_pred, smooth=100):
    """
    Jaccard = (|X & Y|)/ (|X|+ |Y| - |X & Y|)
            = sum(|A*B|)/(sum(|A|)+sum(|B|)-sum(|A*B|))

    The jaccard distance loss is usefull for unbalanced datasets. This has been
    shifted so it converges on 0 and is smoothed to avoid exploding or disapearing
    gradient.

    Ref: https://en.wikipedia.org/wiki/Jaccard_index

    @url: https://gist.github.com/wassname/17cbfe0b68148d129a3ddaa227696496
    @author: wassname
    """
    intersection= (y_true * y_pred).abs().sum(dim=(2,3))
    sum_ = torch.sum(y_true.abs() + y_pred.abs(), dim=(2,3))
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return torch.mean((1 - jac) * smooth)

# We are using DoubleConv In many places in the architechture. So first we will define this one.
# This will be the class double conv
class DoubleConv(nn.Module): # If we want to use a class as a building block of an architechture the we have to inherit nn.Module
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__() # First we have to initialize the ancestor
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.double_conv(x)


# First lets define a class for creating downsampling layers:
class Down(nn.Module):
    """Downscaling with maxpool then double conv"""
    def __init__(self, in_channels, out_channels): # This will wait for two inputs: the number of in_channels and ...
        super().__init__()
        self.maxpool_conv = nn.Sequential( # After we will define maxpool_conv which consist of a..
            DoubleConv(in_channels, out_channels),
            nn.MaxPool2d(2)
        )
    def forward(self, x): # In the forward prop we will apply this on the input x.
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)
    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
    def forward(self, x):
        return self.conv(x)

# We are using DoubleConv In many places in the architechture. So first we will define this one.
# This will be the class double conv
class DoubleConv(nn.Module): # If we want to use a class as a building block of an architechture the we have to inherit nn.Module
    def __init__(self, in_channels, out_channels):
        super().__init__() # First we have to initialize the ancestor
        self.double_conv = nn.Sequential( # nOW WE WILL define the double_conv which will contain the 2 convolutions with batchnorm
            nn.Conv2d(in_channels, out_channels, kernel_size=5, padding=2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=5, padding=2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.double_conv(x)


# First lets define a class for creating downsampling layers:
class Down(nn.Module):
    """Downscaling with maxpool then double conv"""
    def __init__(self, in_channels, out_channels): # This will wait for two inputs: the number of in_channels and ...
        super().__init__()
        self.maxpool_conv = nn.Sequential( # After we will define maxpool_conv which consist of a..
            DoubleConv(in_channels, out_channels),
            nn.MaxPool2d(2)
        )
    def forward(self, x): # In the forward prop we will apply this on the input x.
        return self.maxpool_conv(x)

# After lets define a class for UP layers
class Up(nn.Module):
    """Upscaling then double conv"""
    def __init__(self, in_channels, out_channels):
        super().__init__() # First we have to initialize the ancestor

        # After lets define the parts of up layer. In this we have
        self.up = nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size=3, stride=2)
            #the floor division // rounds the result down to the nearest whole number
        self.conv = DoubleConv(in_channels, out_channels)
    def forward(self, x1, x2): # In the forward prop we have two inputs: one is coming from down (x1) and one
                               # is coming from the skip connection (x2)
        x1 = self.up(x1) # We will apply the transpose conv o the x1

        # NOw we will calculate what is the difference in x and y direction btween the size of x1 and x2
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        # It is important to calculate how mutch padding we needin the different
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        # wITH TORCH cat we can concatenate the two tensor
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

# Finally we will define a class for the outCon which
# will take us back to the out_channel num.
class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
    def forward(self, x):
        return self.conv(x)

"""#Build Network"""

class UNET(nn.Module):
  def __init__(self, i_ch, o_ch):
    super().__init__()

    # downsampling
    self.inc = DoubleConv(i_ch, 64) #i_ch = input_channel
    self.down1 = Down(64,128)
    self.down2 = Down(128,256)
    self.down3 = Down(256,512)
    self.down4 = Down(512,1024)

    # upsampling
    self.up1 = Up(1024, 512)
    self.up2 = Up(512, 256)
    self.up3 = Up(256, 128)
    self.up4 = Up(128, 64)
    self.outconv = OutConv(64, o_ch) #o_ch = out_channel

  def forward(self, x):
    x1 = self.inc(x)
    x2 = self.down1(x1)
    x3 = self.down2(x2)
    x4 = self.down3(x3)
    x5 = self.down4(x4)

    x = self.up1(x5, x4)
    x = self.up2(x, x3)
    x = self.up3(x, x2)
    x = self.up4(x, x1)
    x = self.outconv(x)

    return x

dev = torch.device("cuda:0")

#net = UNET(4,4).to(dev)  #4 input channel, 4 output channel
net = UNET(3,3).to(dev)  #3 input channel, 3 output channel
print(net)

entries = os.listdir(reading_path_original)

img_batch, mask_batch, indices = get_next_batch(entries, 2)

img_batch = img_batch.to(dev,dtype=torch.float32)
mask_batch = mask_batch.to(dev,dtype=torch.float32)

out = net(img_batch)

print(out.shape)

"""#Loss and Optimizer"""

optimizer = optim.RMSprop(net.parameters(), 0.001, 1e-8, 0.9)
criterion = nn.MSELoss()
#criterion = nn.CrossEntropyLoss()
'''
if 1 > 1:
    criterion = nn.CrossEntropyLoss()
else:
    criterion = nn.BCEWithLogitsLoss()
'''

"""#Training"""

#[N,CH,H,W]
#[N,W,H,CH]

#ls checkpoints

'''
Load Model
'''
#net.load_state_dict(torch.load("checkpoints/3channel-UNET_version-450.pth"))
#net.load_state_dict(torch.load("checkpoints/200x200cross3channel-UNET_version-0190.pth"))
#net.load_state_dict(torch.load("checkpoints/OnKerasDataset200x200cross3channel-UNET_version-0032.pth"))
#net.load_state_dict(torch.load("checkpoints/autoAnnotation200x200cross3channel-UNET_version-0005.pth"))
#net.load_state_dict(torch.load("checkpoints/autoAnnotation200x200cross3channel-crossCorrelation-UNET_version-0010.pth"))

entries = os.listdir(reading_path_original)

print(entries)

print(int((entries[1][6:-4])))

entries_train=[]
entries_test=[]
for i in range(len(entries)):
  if(373<int((entries[i][6:-4])) and int((entries[i][6:-4]))<874):
    entries_test.append(entries[i])
  else:
    entries_train.append(entries[i])

print(entries_train)

epoch_num = 66    #66  # 202: aumnetalas rotation nelkul;   205: augmentalas rotationnal
batch_size = 8
losses = []

net.load_state_dict(torch.load("checkpoints/autoAnnotation1_150x150_4_mely_nagyobb_kernelek-UNET_version-0020.pth"))

#import random

for epoch in range(20,epoch_num):
  epoch_loss = 0

  entries = os.listdir(reading_path_original)
  entries_train=[]
  entries_test=[]
  for i in range(len(entries)):
    if(373<int((entries[i][6:-4])) and int((entries[i][6:-4]))<874):
      entries_test.append(entries[i])
    else:
      entries_train.append(entries[i])

  n = 0
  while (batch_size<len(entries_train)):

    img_batch, mask_batch, entries_train = get_next_batch(entries_train, batch_size)
    img_batch = img_batch.to(dev,dtype=torch.float32)
    mask_batch = mask_batch.to(dev,dtype=torch.float32)
    #mask_batch = mask_batch / 255
    #mask_batch = mask_batch.to(dev,dtype=torch.long)

    # If your targets contain the class indices already, you should remove the channel dimension:
    #print(mask_batch.shape)
    #mask_batch = mask_batch.squeeze(1)
    #print(mask_batch.shape)

    # forward
    mask_pred = net(img_batch)

    # calculate loss
    loss = criterion(mask_pred,mask_batch)
    epoch_loss += loss.item()
    n = n + 1

    # zero grad
    optimizer.zero_grad()

    # backpropagation
    loss.backward()

    # optimization
    optimizer.step()

  print('Epoch ' + str(epoch))
  print("actual loss: " + str(epoch_loss/n))
  losses.append(epoch_loss/n)
#aaaaa
  if epoch % 5 == 0:
        #torch.save(net.state_dict(), "checkpoints/cross3channel-UNET_version-%04d.pth" % epoch)
        torch.save(net.state_dict(), "checkpoints/autoAnnotation1_150x150_4_mely_nagyobb_kernelek-UNET_version-%04d.pth" % epoch)

#aaaaaaaaaaaaaaa

for epoch in range(epoch_num):
  epoch_loss = 0

  entries = os.listdir(reading_path_original)
  entries_train=[]
  entries_test=[]
  for i in range(len(entries)):
    if(373<int((entries[i][6:-4])) and int((entries[i][6:-4]))<874):
      entries_test.append(entries[i])
    else:
      entries_train.append(entries[i])

  n = 0
  while (batch_size<len(entries_train)):

    img_batch, mask_batch, entries_train = get_next_batch(entries_train, batch_size)
    img_batch = img_batch.to(dev,dtype=torch.float32)
    mask_batch = mask_batch.to(dev,dtype=torch.float32)
    #mask_batch = mask_batch / 255
    #mask_batch = mask_batch.to(dev,dtype=torch.long)

    # If your targets contain the class indices already, you should remove the channel dimension:
    #print(mask_batch.shape)
    #mask_batch = mask_batch.squeeze(1)
    #print(mask_batch.shape)

    # forward
    mask_pred = net(img_batch)

    # calculate loss
    loss = criterion(mask_pred,mask_batch)
    epoch_loss += loss.item()
    n = n + 1

    # zero grad
    optimizer.zero_grad()

    # backpropagation
    loss.backward()

    # optimization
    optimizer.step()

  print('Epoch ' + str(epoch))
  print("actual loss: " + str(epoch_loss/n))
  losses.append(epoch_loss/n)
#aaaaaaaaaaaaaaaa
  if epoch % 5 == 0:
        #torch.save(net.state_dict(), "checkpoints/cross3channel-UNET_version-%04d.pth" % epoch)
        torch.save(net.state_dict(), "checkpoints/2MSEautoAnnotation200x200cross3channel-crossCorrelation-UNET_version-%04d.pth" % epoch)

#aaaaaadddfffffyyxaaaaaaaaaaaassssAadggssswwwaaaaaaataaaaaaaaaaaa

'''img_batch, mask_batch, entries = get_next_batch(entries, batch_size)
img_batch = img_batch.to(dev,dtype=torch.float32)
#mask_batch = mask_batch.to(dev,dtype=torch.float32)
#mask_batch = mask_batch / 255
mask_batch = mask_batch.to(dev,dtype=torch.long)

# If your targets contain the class indices already, you should remove the channel dimension:
print(mask_batch[mask_batch!=0])
#mask_batch = mask_batch.squeeze(1)
print(mask_batch.shape)


# forward
mask_pred = net(img_batch)

mask_pred[mask_pred<0.5] = 0
mask_pred[0.5<mask_pred] = 1

print(mask_pred[mask_pred!=0])

loss = jaccard_distance_loss(mask_batch,mask_pred)
print(loss)'''

plt.plot(losses) #aahhhhhhhh

import matplotlib.pyplot as plt
plt.plot(losses)

mask_pred = net(img_batch)

"""# DEMO"""

pwd

#os.chdir("/content/drive/My Drive/")

'''
Load Model aaa
'''
net.load_state_dict(torch.load("checkpoints/autoAnnotation1_150x150_4_mely_nagyobb_kernelek-UNET_version-0045.pth"))
#autoAnnotation1_150x150_4_mely_nagyobb_kernelek-UNET_version-0045.pth

#net.load_state_dict(torch.load("checkpoints/3channel-UNET_version-0095.pth"))
#net.load_state_dict(torch.load("checkpoints/200x200cross3channel-UNET_version-0185.pth"))
#net.load_state_dict(torch.load("checkpoints/200x200cross3channel-UNET_version-0190.pth"))
#net.load_state_dict(torch.load("checkpoints/250x250cross3channel-UNET_version-0100.pth"))
#net.load_state_dict(torch.load("checkpoints/OnKerasDataset200x200cross3channel-UNET_version-0070.pth"))
#net.load_state_dict(torch.load("checkpoints/WithJaccardOnKerasDataset200x200cross3channel-UNET_version-0005.pth"))
#net.load_state_dict(torch.load("checkpoints/autoAnnotation200x200cross3channel-crossCorrelation-UNET_version-0045.pth"))
#net.load_state_dict(torch.load("checkpoints/2MSEautoAnnotation200x200cross3channel-crossCorrelation-UNET_version-0035.pth"))

used_in_batch = [entries[1],entries[5]]
print(used_in_batch[1])

def get_next_batch(entries, batch_length):
    input_images = []
    mask_images = []
    used_in_batch = random.sample(entries, batch_length)
    used_in_batch_mask = []
    for l in range(len(used_in_batch)):
      name = used_in_batch[l].split('_')[1]
      used_in_batch_mask.append("mask_"+name)
    for i in range(len(used_in_batch)):
      img = cv2.imread(reading_path_original+used_in_batch[i])
      #diff = cv2.imread(used_in_batch_mask[i])
      mask = cv2.imread(reading_path_mask+used_in_batch_mask[i])



      img = cv2.resize(img, (w_size,h_size))
      mask = cv2.resize(mask, (w_size,h_size))


      if (0.5<random.random()):
        img = cv2.flip(img, 0)
        #diff = cv2.flip(diff, 0)
        mask = cv2.flip(mask, 0)
      if (0.5<random.random()):
        img = cv2.flip(img, 1)
        #diff = cv2.flip(diff, 1)
        mask = cv2.flip(mask, 1)

      #angle = random.randint(-30,30)
      #img = rotate_image(img,angle)
      #diff = rotate_image(diff,angle)
      #mask = rotate_image(mask,angle)

      #diff = cv2.cvtColor(diff, cv2.COLOR_RGB2GRAY)
      #image = cv2.merge((img, diff))
      image = img
      input_images.append(image)
      #mask_image = cv2.merge((mask, diff))
      mask_image = mask
      mask_images.append(mask_image)

    imgs = np.asarray(input_images)
    masks = np.asarray(mask_images)
    imgs = np.swapaxes(imgs,1,3)
    masks = np.swapaxes(masks,1,3)
    imgs = torch.from_numpy(imgs)
    masks = torch.from_numpy(masks)


    for l in range(len(used_in_batch)):
      entries.remove(used_in_batch[l]);

    return imgs, masks, entries

img_batch, mask_batch, entries = get_next_batch(entries, 2)
plt.imshow(img_batch[0,0,:,:])
img_batch = img_batch.to(dev,dtype=torch.float32)
#mask_batch = mask_batch.to(dev,dtype=torch.float32)
#mask_batch = mask_batch / 255
plt.figure()
plt.imshow(mask_batch[0,0,:,:])
mask_batch = mask_batch.to(dev,dtype=torch.long)
print(mask_batch[0,0,125,100])

# If your targets contain the class indices already, you should remove the channel dimension:
#print(mask_batch.shape)
#mask_batch = mask_batch.squeeze(1)
#print(mask_batch.shape)

# forward
mask_pred = net(img_batch)

print(reading_path_original+"frame_002222.jpg")

print(img_batch.dtype)
print(img_batch.shape)

img_batch = np.swapaxes(img_batch,1,3)
#img_batch = np.swapaxes(img_batch,2,3)
input_img = img_batch.cpu().detach().numpy()
plt.imshow(input_img[0,:,:,:]/255)

array = mask_pred.cpu().detach().numpy()
array = np.swapaxes(array,1,3)
print(array.shape)
plt.imshow(array[0,:,:,:])
plt.colorbar()

array = mask_pred.cpu().detach().numpy()
array = np.swapaxes(array,1,3)
print(array.shape)
plt.imshow(array[0,:,:,0])
plt.colorbar()

m = array[0,:,:,0]
#m = cv2.cvtColor(m, cv2.COLOR_RGB2GRAY)
#m = m[200<m]
m[m<200] = 0
m[200<m] = 1
plt.imshow(m.astype(np.uint))
plt.colorbar()

img = input_img[0,:,:,:]/255
img[:,:,0]=img[:,:,0]*m
img[:,:,1]=img[:,:,1]*m
img[:,:,2]=img[:,:,2]*m
plt.imshow(img)

"""# Try on TESTSET"""

os.chdir("/content/drive/My Drive/DiffBasedAutoAnnotation")

input_images=[]

cap = cv2.VideoCapture("vid.mp4")

ret,frame = cap.read()
print(ret)
#frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)

plt.imshow(frame)

img = cv2.resize(frame, (w_size,h_size))
imgs_original = img

image = img
input_images.append(img)

imgs = np.asarray(input_images)
print(imgs.shape)
imgs = np.swapaxes(imgs,1,3)
imgs = torch.from_numpy(imgs)

print(imgs.dtype)

img_batch = imgs.to(dev,dtype=torch.float32)
#mask_batch = masks.to(dev,dtype=torch.float32)

print(img_batch.dtype)

print(img_batch.shape)

mask_pred = net(img_batch)

array = mask_pred.cpu().detach().numpy()
array = np.swapaxes(array,1,3)
print(array.shape)
plt.imshow(array[0,:,:,:])
plt.colorbar()

m = array[0,:,:,:]
m = cv2.cvtColor(m, cv2.COLOR_RGB2GRAY)
#m = m[200<m]
m[m<200] = 0
m[200<m] = 1
plt.imshow(m.astype(np.uint))

img = imgs_original[:,:,0:3]
plt.imshow(img)

img[:,:,0]=img[:,:,0]*m
img[:,:,1]=img[:,:,1]*m
img[:,:,2]=img[:,:,2]*m
plt.imshow(img)

"""# Test"""



os.chdir("/content/drive/My Drive")

entries = os.listdir(reading_path_original)
entries_train=[]
entries_test=[]
for i in range(len(entries)):
  if(373<int((entries[i][6:-4])) and int((entries[i][6:-4]))<874):
    entries_test.append(entries[i])
  else:
    entries_train.append(entries[i])

img_batch, mask_batch, entries_test = get_next_batch(entries_test, batch_size)
img_batch = img_batch.to(dev,dtype=torch.float32)

mask_pred = net(img_batch)

img_batch = np.swapaxes(img_batch,1,3)
input_img = img_batch.cpu().detach().numpy()
plt.imshow(input_img[0,:,:,:]/255)

array = mask_pred.cpu().detach().numpy()
array = np.swapaxes(array,1,3)
print(array.shape)
plt.imshow(array[0,:,:,1])
plt.colorbar()

m = array[0,:,:,1]
#m = cv2.cvtColor(m, cv2.COLOR_RGB2GRAY)
#m = m[200<m]
m[m<200] = 0
m[200<m] = 1
plt.imshow(m.astype(np.uint))
plt.colorbar()

img = input_img[0,:,:,:]/255
img[:,:,0]=img[:,:,0]*m
img[:,:,1]=img[:,:,1]*m
img[:,:,2]=img[:,:,2]*m
plt.imshow(img)

img = input_img[0,:,:,:]/255
plt.imshow(img)

def get_next_batch(entries, batch_length):
    input_images = []
    mask_images = []
    used_in_batch = random.sample(entries, batch_length)
    used_in_batch_mask = []
    for l in range(len(used_in_batch)):
      name = used_in_batch[l].split('_')[1]
      used_in_batch_mask.append("mask_"+name)
    for i in range(len(used_in_batch)):
      img = cv2.imread(reading_path_original+used_in_batch[i])
      #diff = cv2.imread(used_in_batch_mask[i])
      mask = cv2.imread(reading_path_mask+used_in_batch_mask[i])



      img = cv2.resize(img, (w_size,h_size))
      mask = cv2.resize(mask, (w_size,h_size))
      mask = getMask(mask)

      #angle = random.randint(-30,30)
      #img = rotate_image(img,angle)
      #diff = rotate_image(diff,angle)
      #mask = rotate_image(mask,angle)

      #diff = cv2.cvtColor(diff, cv2.COLOR_RGB2GRAY)
      #image = cv2.merge((img, diff))
      image = img
      input_images.append(image)
      #mask_image = cv2.merge((mask, diff))
      mask_image = mask
      mask_images.append(mask_image)

    imgs = np.asarray(input_images)
    masks = np.asarray(mask_images)
    imgs = np.swapaxes(imgs,1,3)
    masks = np.swapaxes(masks,1,3)
    imgs = torch.from_numpy(imgs)
    masks = torch.from_numpy(masks)


    for l in range(len(used_in_batch)):
      entries.remove(used_in_batch[l]);

    return imgs, masks, entries

os.chdir("/content/drive/My Drive")

entries = os.listdir(reading_path_original)
entries_train=[]
entries_test=[]
for i in range(len(entries)):
  if(373<int((entries[i][6:-4])) and int((entries[i][6:-4]))<874):
    entries_test.append(entries[i])
  else:
    entries_train.append(entries[i])

img_batch, mask_batch, entries_test = get_next_batch(entries_test, batch_size)
img_batch = img_batch.to(dev,dtype=torch.float32)

mask_pred = net(img_batch)

img_batch = np.swapaxes(img_batch,1,3)
input_img = img_batch.cpu().detach().numpy()
plt.imshow(input_img[0,:,:,:]/255)

array = mask_pred.cpu().detach().numpy()
array = np.swapaxes(array,1,3)
print(array.shape)
plt.imshow(array[0,:,:,1])
plt.colorbar()

img_batch = np.swapaxes(img_batch,1,3)
input_img = img_batch.cpu().detach().numpy()
plt.imshow(input_img[0,:,:,:]/255)



array = mask_pred.cpu().detach().numpy()
array = np.swapaxes(array,1,3)
print(array.shape)
plt.imshow(array[0,:,:,1])
plt.colorbar()

m = array[0,:,:,1]
#m = cv2.cvtColor(m, cv2.COLOR_RGB2GRAY)
#m = m[200<m]
m[m<200] = 0
m[200<m] = 1
plt.imshow(m.astype(np.uint))
plt.colorbar()

img = input_img[0,:,:,:]/255
img[:,:,0]=img[:,:,0]*m
img[:,:,1]=img[:,:,1]*m
img[:,:,2]=img[:,:,2]*m
plt.imshow(img)

"""# test"""

os.chdir("/content/drive/My Drive")

#frame = cv2.imread("DiffBasedAutoAnnotation/autoAnnonation1/frames_test2/"+"frame_2605.png") # test_sample9.png
frame = cv2.imread("DiffBasedAutoAnnotation/autoAnnonation1/test_sample9.png")
input_images=[]

img = cv2.resize(frame, (w_size,h_size))
imgs_original = img

image = img
input_images.append(img)

imgs = np.asarray(input_images)
imgs = np.swapaxes(imgs,1,3)
imgs = torch.from_numpy(imgs)

print(imgs.dtype)

img_batch = imgs.to(dev,dtype=torch.float32)
#mask_batch = masks.to(dev,dtype=torch.float32)

print(img_batch.dtype)

print(img_batch.shape)

mask_pred = net(img_batch)

array = mask_pred.cpu().detach().numpy()
array = np.swapaxes(array,1,3)
print(array.shape)
array = array[0,:,:,:]
array = cv2.cvtColor(array,cv2.COLOR_RGB2GRAY)
plt.imshow(array)
plt.colorbar()

m = array
#m = cv2.cvtColor(m, cv2.COLOR_RGB2GRAY)
#m = m[200<m]
m[m<200] = 0
m[200<m] = 1
plt.imshow(m.astype(np.uint))
plt.colorbar()

plt.imshow(img)

img = input_images[0][:,:,:]/255
img[:,:,0]=img[:,:,0]*m
img[:,:,1]=img[:,:,1]*m
img[:,:,2]=img[:,:,2]*m
plt.imshow(img)





#frame = cv2.imread("DiffBasedAutoAnnotation/autoAnnonation1/frames_test2/"+"frame_2605.png") # test_sample9.png
frame = cv2.imread("DiffBasedAutoAnnotation/autoAnnonation1/test_sample6.png")
input_images=[]

img = cv2.resize(frame, (w_size,h_size))
imgs_original = img

image = img
input_images.append(img)

imgs = np.asarray(input_images)
imgs = np.swapaxes(imgs,1,3)
imgs = torch.from_numpy(imgs)

print(imgs.dtype)

img_batch = imgs.to(dev,dtype=torch.float32)
#mask_batch = masks.to(dev,dtype=torch.float32)

print(img_batch.dtype)

print(img_batch.shape)

mask_pred = net(img_batch)

array = mask_pred.cpu().detach().numpy()
array = np.swapaxes(array,1,3)
print(array.shape)
array = array[0,:,:,:]
array = cv2.cvtColor(array,cv2.COLOR_RGB2GRAY)
plt.imshow(array)
plt.colorbar()

m = array
#m = cv2.cvtColor(m, cv2.COLOR_RGB2GRAY)
#m = m[200<m]
m[m<200] = 0
m[200<m] = 1
plt.imshow(m.astype(np.uint))
plt.colorbar()

plt.imshow(img)

img = input_images[0][:,:,:]/255
img[:,:,0]=img[:,:,0]*m
img[:,:,1]=img[:,:,1]*m
img[:,:,2]=img[:,:,2]*m
plt.imshow(img)

