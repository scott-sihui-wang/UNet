import torch
import torch.nn as nn
import torch.nn.functional as F

#import any other libraries you need below this line

class twoConvBlock(nn.Module):
  def __init__(self, input_channel, output_channel):
    super(twoConvBlock, self).__init__()
    #todo
    #initialize the block
    self.conv1 = nn.Conv2d(input_channel, output_channel, 3)
    self.relu1 = nn.ReLU()
    self.conv2 = nn.Conv2d(output_channel, output_channel, 3)
    self.bn = nn.BatchNorm2d(output_channel)
    self.relu2 = nn.ReLU()

  def forward(self, x):
    #todo
    #implement the forward path
    x = self.conv1(x)
    x = self.relu1(x)
    x = self.conv2(x)
    x = self.bn(x)
    x = self.relu2(x)
    return x
    

class downStep(nn.Module):
  def __init__(self, input_channel):
    super(downStep, self).__init__()
    #todo
    #initialize the down path
    self.pool = nn.MaxPool2d(2,stride = 2)
    self.conv = twoConvBlock(input_channel, 2 * input_channel)

  def forward(self, x):
    #todo
    #implement the forward path
    x = self.pool(x)
    x = self.conv(x)
    return x

class upStep(nn.Module):
  def __init__(self, input_channel):
    super(upStep, self).__init__()
    #todo
    #initialize the up path
    self.upconv = nn.ConvTranspose2d(input_channel, input_channel//2, 2, stride = 2)
    self.conv = twoConvBlock(input_channel, input_channel//2)

  def forward(self, x, x_down):
    #todo
    #implement the forward path
    x = self.upconv(x)
    x = torch.cat((x, x_down),1)
    x = self.conv(x)
    return x

class UNet(nn.Module):
  def __init__(self, n_classes):
    super(UNet, self).__init__()
    #todo
    #initialize the complete model
    self.input_layer = twoConvBlock(1, 64)
    self.down1 = downStep(64)
    self.down2 = downStep(128)
    self.down3 = downStep(256)
    self.down4 = downStep(512)
    self.crop1 = nn.ZeroPad2d(-88)
    self.crop2 = nn.ZeroPad2d(-40)
    self.crop3 = nn.ZeroPad2d(-16)
    self.crop4 = nn.ZeroPad2d(-4)
    self.up4 = upStep(1024)
    self.up3 = upStep(512)
    self.up2 = upStep(256)
    self.up1 = upStep(128)
    self.output_layer = nn.Conv2d(64, n_classes, 1)

  def forward(self, x):
    #todo
    #implement the forward path
    x1 = self.input_layer(x)
    x2 = self.down1(x1)
    x3 = self.down2(x2)
    x4 = self.down3(x3)
    x5 = self.down4(x4)
    x4 = self.crop4(x4)
    x3 = self.crop3(x3)
    x2 = self.crop2(x2)
    x1 = self.crop1(x1)
    y4 = self.up4(x5, x4)
    y3 = self.up3(y4, x3)
    y2 = self.up2(y3, x2)
    y1 = self.up1(y2, x1)
    x = self.output_layer(y1)        
    return x

