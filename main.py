from dataset import VOCDataSet
import matplotlib.pyplot as plt
import torch
from torch.utils import data
from torchvision.utils import make_grid
import PIL.Image as Image
from transform import ReLabel, ToLabel, ToSP, Scale, Colorize
from torchvision.transforms import Compose, Normalize, ToTensor
from myfunc import make_image_grid, make_label_grid
from criterion import CrossEntropyLoss2d
from resnet import resnet101
from torch.autograd.variable import Variable
import gc
from model import Seg
import torch.nn.functional as F
import pdb
from tensorboard import SummaryWriter
import os
from datetime import datetime
import torchvision

os.system('rm -rf ./runs/*')
writer = SummaryWriter('./runs/'+datetime.now().strftime('%B%d  %H:%M:%S'))

std = [.229, .224, .225]
mean = [.485, .456, .406]

input_transform = Compose([
    Scale((256, 256), Image.BILINEAR),
    ToTensor(),
    Normalize(mean, std),

])
target_transform = Compose([
    Scale((256, 256), Image.NEAREST),
    ToSP(256),
    ToLabel(),
    ReLabel(255, 21),
])


loader = data.DataLoader(VOCDataSet("/home/zeng/data/datasets/segmentation_Dataset",
                                    img_transform=input_transform,
                                    label_transform=target_transform),
                                batch_size=12, shuffle=True, pin_memory=True)

res101 = resnet101(pretrained=True).cuda()
seg = Seg().cuda()

weight = torch.ones(22)
weight[21] = 0

criterion = CrossEntropyLoss2d(weight.cuda())

optimizer_seg = torch.optim.Adam(seg.parameters(), lr=1e-3)
optimizer_feat = torch.optim.Adam(res101.parameters(), lr=1e-4)

for t in range(10):
    for i, (img, label) in enumerate(loader):
        img = img.cuda()
        label = label[0].cuda()
        label = Variable(label)
        input = Variable(img)

        feats = res101(input)
        output = seg(feats)

        seg.zero_grad()
        res101.zero_grad()
        loss = criterion(output, label)
        loss.backward()
        optimizer_feat.step()
        optimizer_seg.step()

        ## see
        input = make_image_grid(img, mean, std)
        label = make_label_grid(label.data)
        label = Colorize()(label).type(torch.FloatTensor)
        output = make_label_grid(torch.max(output, dim=1)[1].data)
        output = Colorize()(output).type(torch.FloatTensor)
        writer.add_image('image', input, i)
        writer.add_image('label', label, i)
        writer.add_image('pred', output, i)
        writer.add_scalar('loss', loss.data[0], i)

        print ("epoch %d step %d, loss=%.4f" %(t, i, loss.data.cpu()[0]))
