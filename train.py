import yaml
import os
import numpy as np
import cv2
import torchvision.models.segmentation
import torch
import torchvision.transforms as tf
import random
import model.py

with open("config.yml") as fp:
    cfg = yaml.safe_load(fp)

batchSize=cfg["training"].get("batchSize", 3)

TrainFolder=cfg["data"].get("path")

height = cfg["image"].get("height",450)
width = cfg["image"].get("width",450)

ListImages=os.listdir(os.path.join(TrainFolder, "lytro-img/A")) # Create list of images
#----------------------------------------------Transform image-------------------------------------------------------------------
transformImg=tf.Compose([tf.ToPILImage(),tf.Resize((height,width)),tf.ToTensor(),tf.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
transformAnn=tf.Compose([tf.ToPILImage(),tf.Resize((height,width),tf.InterpolationMode.NEAREST),tf.ToTensor()])
conv_prelayer=torch.nn.Conv2d(6, 3, 3, 1, 1)
#---------------------Read image ---------------------------------------------------------
def ReadRandomImage(): # First lets load random image and  the corresponding annotation
    idx=np.random.randint(0,len(ListImages)) # Select random image
    Img1=cv2.imread(os.path.join(TrainFolder, "lytro-img/A", ListImages[idx]).replace("._", ""))[:,:,0:3]
    Img2=cv2.imread(os.path.join(TrainFolder, "lytro-img/B", ListImages[idx]).replace("A", "B").replace("._", ""))[:,:,0:3]
    AnnMapBuf = cv2.imread(os.path.join(TrainFolder, "lytro-trimap", ListImages[idx].replace("A", "trimap").replace("._", "")))[:,:,0]
    AnnMap = np.zeros(AnnMapBuf.shape[0:2],np.float32)
    if AnnMapBuf is not None:  AnnMap[AnnMapBuf >= 191] = 2
    if AnnMapBuf is not None:  AnnMap[AnnMapBuf < 191] = 1
    if AnnMapBuf is not None:  AnnMap[AnnMapBuf <= 64] = 0
    Img1=transformImg(Img1)
    Img2=transformImg(Img2)
    Img3=torch.zeros([2,3, height, width])
    Img3[0,:,:,:]=Img1
    Img3[1,:,:,:]=Img2
    # Img=conv_prelayer(Img3)
    AnnMap=transformAnn(AnnMap)
    return Img3,AnnMap
#--------------Load batch of images-----------------------------------------------------
def LoadBatch(): # Load batch of images
    images = torch.zeros([batchSize,3,height,width])
    ann = torch.zeros([batchSize, height, width])
    for i in range(batchSize):
        images[i],ann[i]=ReadRandomImage()
    return images, ann
#--------------Load and set net and optimizer-------------------------------------
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
torch.cuda.empty_cache()
Net = torchvision.models.segmentation.deeplabv3_resnet50(pretrained=True) # Load net
Net.classifier[4] = torch.nn.Conv2d(256, 3, kernel_size=(1, 1), stride=(1, 1)) # Change final layer to 3 classes
Net=Net.to(device)
optimizer=torch.optim.Adam(params=Net.parameters(),lr=cfg["training"].get("Learning_Rate", 1e-5)) # Create adam optimizer
#----------------Train--------------------------------------------------------------------------
for itr in range(10001): # Training loop
   images,ann=LoadBatch() # Load taining batch
   images=torch.autograd.Variable(images,requires_grad=False).to(device) # Load image
   ann = torch.autograd.Variable(ann, requires_grad=False).to(device) # Load annotation
   Pred=Net(images)['out'] # make prediction
   Net.zero_grad()
   criterion = torch.nn.CrossEntropyLoss() # Set loss function
   Loss=criterion(Pred,ann.long()) # Calculate cross entropy loss
   Loss.backward() # Backpropogate loss
   optimizer.step() # Apply gradient descent change to weight
   seg = torch.argmax(Pred[0], 0).cpu().detach().numpy()  # Get  prediction classes
   print(itr,") Loss=",Loss.data.cpu().numpy())
   if itr % 100 == 0:
        print("Saving image" +str(itr))
        cv2.imwrite("img"+str(itr)+".jpg",(seg*127).astype(np.uint8))  # save image
   if itr % 1000 == 0: #Save model weight once every 1k steps permenant file
        print("Saving Model" +str(itr) + ".torch")
        torch.save(Net.state_dict(),   str(itr) + ".torch")
