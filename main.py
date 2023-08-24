import os
import cv2
import numpy as np
import torchvision.models.segmentation
import torch
import torchvision.transforms as tf
import yaml
import model
from normalize import normalize_pair as normalize

with open("config.yml") as fp:
    cfg = yaml.safe_load(fp)

modelPath = "9000.torch"  # Path to trained model
ImgFolder = "test"  # Test image
ImgN = 0
height=cfg["image"].get("height",450)
width=cfg["image"].get("width",450)
ListImages=os.listdir(os.path.join(ImgFolder,"A")) # Create list of images
(height_orgin , widh_orgin ,d) = cv2.imread(os.path.join(ImgFolder,"A", ListImages[ImgN]).replace("._", ""))[:,:,0:3].shape # Get image original size 

def imageLoader():
    Img1=cv2.imread(os.path.join(ImgFolder,"A", ListImages[ImgN]).replace("._", ""))[:,:,0:3]
    Img2=cv2.imread(os.path.join(ImgFolder,"B", ListImages[ImgN]).replace("A", "B").replace("._", ""))[:,:,0:3]
    print(os.path.join(ImgFolder,"A", ListImages[ImgN]).replace("._", ""))
    mean, std = normalize(Img1, Img2)
    transformImg = tf.Compose([tf.ToPILImage(), tf.Resize((height, width)), tf.ToTensor(),tf.Normalize(mean, std)])
    Img1=transformImg(Img1)
    Img2=transformImg(Img2)
    Img3=torch.zeros([1,6, height, width])
    Img3[0,0:3,:,:]=Img1
    Img3[0,3:6,:,:]=Img2
    return Img3.squeeze()


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')  # Check if there is GPU if not set trainning to CPU (very slow)
Net = model.fcn() # Load net
Net = Net.to(device)  # Set net to GPU or CPU
Net.load_state_dict(torch.load(modelPath)) # Load trained model
Net.eval() # Set to evaluation mode
Img = imageLoader()  # Load and transform to pytorch
Img = torch.autograd.Variable(Img, requires_grad=False).to(device).unsqueeze(0)
with torch.no_grad():
    Prd = Net(Img)  # Run net 
Prd = tf.Resize((height_orgin,widh_orgin))(Prd[0]) # Resize to origninal size
seg = torch.argmax(Prd, 0).cpu().detach().numpy()  # Get  prediction classes
cv2.imwrite("HERE.jpg",(seg*127).astype(np.uint8))  # display image
print("DONE")