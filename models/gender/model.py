import os
import argparse

from PIL import Image

import torch
from torch.nn import functional as F
from torchvision import transforms, models
from torch import nn

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=str, help='Image file for evaluation.')
 
    return parser.parse_args()

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x

class Model():
    def __init__(self, checkpoint_path, only_features=False):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        self.model = models.resnet18(pretrained=True)
        self.num_features = self.model.fc.in_features
        if not only_features:
            self.model.fc = nn.Linear(self.num_features, 2) # binary classification (num_of_class == 2)
        self.model.load_state_dict(torch.load(checkpoint_path), strict = False if only_features else True)
        
        #if only_features:
        #self.model.fc = Identity()
        self.model.to(self.device)
        self.model.eval()


        
        self.feat_model = models.resnet18(pretrained=True)
        self.num_features = self.feat_model.fc.in_features
        if not only_features:
            self.feat_model.fc = nn.Linear(self.num_features, 2) # binary classification (num_of_class == 2)
        self.feat_model.load_state_dict(torch.load(checkpoint_path), strict = False if only_features else True)
        
        #if only_features:
        self.feat_model.fc = Identity()
        self.feat_model.to(self.device)
        self.feat_model.eval()
        
        #transforms_val = transforms.Compose([
        #	transforms.Resize((224, 224)),
        #	transforms.ToTensor(),
        #	transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        #])
        
        self.data_transforms = transforms.Compose([
                                    #transforms.CenterCrop(200), # only facial region
                                    transforms.Resize((224, 224)),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
                                ])
        self.labels =  ['female', 'male']
        #self.labels = ['neutral', 'happy', 'sad', 'surprise', 'fear', 'disgust', 'anger', 'contempt']
        #self.model = DAN(num_head=4, num_class=8)
        #checkpoint = torch.load('./checkpoints/affecnet8_epoch6_acc0.6326.pth',
        #    map_location=self.device)
        #checkpoint = torch.load('../../pretrained_models/expression/affecnet8_epoch5_acc0.6209.pth',
        #    map_location=self.device)       
        #self.model.load_state_dict(checkpoint['model_state_dict'],strict=True)
        #self.model.to(self.device)
        #self.model.eval()
    '''
    def extract_features(self, img):
        tensor_transforms = transforms.Compose([
                                    #transforms.CenterCrop(w*0.75), # only facial region
                                    transforms.Resize((224, 224)),
                                    #transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
        img = self.data_transforms(img)
        img = img.view(1,3,224,224)
        #img = img[:, :, 35:223, 32:220]  # Crop interesting region
        img = img.to(self.device)

        with torch.set_grad_enabled(False):
            feats = self.feat_model(img)
            return feats
    '''

    def predict_on_tensor(self, img):
        _, _, w, h = img.shape
        tensor_transforms = transforms.Compose([
                                    #transforms.CenterCrop(w*0.75), # only facial region
                                    transforms.Resize((224, 224)),
                                    #transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
                                ])
        img = tensor_transforms(img)
        img = img.view(1,3,224,224)
        #img = img[:, :, 35:223, 32:220]  # Crop interesting region
        img = img.to(self.device)

        with torch.set_grad_enabled(False):
            out = self.model(img)
            _, pred = torch.max(out,1)
            index = int(pred)
            label = self.labels[index]

            return index, label

    def predict(self, img):
        img = self.data_transforms(img)
        img = img.view(1,3,224,224)
        #img = img[:, :, 35:223, 32:220]  # Crop interesting region
        img = img.to(self.device)

        with torch.set_grad_enabled(False):
            out = self.model(img)
            _, pred = torch.max(out,1)
            index = int(pred)
            label = self.labels[index]

            return index, label

    def compute_loss(self, ref_img, img):
        _, _, w, h = img.shape
        tensor_transforms = transforms.Compose([
                                    #transforms.CenterCrop(w*0.75), # only facial region
                                    transforms.Resize((224, 224)),
                                    #transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
                                ])
        img = tensor_transforms(img)
        img = img.view(1,3,224,224)
        #img = img[:, :, 35:223, 32:220]  # Crop interesting region
        img = img.to(self.device)

        ref_img = tensor_transforms(img)
        ref_img = ref_img.view(1,3,224,224)
        #img = img[:, :, 35:223, 32:220]  # Crop interesting region
        ref_img = ref_img.to(self.device)

        with torch.set_grad_enabled(True):
            img_out = self.feat_model(img)
            #print("Gender feature shape: ", img_out.shape)
            ref_img_out = self.feat_model(ref_img)

            return F.mse_loss(img_out, ref_img_out)

    def fit(self, path):
        img = Image.open(path).convert('RGB')
        width, height = img.size
        crop = transforms.CenterCrop(int(width*0.75))
        img = crop(img)
        img = self.data_transforms(img)
        img = img.view(1,3,224,224)
        #img = img[:, :, 35:223, 32:220]  # Crop interesting region
        img = img.to(self.device)

        with torch.set_grad_enabled(False):
            out = self.model(img)
            _, pred = torch.max(out,1)
            index = int(pred)
            label = self.labels[index]

            return index, label

if __name__ == "__main__":
    args = parse_args()

    checkpoint_path = '../../pretrained_models/gender/face_gender_classification_transfer_learning_with_ResNet18.pth'
    model = Model()

    image = args.image
    assert os.path.exists(image)

    index, label = model.fit(image)

    print(f'gender label: {label}')