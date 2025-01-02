import os
import argparse

from PIL import Image

import torch
from torchvision import transforms
from torch.nn import functional as F

from .dan import DAN

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=str, help='Image file for evaluation.')
 
    return parser.parse_args()

class Model():
    def __init__(self):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.data_transforms = transforms.Compose([
                                    #transforms.CenterCrop(200), # only facial region
                                    transforms.Resize((224, 224)),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
                                ])
        self.labels = ['neutral', 'happy', 'sad', 'surprise', 'fear', 'disgust', 'anger', 'contempt']

        self.model = DAN(num_head=4, num_class=8)
        #checkpoint = torch.load('./checkpoints/affecnet8_epoch6_acc0.6326.pth',
        #    map_location=self.device)
        checkpoint = torch.load('../pretrained_models/expression/affecnet8_epoch5_acc0.6209.pth',
            map_location=self.device)       
        self.model.load_state_dict(checkpoint['model_state_dict'],strict=True)
        self.model.to(self.device)
        self.model.eval()
    
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
            out, _, _ = self.model(img)
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
            out, _, _ = self.model(img)
            _, pred = torch.max(out,1)
            index = int(pred)
            label = self.labels[index]

            return index, label

    def compute_loss(self, ref_img, img):
        _, _, w, h = img.shape
        tensor_transforms = transforms.Compose([
                                    transforms.CenterCrop(w*0.75), # only facial region
                                    transforms.Resize((224, 224)),
                                    #transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
                                ])
        img = tensor_transforms(img)
        #img = self.data_transforms(img)
        img = img.view(1,3,224,224)
        #img = img[:, :, 35:223, 32:220]  # Crop interesting region
        img = img.to(self.device)

        ref_img = tensor_transforms(img)
        ref_img = ref_img.view(1,3,224,224)
        #img = img[:, :, 35:223, 32:220]  # Crop interesting region
        ref_img = ref_img.to(self.device)

        with torch.set_grad_enabled(True):
            #img_out, _, _ = self.model(img)
            #ref_img_out, _, _ = self.model(ref_img)

            # if computing loss on features
            img_out = self.model.extract_features(img)
            #print("Expression feature shape: ", img_out.shape)
            ref_img_out = self.model.extract_features(ref_img)

            return F.mse_loss(img_out, ref_img_out)

    def fit(self, path):
        img = Image.open(path).convert('RGB')
        width, height = img.size
        crop = transforms.CenterCrop(int(width*0.75)) # this could be parameter, estimated for each model (based on croppings of training data)
        img = crop(img)
        img = self.data_transforms(img)
        img = img.view(1,3,224,224)
        #img = img[:, :, 35:223, 32:220]  # Crop interesting region
        img = img.to(self.device)

        with torch.set_grad_enabled(False):
            out, _, _ = self.model(img)
            _, pred = torch.max(out,1)
            index = int(pred)
            label = self.labels[index]

            return index, label

if __name__ == "__main__":
    args = parse_args()

    model = Model()

    image = args.image
    assert os.path.exists(image)

    index, label = model.fit(image)

    print(f'emotion label: {label}')