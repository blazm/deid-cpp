import os
import argparse

import math
import cv2
import numpy as np
from PIL import Image

import torch
from torchvision import transforms, models
from torch import nn

def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('--image', type=str, help='Image file for evaluation.')
 
	return parser.parse_args()
	
from models.pose.detect import AntiSpoofPredict
from models.pose.pfld import PFLDInference #, AuxiliaryNet

from models.pose.utils import *
from models.pose.video import point_point, point_line, cross_point, get_num

class Model():
	def __init__(self, checkpoint_path, only_features=False):
		self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
		
		self.detector = AntiSpoofPredict(0, checkpoint_path)
		
		checkpoint_full_path = os.path.join(checkpoint_path + "snapshot/checkpoint.pth.tar")
		#print(checkpoint_path, checkpoint_full_path)
		checkpoint = torch.load(checkpoint_full_path, map_location=self.device)
		self.plfd_backbone = PFLDInference().to(self.device)
		self.plfd_backbone.load_state_dict(checkpoint['plfd_backbone'])
		self.plfd_backbone.eval()
		self.plfd_backbone = self.plfd_backbone.to(self.device)
		self.transform = transforms.Compose([transforms.ToTensor()])
		
		

		
		#self.feat_model = models.resnet18(pretrained=True)
		#self.feat_model.load_state_dict(torch.load(checkpoint_path), strict=False)
		#self.feat_model.to(self.device)
		#self.feat_model.eval()
		#transforms_val = transforms.Compose([
		#	transforms.Resize((224, 224)),
		#	transforms.ToTensor(),
		#	transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
		#])
		'''
		self.data_transforms = transforms.Compose([
									#transforms.CenterCrop(200), # only facial region
									transforms.Resize((224, 224)),
									transforms.ToTensor(),
									transforms.Normalize(mean=[0.485, 0.456, 0.406],
									std=[0.229, 0.224, 0.225])
								])
		self.labels =  ['female', 'male']
		'''
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
		img = self.data_transforms(img)
		img = img.view(1,3,224,224)
		#img = img[:, :, 35:223, 32:220]  # Crop interesting region
		img = img.to(self.device)

		with torch.set_grad_enabled(False):
			feats = self.feat_model(img)
			return feats
	'''

	def predict_on_numpy(self, img):
		_, w, h = img.shape 
		tensor_transforms = transforms.Compose([
									transforms.CenterCrop(w*0.75), # only facial region
									transforms.Resize((224, 224)),
									#transforms.ToTensor(),
									transforms.Normalize(mean=[0.485, 0.456, 0.406],
									std=[0.229, 0.224, 0.225])
								])
		#img = tensor_transforms(img)
		#img = img.view(1,3,224,224)
		#img = img[:, :, 35:223, 32:220]  # Crop interesting region
		#img = img.to(self.device)

		with torch.set_grad_enabled(False):

			height, width = img.shape[:2]
			#model_test = AntiSpoofPredict(args.device_id)
			image_bbox = self.detector.get_bbox(img)
			x1 = image_bbox[0]
			y1 = image_bbox[1]
			x2 = image_bbox[0] + image_bbox[2]
			y2 = image_bbox[1] + image_bbox[3]
			w = x2 - x1
			h = y2 - y1

			size = int(max([w, h]))
			cx = x1 + w/2
			cy = y1 + h/2
			x1 = cx - size/2
			x2 = x1 + size
			y1 = cy - size/2
			y2 = y1 + size

			dx = max(0, -x1)
			dy = max(0, -y1)
			x1 = max(0, x1)
			y1 = max(0, y1)

			edx = max(0, x2 - width)
			edy = max(0, y2 - height)
			x2 = min(width, x2)
			y2 = min(height, y2)

			cropped = img[int(y1):int(y2), int(x1):int(x2)]
			if (dx > 0 or dy > 0 or edx > 0 or edy > 0):
				cropped = cv2.copyMakeBorder(cropped, dy, edy, dx, edx, cv2.BORDER_CONSTANT, 0)
				
			cropped = cv2.resize(cropped, (112, 112))

			input = cv2.resize(cropped, (112, 112))
			input = cv2.cvtColor(input, cv2.COLOR_BGR2RGB)
			input = self.transform(input).unsqueeze(0).to(self.device)
			_, landmarks = self.plfd_backbone(input)
			pre_landmark = landmarks[0]
			pre_landmark = pre_landmark.cpu().detach().numpy().reshape(-1, 2) * [112, 112]
			point_dict = {}
			i = 0
			for (x,y) in pre_landmark.astype(np.float32):
				point_dict[f'{i}'] = [x,y]
				i += 1

			#yaw
			point1 = [get_num(point_dict, 1, 0), get_num(point_dict, 1, 1)]
			point31 = [get_num(point_dict, 31, 0), get_num(point_dict, 31, 1)]
			point51 = [get_num(point_dict, 51, 0), get_num(point_dict, 51, 1)]
			crossover51 = point_line(point51, [point1[0], point1[1], point31[0], point31[1]])
			yaw_mean = point_point(point1, point31) / 2
			yaw_right = point_point(point1, crossover51)
			yaw = (yaw_mean - yaw_right) / yaw_mean
			yaw = int(yaw * 71.58 + 0.7037)

			#pitch
			pitch_dis = point_point(point51, crossover51)
			if point51[1] < crossover51[1]:
				pitch_dis = -pitch_dis
			pitch = int(1.497 * pitch_dis + 18.97)

			#roll
			roll_tan = abs(get_num(point_dict,60,1) - get_num(point_dict,72,1)) / abs(get_num(point_dict,60,0) - get_num(point_dict,72,0))
			roll = math.atan(roll_tan)
			roll = math.degrees(roll)
			if get_num(point_dict, 60, 1) > get_num(point_dict, 72, 1):
				roll = -roll
			roll = int(roll)
			
			return yaw, pitch, roll

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

	def fit(self, path):
		'''
		img = Image.open(path).convert('RGB')
		width, height = img.size
		crop = transforms.CenterCrop(int(width*0.75))
		img = crop(img)
		img = self.data_transforms(img)
		img = img.view(1,3,224,224)
		#img = img[:, :, 35:223, 32:220]  # Crop interesting region
		img = img.to(self.device)
		'''

		with torch.set_grad_enabled(False):
			
			img = cv2.imread(path)

			height, width = img.shape[:2]
			#model_test = AntiSpoofPredict(args.device_id)
			image_bbox = self.detector.get_bbox(img)
			x1 = image_bbox[0]
			y1 = image_bbox[1]
			x2 = image_bbox[0] + image_bbox[2]
			y2 = image_bbox[1] + image_bbox[3]
			w = x2 - x1
			h = y2 - y1

			size = int(max([w, h]))
			cx = x1 + w/2
			cy = y1 + h/2
			x1 = cx - size/2
			x2 = x1 + size
			y1 = cy - size/2
			y2 = y1 + size

			dx = max(0, -x1)
			dy = max(0, -y1)
			x1 = max(0, x1)
			y1 = max(0, y1)

			edx = max(0, x2 - width)
			edy = max(0, y2 - height)
			x2 = min(width, x2)
			y2 = min(height, y2)

			cropped = img[int(y1):int(y2), int(x1):int(x2)]
			if (dx > 0 or dy > 0 or edx > 0 or edy > 0):
				#print(edx)
				cropped = cv2.copyMakeBorder(cropped, int(dy), int(edy), int(dx), int(edx), cv2.BORDER_CONSTANT, 0)

			cropped = cv2.resize(cropped, (112, 112))

			input = cv2.resize(cropped, (112, 112))
			input = cv2.cvtColor(input, cv2.COLOR_BGR2RGB)
			input = self.transform(input).unsqueeze(0).to(self.device)
			_, landmarks = self.plfd_backbone(input)
			pre_landmark = landmarks[0]
			pre_landmark = pre_landmark.cpu().detach().numpy().reshape(-1, 2) * [112, 112]
			point_dict = {}
			i = 0
			for (x,y) in pre_landmark.astype(np.float32):
				point_dict[f'{i}'] = [x,y]
				i += 1

			#yaw
			point1 = [get_num(point_dict, 1, 0), get_num(point_dict, 1, 1)]
			point31 = [get_num(point_dict, 31, 0), get_num(point_dict, 31, 1)]
			point51 = [get_num(point_dict, 51, 0), get_num(point_dict, 51, 1)]
			crossover51 = point_line(point51, [point1[0], point1[1], point31[0], point31[1]])
			yaw_mean = point_point(point1, point31) / 2
			yaw_right = point_point(point1, crossover51)
			yaw = (yaw_mean - yaw_right) / yaw_mean
			yaw = int(yaw * 71.58 + 0.7037)

			#pitch
			pitch_dis = point_point(point51, crossover51)
			if point51[1] < crossover51[1]:
				pitch_dis = -pitch_dis
			pitch = int(1.497 * pitch_dis + 18.97)

			#roll
			roll_tan = abs(get_num(point_dict,60,1) - get_num(point_dict,72,1)) / abs(get_num(point_dict,60,0) - get_num(point_dict,72,0))
			roll = math.atan(roll_tan)
			roll = math.degrees(roll)
			if get_num(point_dict, 60, 1) > get_num(point_dict, 72, 1):
				roll = -roll
			roll = int(roll)
			
			return yaw, pitch, roll

if __name__ == "__main__":
	args = parse_args()

	model = Model(checkpoint_path='../../pretrained_models/pose/')
	#print(model.plfd_backbone.summary())
	#image = args.image
	#assert os.path.exists(image)

	#index, label = model.fit(image)

	#print(f'gender label: {label}')