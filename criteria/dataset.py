import os
from PIL import Image
import torch
from torch.utils import data
import numpy as np
from torchvision import transforms as T
import torchvision
import cv2
import sys


class Dataset(data.Dataset):

    def __init__(self, root, images_list, labels_list, phase='train', input_shape=(1, 128, 128)):
        self.phase = phase
        self.input_shape = input_shape

        #with open(os.path.join(data_list_file), 'r') as fd:
        #    imgs = fd.readlines()
        self.imgs = images_list
        self.labels = labels_list

        self.imgs = [os.path.join(root, img) for img in self.imgs]
        #self.imgs = np.random.permutation(imgs)

        # normalize = T.Normalize(mean=[0.5, 0.5, 0.5],
        #                         std=[0.5, 0.5, 0.5])

        normalize = T.Normalize(mean=[0.5], std=[0.5])

        if self.phase == 'train':
            self.transforms = T.Compose([
                T.RandomCrop(self.input_shape[1:]),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                normalize
            ])
        else:
            self.transforms = T.Compose([
                T.CenterCrop(self.input_shape[1:]),
                T.ToTensor(),
                normalize
            ])

    def __getitem__(self, index):
        sample = self.imgs[index]
        #splits = sample.split()
        img_path = sample # splits[0]
        data = Image.open(img_path)
        data = data.convert('RGB')
        data = self.transforms(data)
        label = self.labels[index] # np.int32(splits[1])
        return data.float(), label

    def __len__(self):
        return len(self.imgs)

def get_rafd_list_and_labels(images_path):

    file_list = os.listdir(images_path)
    id_list = [int(image_file.split('_')[1]) for image_file in file_list]
    unique_id_list = list(set(id_list))
    range_list = [unique_id_list.index(id) for id in id_list]

    labels = range_list
    return file_list, labels

    with open(pair_list, 'r') as fd:
        pairs = fd.readlines()
    data_list = []
    for pair in pairs:
        splits = pair.split()

        if splits[0] not in data_list:
            data_list.append(splits[0])

        if splits[1] not in data_list:
            data_list.append(splits[1])
    return data_list

if __name__ == '__main__':
    images_path = '/home/blaz/github/stylegan2-pytorch/datasets/rafd-frontal_aligned'
    #from train import get_rafd_list_and_labels
    image_list, labels_list = get_rafd_list_and_labels(images_path)

    dataset = Dataset(root=images_path,
                      images_list=image_list,
                      labels_list=labels_list,
                      phase='test',
                      input_shape=(1, 128, 128))

    trainloader = data.DataLoader(dataset, batch_size=10)
    for i, (data, label) in enumerate(trainloader):
        # imgs, labels = data
        # print imgs.numpy().shape
        # print data.cpu().numpy()
        # if i == 0:
        img = torchvision.utils.make_grid(data).numpy()
        # print img.shape
        # print label.shape
        # chw -> hwc
        img = np.transpose(img, (1, 2, 0))
        # img *= np.array([0.229, 0.224, 0.225])
        # img += np.array([0.485, 0.456, 0.406])
        img += np.array([1, 1, 1])
        img *= 127.5
        img = img.astype(np.uint8)
        img = img[:, :, [2, 1, 0]]

        cv2.imshow('img', img)
        cv2.waitKey()
        # break
        # dst.decode_segmap(labels.numpy()[0], plot=True)