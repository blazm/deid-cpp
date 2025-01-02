import torch
from torch import nn
from dataset import Dataset
from torch.utils import data
from torch.optim.lr_scheduler import StepLR
import sys
sys.path.append("..")
from models.facial_recognition.model_irse import Backbone
import os 
import time
from dataset import get_rafd_list_and_labels
from config import Config
import numpy as np

def save_model(model, save_path, name, iter_cnt):
    save_name = os.path.join(save_path, name + '_' + str(iter_cnt) + '.pth')
    torch.save(model.state_dict(), save_name)
    return save_name

if __name__ == '__main__':

    opt = Config()

    device = torch.device("cuda")

    images_path = '/home/blaz/github/stylegan2-pytorch/datasets/rafd-frontal_aligned/'
    image_list, labels_list = get_rafd_list_and_labels(images_path)
    #labels_list = '' # TODO: read files

    opt.num_classes = len(list(set(labels_list))) # set number of classes dynamically

    model = Backbone(input_size=112, num_layers=50, drop_ratio=0.6, mode='ir_se')

    #model.train()
    model.cuda()
    
    train_dataset = Dataset(images_path, image_list, labels_list, phase='train', input_shape=opt.input_shape)
    trainloader = data.DataLoader(train_dataset,
                                  batch_size=opt.train_batch_size,
                                  shuffle=True,
                                  num_workers=opt.num_workers)

    
    if opt.loss == 'focal_loss':
        from focal_loss import FocalLoss
        criterion = FocalLoss(gamma=2)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    from metrics import AddMarginProduct, ArcMarginProduct, SphereProduct
    if opt.metric == 'add_margin':
        metric_fc = AddMarginProduct(512, opt.num_classes, s=30, m=0.35)
    elif opt.metric == 'arc_margin':
        metric_fc = ArcMarginProduct(512, opt.num_classes, s=30, m=0.5, easy_margin=opt.easy_margin)
    elif opt.metric == 'sphere':
        metric_fc = SphereProduct(512, opt.num_classes, m=4)
    else:
        metric_fc = nn.Linear(512, opt.num_classes) # Softmax?


    # view_model(model, opt.input_shape)
    #print(model)
    model.to(device)
    #model = DataParallel(model)
    metric_fc.to(device)
    #metric_fc = DataParallel(metric_fc)

    #if opt.optimizer == 'sgd':
    #    optimizer = torch.optim.SGD([{'params': model.parameters()}, {'params': metric_fc.parameters()}],
    #                                lr=opt.lr, weight_decay=opt.weight_decay)
    #else:
    optimizer = torch.optim.Adam([{'params': model.parameters()}, {'params': metric_fc.parameters()}],
                                     lr=opt.lr, weight_decay=opt.weight_decay)
    scheduler = StepLR(optimizer, step_size=opt.lr_step, gamma=0.1)

    start = time.time()
    for i in range(opt.max_epoch):
        
        model.train()
        for ii, data in enumerate(trainloader):
            data_input, label = data
            print(data)
            data_input = data_input.to(device)
            label = label.to(device).long()
            print("data_input shape: ",data_input.shape)
            feature = model(data_input)
            output = metric_fc(feature, label)
            loss = criterion(output, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            iters = i * len(trainloader) + ii

            if iters % opt.print_freq == 0:
                output = output.data.cpu().numpy()
                output = np.argmax(output, axis=1)
                label = label.data.cpu().numpy()
                # print(output)
                # print(label)
                acc = np.mean((output == label).astype(int))
                speed = opt.print_freq / (time.time() - start)
                time_str = time.asctime(time.localtime(time.time()))
                print('{} train epoch {} iter {} {} iters/s loss {} acc {}'.format(time_str, i, ii, speed, loss.item(), acc))
                #if opt.display:
                #    visualizer.display_current_results(iters, loss.item(), name='train_loss')
                #    visualizer.display_current_results(iters, acc, name='train_acc')

                start = time.time()

        if i % opt.save_interval == 0 or i == opt.max_epoch:
            save_model(model, opt.checkpoints_path, opt.backbone, i)
        scheduler.step()

        #model.eval()
        #acc = lfw_test(model, img_paths, identity_list, opt.lfw_test_list, opt.test_batch_size)
        #if opt.display:
        #    visualizer.display_current_results(iters, acc, name='test_acc')