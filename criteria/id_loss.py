import torch
from torch import nn

from models.facial_recognition.model_irse import Backbone


class IDLoss(nn.Module):
    def __init__(self, opts):
        super(IDLoss, self).__init__()
        print('Loading ResNet ArcFace (ID)')
        self.facenet = Backbone(input_size=112, num_layers=50, drop_ratio=0.6, mode='ir_se')
        self.facenet.load_state_dict(torch.load(opts.ir_se50_weights))
        self.pool = torch.nn.AdaptiveAvgPool2d((256, 256))
        self.face_pool = torch.nn.AdaptiveAvgPool2d((112, 112))
        self.facenet.eval()
        self.facenet.cuda()
        self.opts = opts

    def extract_feats(self, x):
        if x.shape[2] != 256:
            x = self.pool(x)
        x = x[:, :, 35:223, 32:220]  # Crop interesting region
        x = self.face_pool(x)
        x_feats = self.facenet(x)
        return x_feats

    def forward(self, y_hat, y, y_feats=None):
        n_samples = y.shape[0]
        if y_feats is None:
            y_feats = self.extract_feats(y)  # Otherwise use the feature from there
            y_feats = y_feats.detach()
        y_hat_feats = self.extract_feats(y_hat)
        loss = 0
        sim_improvement = 0
        count = float(n_samples)
        loss_para = torch.flatten(y_hat_feats).dot(torch.flatten(y_feats)).sum() # to maximize
        #loss_para = (1.0 - torch.flatten(y_hat_feats).dot(torch.flatten(y_feats))).sum() # to minimize
        '''
        for i in range(n_samples):
            diff_target = y_hat_feats[i].dot(y_feats[i])
            #loss += 1 - diff_target # original # minimizes the difference between id features
            loss += diff_target # maximizes the difference
            count += 1
        print("Loss comparisons: ", loss, loss_para)
        '''
        return loss_para / count, sim_improvement / count


class GenderLoss(nn.Module):
    def __init__(self, opts):
        super(GenderLoss, self).__init__()
        print('Loading ResNet ArcFace (GENDER)')
        self.facenet = Backbone(input_size=112, num_layers=50, drop_ratio=0.6, mode='ir_se')
        self.facenet.load_state_dict(torch.load(opts.ir_se50_weights_gender))
        self.pool = torch.nn.AdaptiveAvgPool2d((256, 256))
        self.face_pool = torch.nn.AdaptiveAvgPool2d((112, 112))
        self.facenet.eval()
        self.facenet.cuda()
        self.opts = opts

    def extract_feats(self, x):
        if x.shape[2] != 256:
            x = self.pool(x)
        #x = x[:, :, 35:223, 32:220]  # Crop interesting region
        x = self.face_pool(x)
        x_feats = self.facenet(x)
        return x_feats

    def forward(self, y_hat, y, y_feats=None):
        n_samples = y.shape[0]
        if y_feats is None:
            y_feats = self.extract_feats(y)  # Otherwise use the feature from there
            y_feats = y_feats.detach()
        y_hat_feats = self.extract_feats(y_hat)
        loss = 0
        sim_improvement = 0
        count = float(n_samples)
        loss = (1.0 - torch.flatten(y_hat_feats).dot(torch.flatten(y_feats))).sum() 
        '''
        for i in range(n_samples):
            diff_target = y_hat_feats[i].dot(y_feats[i])
            loss += 1 - diff_target # original # minimizes the difference between id features
            #loss += diff_target # maximizes the difference
            
            count += 1
        '''
        return loss / count, sim_improvement / count

class ExpressionLoss(nn.Module):
    def __init__(self, opts):
        super(ExpressionLoss, self).__init__()
        print('Loading ResNet ArcFace (EXPRESSION)')
        self.facenet = Backbone(input_size=112, num_layers=50, drop_ratio=0.6, mode='ir_se')
        self.facenet.load_state_dict(torch.load(opts.ir_se50_weights_expression))
        self.pool = torch.nn.AdaptiveAvgPool2d((256, 256))
        self.face_pool = torch.nn.AdaptiveAvgPool2d((112, 112))
        self.facenet.eval()
        self.facenet.cuda()
        self.opts = opts

    def extract_feats(self, x):
        if x.shape[2] != 256:
            x = self.pool(x)
        #x = x[:, :, 35:223, 32:220]  # Crop interesting region # double check if this is required for gender and expression
        x = self.face_pool(x)
        x_feats = self.facenet(x)
        return x_feats

    def forward(self, y_hat, y, y_feats=None):
        n_samples = y.shape[0]
        if y_feats is None:
            y_feats = self.extract_feats(y)  # Otherwise use the feature from there
            y_feats = y_feats.detach()
        y_hat_feats = self.extract_feats(y_hat)
        loss = 0
        sim_improvement = 0
        count = float(n_samples)
        loss = (1.0 - torch.flatten(y_hat_feats).dot(torch.flatten(y_feats))).sum() 
        '''
        for i in range(n_samples):
            diff_target = y_hat_feats[i].dot(y_feats[i])
            loss += 1 - diff_target # original # minimizes the difference between id features
            #loss += diff_target # maximizes the difference
            
            count += 1
        '''
        return loss / count, sim_improvement / count