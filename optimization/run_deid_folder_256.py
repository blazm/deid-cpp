import argparse
from email.mime import image
import math
from multiprocessing import reduction
import os
from cv2 import exp

import torch
import torchvision
from torch import device, long, optim
from tqdm import tqdm
from PIL import Image
from torchvision import transforms as T

import sys
sys.path.append(".")
sys.path.append("..")
import numpy as np
import random

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False # set True to be faster
    print('Setting all seeds to be {seed} to reproduce...'.format(seed=seed))
seed_everything(42) # reproducibility

from criteria.id_loss import IDLoss
from mapper.training.train_utils import STYLESPACE_DIMENSIONS
from models.stylegan2.model import Generator
from utils import ensure_checkpoint_exists

def ensure_dir(d):
    if not os.path.exists(d):
        os.makedirs(d)

STYLESPACE_INDICES_WITHOUT_TORGB = [i for i in range(len(STYLESPACE_DIMENSIONS)) if i not in list(range(1, len(STYLESPACE_DIMENSIONS), 3))]

def get_lr(t, initial_lr, rampdown=0.25, rampup=0.05):
    lr_ramp = min(1, (1 - t) / rampdown)
    lr_ramp = 0.5 - 0.5 * math.cos(lr_ramp * math.pi)
    lr_ramp = lr_ramp * min(1, t / rampup)

    return initial_lr * lr_ramp

def main(args):
    import os
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # GPU selection, this selects second GPU #see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"]="1"

    import sys

    platform = sys.platform
    ablation_arguments = args.ablation
    dataset_name = args.dataset
    experiment_label = args.experiment_label
    
    from models.gender.model import Model as GenderModel
    gender_model = GenderModel('../pretrained_models/gender/face_gender_classification_transfer_learning_with_ResNet18.pth')
    
    from models.expression.model import Model as ExpressionModel
    expression_model = ExpressionModel()
    
    #from models.pose.model import Model as PoseModel # pose not used at the moment
    #pose_model = PoseModel(checkpoint_path='../pretrained_models/pose/')

    ensure_checkpoint_exists(args.ckpt)
    os.makedirs(args.results_dir, exist_ok=True)

    g_ema = Generator(args.stylegan_size, 512, 8)
    g_ema.load_state_dict(torch.load(args.ckpt)["g_ema"], strict=False)
    g_ema.eval()
    g_ema = g_ema.cuda()
    mean_latent = g_ema.mean_latent(4096)

    def get_n_params(model):
        pp=0
        for p in list(model.parameters()):
            nn=1
            for s in list(p.size()):
                nn = nn*s
            pp += nn
        return pp
    
    # set dataset paths here to process the whole folder of images
    dataset_path = '../../DAN/datasets/AffectNet/val_set/aligned_images'
    # encodings into stylegan latent space must be precomputed and put into encodings path for each dataset
    encodings_path = '../../DAN/datasets/AffectNet/val_set/projections_hyperstyle'
    dataset_save = '../../stylegan2-pytorch/datasets/results/AffectNet_cpp-deid'
    
    # filetype conversion if needed
    dataset_filetype = 'jpg'
    #dataset_filetype = 'png'
    dataset_newtype = 'jpg'

    img_names = [i for i in os.listdir(dataset_path) if dataset_filetype in i] # change ppm into jpg
    img_names = sorted(img_names)
    img_paths = [os.path.join(dataset_path, i) for i in img_names]
    save_paths = [os.path.join(dataset_save, i.replace(dataset_filetype, dataset_newtype)) for i in img_names]
    encoding_paths = [os.path.join(encodings_path, i.replace(dataset_filetype, 'pt')) for i in img_names]

    ensure_dir(dataset_save)
    
    from torch.nn import MSELoss   
    from tqdm import tqdm

    mse_loss = MSELoss().cuda()    
    id_loss = IDLoss(args).cuda()
    
    print("Generator params: ", get_n_params(g_ema))
    print("Identity params: ", get_n_params(id_loss.facenet))
    print("Expression params: ", get_n_params(expression_model.model))
    print("Gender params: ", get_n_params(gender_model.feat_model))
    #exit()

    ensure_dir('./log')
    from tensorboardX import SummaryWriter
    enable_logging = False # set to True to save resulting data to tensorboard
    writer = SummaryWriter('./log')

    for ix, (image_path, output_path, encoding_path) in tqdm(enumerate(zip(img_paths, save_paths, encoding_paths)), total=len(img_names)):
    
        image_name = os.path.split(image_path)[1]
        
        output_path_tsim = os.path.join(dataset_save, str(0.1), image_name.replace(dataset_filetype, dataset_newtype))
        
        if os.path.exists(output_path_tsim): # skip if image already generated on last tsim, to save time
            print(image_name, " exists in ", 0.0, " skipping.")
            continue

        if args.latent_path:
            latent_code_init = torch.load(args.latent_path).cuda()
        elif args.mode == "edit":
            # load from folder file by file if in edit mode - this is the default mode
            latent_code_init = torch.load(encoding_path).cuda()
        else:
            latent_code_init = mean_latent.detach().clone().repeat(1, 18, 1)

        def load_image(img_path):
            transforms = T.Compose([
                T.Resize((256, 256)),
                T.ToTensor(),
            ])

            data = Image.open(img_path)
            data = data.convert('RGB')
            data = transforms(data)
            return data

        img2gan_normalize = T.Compose([T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])]);
        def gan2img_normalize(img):
            return (img - img.min()) / (img.max() - img.min()) 

        img_orig = load_image(image_path).unsqueeze(0).cuda()
        print("Path: ", image_name) #, " GENDER: ", gender_lbl, " EXPRE: ", expression_lbl)
        
        orig_id_feats = id_loss.extract_feats(img_orig).detach() 
        
        pbar = (range(args.step))

        im_samples = []
        id_scores = []

        target_similarities = [1.0] # closest to the original image
        if 'ID' in ablation_arguments: # full range
            target_similarities = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        target_similarities = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0] # paper results

        target_similarities.reverse()
        target_images = []
        
        def scale_tsim(tsim, min_tsim=0.0, max_tsim=1.0, a=-0.6, b=1.0):
            return a + (((tsim - min_tsim) * (b - a) ) / (max_tsim - min_tsim))

        latent_rand = g_ema.mean_latent(1).detach().clone() #*0.05

        for tsim in target_similarities:
            init_latent = latent_code_init.detach().clone() 
           
            # without adjusting the starting position (limited range)  
            #latent = init_latent            
            
            mean_latent_vector = (mean_latent).detach().clone().repeat(1, 18, 1)
            initial_to_mean_latent_direction = latent_code_init - mean_latent_vector 

            # adjusting the starting position of the optimization 
            latent = mean_latent_vector + initial_to_mean_latent_direction * tsim 

            latent.requires_grad = True

            if args.work_in_stylespace:
                optimizer = optim.Adam(latent, lr=args.lr)
            else:
                optimizer = optim.Adam([latent], lr=args.lr)

            mapped_tsim = tsim

            for i in pbar:
                t = i / args.step
                lr = get_lr(t, args.lr)
                optimizer.param_groups[0]["lr"] = lr

                img_gen, _ = g_ema([latent], input_is_latent=True, randomize_noise=False, input_is_stylespace=args.work_in_stylespace)
                img_gen = gan2img_normalize(img_gen)
                
                c_loss = mse_loss(img_gen, img_orig)
                g_loss = gender_model.compute_loss(img_orig, img_gen)
                e_loss = expression_model.compute_loss(img_orig, img_gen)

                actual_id_loss = None
                if args.id_lambda > 0:
                    i_loss = id_loss(img_gen, img_orig, orig_id_feats)[0]
                    actual_id_loss = i_loss
                    i_loss = ((mapped_tsim - i_loss) ** 2)
                else:
                    i_loss = 0

                if args.mode == "edit":
                    if args.work_in_stylespace:
                        l2_loss = sum([((latent_code_init[c] - latent[c]) ** 2).sum() for c in range(len(latent_code_init))])
                    else:
                        l2_loss = ((latent_code_init - latent) ** 2).sum()
                    loss = 0.0
                    
                    if 'PX' in ablation_arguments:
                        loss += args.px_lambda * c_loss # pixel loss
                    if 'LT' in ablation_arguments:
                        loss += args.l2_lambda * l2_loss # latent loss
                    if 'ID' in ablation_arguments:
                        loss += args.id_lambda * i_loss # id loss
                    if 'EX' in ablation_arguments:
                        loss += args.ex_lambda * e_loss # expression loss
                    if 'GD' in ablation_arguments:
                        loss += args.gd_lambda * g_loss # gender loss

                else:
                    loss = c_loss


                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                '''
                pbar.set_description(
                    (
                        "MSE:{:.4f} ID:{:.4f} ID-TSIM:{:.4f} TH:{:.4f};".format(c_loss.item(), i_loss.item(), actual_id_loss, tsim)
                    )
                )
                #'''
                
                enable_logging = False

                if enable_logging and args.save_intermediate_image_every > 0 and i % args.save_intermediate_image_every == 0:
                    
                    
                    with torch.no_grad():
                        img_gen, _ = g_ema([latent], input_is_latent=True, randomize_noise=False, input_is_stylespace=args.work_in_stylespace)
                    
                    display_image = img_gen.clamp_(min=-1, max=1).add(1).div_(2).mul(255).type(torch.uint8) #.permute(0, 2, 3, 1)
                    display_image = display_image[0]

                    writer.add_scalar('{}_{}_mse_loss'.format(image_name.split('.')[0], tsim), c_loss.item(), i)
                    writer.add_scalar('{}_{}_EXP_loss'.format(image_name.split('.')[0], tsim), e_loss.item(), i)
                    writer.add_scalar('{}_{}_GEN_loss'.format(image_name.split('.')[0], tsim), g_loss.item(), i)
                    writer.add_scalar('{}_{}_ID-tsim_loss'.format(image_name.split('.')[0], tsim), i_loss.item(), i)
                    writer.add_scalar('{}_{}_ID_loss'.format(image_name.split('.')[0], tsim), actual_id_loss, i)
                    writer.add_scalar('{}_{}_L2_loss'.format(image_name.split('.')[0], tsim), l2_loss.item(), i)
                    writer.add_scalar('{}_{}_total_loss'.format(image_name.split('.')[0], tsim), loss.item(), i)
                    writer.add_image('{}_{}_image'.format(image_name.split('.')[0], tsim), display_image, i)
                    

                #torchvision.utils.save_image(img_gen, f"results/{str(i).zfill(5)}.jpg", normalize=True, range=(-1, 1))
                if (i == args.step-1): # final loop where we save the image
                    with torch.no_grad():
                        img_gen, _ = g_ema([latent], input_is_latent=True, randomize_noise=False, input_is_stylespace=args.work_in_stylespace)
                    #img_gen = g_transforms(img_gen) # save in full res
                    base_path = os.path.join(dataset_save, str(tsim))
                    ensure_dir(base_path)
                    torchvision.utils.save_image(img_gen, os.path.join(base_path, image_name.replace(dataset_filetype, dataset_newtype)), normalize=True, range=(-1, 1))
                
                #torchvision.utils.save_image(img_gen, output_path, normalize=True, range=(-1, 1))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ablation", type=str, default="ID_PX_LT_GD_EX", help="modules which can be switched on and off during ablation") # _PO for pose
    parser.add_argument("--dataset", type=str, default="lrv", help="datasets: rafd, lrv, celeba, xm2vts, affectnet")
    parser.add_argument("--experiment_label", type=str, default="deid_256_GD_EX_loss_EXFIX", help="User label for experiment, which will be part of created folders")
     
    parser.add_argument("--ckpt", type=str, default="../pretrained_models/ffhq-256-config-f.pt", help="pretrained StyleGAN2 weights")
    parser.add_argument("--stylegan_size", type=int, default=256, help="StyleGAN resolution") 
    
    parser.add_argument("--lr_rampup", type=float, default=0.05)
    parser.add_argument("--lr", type=float, default=0.02) #0.1
    parser.add_argument("--step", type=int, default=15, help="number of optimization steps")
    parser.add_argument("--mode", type=str, default="edit", choices=["edit", "free_generation"], help="choose between edit an image an generate a free one")
    
    parser.add_argument("--l2_lambda", type=float, default=0.016, help="weight of the latent distance (used for editing only)")
    parser.add_argument("--id_lambda", type=float, default=2.5, help="weight of id loss (used for editing only)")
    parser.add_argument("--px_lambda", type=float, default=1.0, help="weight of px loss (used for editing only)")
    parser.add_argument("--ex_lambda", type=float, default=0.1, help="weight of ex loss (used for editing only)")
    parser.add_argument("--gd_lambda", type=float, default=0.1, help="weight of gd loss (used for editing only)")
    
    parser.add_argument("--latent_path", type=str, default=None, help="starts the optimization from the given latent code if provided. Otherwose, starts from"
                                                                      "the mean latent in a free generation, and from a random one in editing. "
                                                                      "Expects a .pt format")
    parser.add_argument("--truncation", type=float, default=0.7, help="used only for the initial latent vector, and only when a latent code path is"
                                                                      "not provided")
    parser.add_argument('--work_in_stylespace', default=False, action='store_true')
    parser.add_argument("--save_intermediate_image_every", type=int, default=1, help="if > 0 then saves intermidate results during the optimization")
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument('--ir_se50_weights', default='../pretrained_models/model_ir_se50.pth', type=str,
                             help="Path to facial recognition network used in ID loss")
     #parser.add_argument('--ir_se50_weights_expression', default='../pretrained_models/affectnet_expression_acc-0.88_step-89910_final.pth', type=str,
    #                         help="Path to facial recognition network used in Expression loss")
    parser.add_argument('--ir_se50_weights_expression', default='../pretrained_models/new/affectnet_EX_accuracy-0.878_step-18900.pth', type=str,
                             help="Path to facial recognition network used in Expression loss")
    
    #parser.add_argument('--ir_se50_weights_gender', default='../pretrained_models/gender_celeba_accuracy-0.60_step-7560_final.pth', type=str,
    #                         help="Path to facial recognition network used in Gender loss")
    parser.add_argument('--ir_se50_weights_gender', default='../pretrained_models/new/celeba_GD_accuracy-0.889_step-12600.pth', type=str,
                             help="Path to facial recognition network used in Gender loss")

    args = parser.parse_args()

    main(args)
