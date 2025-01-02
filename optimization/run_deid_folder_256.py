import argparse
from email.mime import image
import math
import os
from cv2 import exp

import torch
import torchvision
from torch import optim
from tqdm import tqdm
from PIL import Image
from torchvision import transforms as T

import sys
sys.path.append(".")
sys.path.append("..")
import numpy as np
# reproducibility
#torch.manual_seed(0)
#np.random.seed(0)
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
seed_everything(42)

#from criteria.clip_loss import CLIPLoss
from criteria.id_loss import IDLoss, GenderLoss, ExpressionLoss
from mapper.training.train_utils import STYLESPACE_DIMENSIONS
from models.stylegan2.model import Generator
#import clip
from utils import ensure_checkpoint_exists

#sys.path.insert(0, "../content")
#sys.path.insert(0, "../content/encoder4editing")

#from content.encoder4editing.models.psp import pSp
#from content.encoder4editing.utils.alignment import align_face
#from content.encoder4editing.utils.common import tensor2im

def ensure_dir(d):
    #dd = os.path.dirname(f)
    if not os.path.exists(d):
        os.makedirs(d)
        


STYLESPACE_INDICES_WITHOUT_TORGB = [i for i in range(len(STYLESPACE_DIMENSIONS)) if i not in list(range(1, len(STYLESPACE_DIMENSIONS), 3))]

def get_lr(t, initial_lr, rampdown=0.25, rampup=0.05):
    lr_ramp = min(1, (1 - t) / rampdown)
    lr_ramp = 0.5 - 0.5 * math.cos(lr_ramp * math.pi)
    lr_ramp = lr_ramp * min(1, t / rampup)

    return initial_lr * lr_ramp


def main(args):

    
    ensure_checkpoint_exists(args.ckpt)
    #text_inputs = torch.cat([clip.tokenize(args.description)]).cuda()
    os.makedirs(args.results_dir, exist_ok=True)

    g_ema = Generator(args.stylegan_size, 512, 8)
    g_ema.load_state_dict(torch.load(args.ckpt)["g_ema"], strict=False)
    g_ema.eval()
    g_ema = g_ema.cuda()
    mean_latent = g_ema.mean_latent(4096)
    face_shape_latent = torch.from_numpy(np.load('face_shape.npy')).to(torch.float).cuda()
    face_width_latent = torch.from_numpy(np.load('width.npy')).to(torch.float).cuda()
    face_height_latent = torch.from_numpy(np.load('height.npy')).to(torch.float).cuda()
    
    #'''
    dataset_path = '../../stylegan2-pytorch/datasets/celeba-test_aligned'
    #projections_path = 'projections/pure_projections' # these vectors use 'w' as key
    #dataset_save = 'datasets/celeba-test_deidentified_modular_vgg_mean_std'
    #dataset_save = '/media/blaz/Storage/datasets/results/celeba-test_deidentified_svd_raptor' # on HDD, more space
    encodings_path = '../../stylegan2-pytorch/datasets/results/celeba-test_hyperspace_encodings'
    dataset_save = '../../stylegan2-pytorch/datasets/results/celeba-test_deidentified_openset' # on HDD, more space
    #''' # windows
    dataset_path = 'E:\\datasets\\AffectNet\\val_set\\aligned_images'
    dataset_path = '..\\datasets\\AffectNet\\aligned_images'
    
    encodings_path = 'E:\\datasets\\AffectNet\\val_set\\projections_hyperstyle'
    dataset_save = 'E:\\datasets\\AffectNet\\val_set\\aligned_images_deidentified'
    
    dataset_path = '..\\datasets\\lrv_aligned'
    encodings_path = '..\\datasets\\lrv_encodings'
    dataset_save = '..\\datasets\\lrv_deidentified_EX_GD_lr0.01_256'

    dataset_path = '..\\datasets\\rafd-frontal_aligned'
    encodings_path = '..\\datasets\\rafd-frontal_hyperspace_encodings'
    dataset_save = '..\\datasets\\rafd_deidentified_EX_GD_lr0.01_256'


    #dataset_path = '..\\..\\ArcFace\\datasets\\xm2vts_aligned'
    #encodings_path = '..\\datasets\\xm2vts_hyperspace_encodings'
    #dataset_save = '..\\datasets\\results\\xm2vts_deidentified_EX_GD_RND-1'
    '''
    #'''
    # sample images from lab members
    #''' # TODO: run all images again and rerun all plots ROC, utility (right now running Celeba into Storage)
    #'''
    #'''
    dataset_path = '../../stylegan2-pytorch/datasets/lrv_aligned'
    projections_path = 'datasets/lrv_projections' # these vectors use 'w' as key
    encodings_path = '../../stylegan2-pytorch/datasets/results/lrv_hyperspace_encodings'
    dataset_save = '../../stylegan2-pytorch/datasets/results/lrv_emotions_test'
    #'''
    
    '''
    dataset_path = '../../stylegan2-pytorch/datasets/xm2vts_aligned'
    encodings_path = '../../stylegan2-pytorch/datasets/results/xm2vts_hyperspace_encodings'
    dataset_save = '../../stylegan2-pytorch/datasets/results/xm2vts_deidentified_EX_GD_256_mean_5.0' #closedset_step-260' # on HDD, more space
    #args.ir_se50_weights = '../pretrained_models/model_xm2vts_acc-1.0_step-260_final.pth' # enable this to do closed set experiment
    #'''
    '''
    dataset_path = '../../stylegan2-pytorch/datasets/rafd-frontal_aligned'
    encodings_path = '../../stylegan2-pytorch/datasets/results/rafd-frontal_hyperspace_encodings' # TODO
    dataset_save = '../../stylegan2-pytorch/datasets/results/rafd-frontal_deidentified_EX_GD_ID_256' # _step-680'
    #args.ir_se50_weights = '../pretrained_models/model_rafd_acc-0.98_step-680_final.pth' # enable this to do closed set experiment
    #'''
    '''
    dataset_path = '../../DAN/datasets/AffectNet/val_set/aligned_images'
    encodings_path = '../../DAN/datasets/AffectNet/val_set/projections_hyperstyle'
    dataset_save = '../../stylegan2-pytorch/datasets/results/AffectNet_deidentified_EX_GD' # on HDD, more space
    #'''
    '''
    dataset_path = '../../stylegan2-pytorch/datasets/celeba-test_aligned'
    #projections_path = 'projections/pure_projections' # these vectors use 'w' as key
    #dataset_save = 'datasets/celeba-test_deidentified_modular_vgg_mean_std'
    #dataset_save = '/media/blaz/Storage/datasets/results/celeba-test_deidentified_svd_raptor' # on HDD, more space
    encodings_path = '../../stylegan2-pytorch/datasets/results/celeba-test_hyperspace_encodings'
    dataset_save = '../../stylegan2-pytorch/datasets/results/celeba-test_deidentified_EX_GD_256' # on HDD, more space
    #'''

    dataset_path = '/media/blaz/Storage/datasets/deid-toolkit/aligned/muct/'
    encodings_path = '/media/blaz/Storage/datasets/deid-toolkit/baselines/CPP-DEID-encodings/muct'
    dataset_save = '/media/blaz/Storage/datasets/deid-toolkit/baselines/CPP-DEID/muct_no_DAN/' # on HDD, more space
    #'''

    dataset_filetype = 'jpg'
    dataset_newtype = 'jpg'

    img_names = [i for i in os.listdir(dataset_path) if dataset_filetype in i] # change ppm into jpg
    img_names = sorted(img_names)
    img_paths = [os.path.join(dataset_path, i) for i in img_names]
    save_paths = [os.path.join(dataset_save, i.replace(dataset_filetype, dataset_newtype)) for i in img_names]
    encoding_paths = [os.path.join(encodings_path, i.replace(dataset_filetype, 'pt')) for i in img_names]

    ensure_dir(dataset_save)
    #ensure_dir(encodings_path)

    from torch.nn import MSELoss
    #clip_loss = MSELoss() #CLIPLoss(args)
    #id_loss = IDLoss(args)
    
    from run_deid import get_laplace_noise, get_gaussian_noise
    from tqdm import tqdm

    #from piqa import SSIM
    #class SSIMLoss(SSIM):
    #    def forward(self, x, y):
    #        return 1. - super().forward(x, y)

    mse_loss = MSELoss().cuda() #CLIPLoss(args)
    gender_loss = GenderLoss(args).cuda()
    expr_loss = ExpressionLoss(args).cuda()
    #clip_loss = SSIMLoss().cuda()
    id_loss = IDLoss(args).cuda()
    
    ensure_dir('./log')
    from tensorboardX import SummaryWriter
    enable_logging = False # set to True to save resulting data to tensorboard
    writer = SummaryWriter('./log')

    #im_bar = tqdm(range(args.step))
    for ix, (image_path, output_path, encoding_path) in tqdm(enumerate(zip(img_paths, save_paths, encoding_paths)), total=len(img_names)):
    
        image_name = os.path.split(image_path)[1]
        #if "_1_1" in image_name or "_1_2" in image_name or "_2_1" in image_name or "_2_2" in image_name: # temporary filter for xm2vts
        #    continue
        output_path_tsim = os.path.join(dataset_save, str(0.1), image_name)
        #output_base = os.path.split(output_path)[0]
        #if not 'Manfred' in image_name:
        #    continue
        
        if os.path.exists(output_path_tsim): # skip if image already generated on last tsim, to save time
            print(image_name, " exists in ", 0.1, " skipping.")
            continue

        if args.latent_path:
            latent_code_init = torch.load(args.latent_path).cuda()
        elif args.mode == "edit":
            #print("editing in progress")
            #latent_code_init_not_trunc = torch.randn(1, 18, 512).cuda()*0.25
            #with torch.no_grad():
            #    _, latent_code_init, _ = g_ema([latent_code_init_not_trunc], return_latents=True,
            #                                truncation=args.truncation, truncation_latent=mean_latent)
            #sad_code_init = torch.load("/home/blaz/github/ArcFace/affectnet-train-prototype-sad.pt")[0].cuda()
            #sad_code_init = torch.from_numpy(np.load('/home/blaz/github/generators-with-stylegan2/latent_directions/gender.npy')).to(torch.float).cuda()
            #sad_code_init = torch.unsqueeze(sad_code_init, 0)
            #print("Latent code shape: ", latent_code_init.shape)
            latent_code_init = torch.load(encoding_path).cuda()
            #latent_code_init = torch.load("/media/blaz/Storage/datasets/results/lrv_hyperspace/aligned_VitoStruc.pt")
            #latent_code_init = torch.load("/media/blaz/Storage/datasets/results/lrv_hyperspace/aligned_BorutBatagelj.pt")
            #latent_code_init = torch.load("/media/blaz/Storage/datasets/results/lrv_hyperspace/aligned_NarvikaBovcon.pt")
            #latent_code_init = torch.load("/media/blaz/Storage/datasets/results/lrv_hyperspace/aligned_ZigaEmersic.pt")
            #latent_code_init += latent_code_init_not_trunc

            #latent_code_init = torch.from_numpy(np.array([latent_code_init])).cuda() # already done this in process_folder
            #print(latent_code_init.shape)

        else:
            latent_code_init = mean_latent.detach().clone().repeat(1, 18, 1)

        # TODO: replace this with original image loaded from file?
        with torch.no_grad():
            #latent_code_init[0][:8] = (latent_code_init[0] - 15*sad_code_init)[:8]
            # + sad_code_init
            img_orig, _ = g_ema([latent_code_init], input_is_latent=True, randomize_noise=False)
            #torchvision.utils.save_image(img_orig, os.path.join(dataset_save, image_name), normalize=False) #, range=(-1, 1))
            torchvision.utils.save_image(img_orig, os.path.join(dataset_save, image_name), normalize=True, range=(-1, 1))
        #print(img_orig.shape)
        #g_transforms = T.Compose([
        #    T.ToPILImage(),
        #    T.Resize((256, 256)),
        #    T.ToTensor(),
        #])
        def load_image(img_path):
            transforms = T.Compose([
                T.Resize((256, 256)),
                T.ToTensor(),
                T.Normalize(mean=[0.5], std=[0.5]),
            ])

            data = Image.open(img_path)
            data = data.convert('RGB')
            data = transforms(data)
            return data

        #blurrer = T.GaussianBlur(kernel_size=(9, 9), sigma=5)

        img_orig = load_image(image_path).unsqueeze(0).cuda()

        orig_id_feats = id_loss.extract_feats(img_orig).detach()
        orig_gd_feats = gender_loss.extract_feats(img_orig).detach()
        orig_ex_feats = expr_loss.extract_feats(img_orig).detach()

        #print(img_orig.shape)
        # TODO: try to save original image back. And double check the range if needs to be normalized before feeding the image to loss
        #torchvision.utils.save_image(img_orig, os.path.join(dataset_save, image_name), normalize=False) #, range=(-1, 1))
                
        '''
        if args.work_in_stylespace:
            with torch.no_grad():
                _, _, latent_code_init = g_ema([latent_code_init], input_is_latent=True, return_latents=True)
            latent = [s.detach().clone() for s in latent_code_init]
            for c, s in enumerate(latent):
                if c in STYLESPACE_INDICES_WITHOUT_TORGB:
                    s.requires_grad = True
        else:
            latent = latent_code_init.detach().clone()
            latent.requires_grad = True
        '''
        
        #pbar = tqdm(range(args.step))
        pbar = (range(args.step))

        im_samples = []
        id_scores = []

        target_similarities = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9] # controllable privacy parameter 

        #target_similarities = [0.0]
        target_similarities.reverse()
        target_images = []

        
        from torch.autograd import Variable
        from torch.nn import Parameter
        for tsim in target_similarities:
            latent = latent_code_init.detach().clone() 
            # intial latent could also be mean vector, so that the model can converge anywhere
            #latent_rand = torch.randn(1, 18, 512).cuda()*0.45       
            # enable grad on latent
            latent.requires_grad = True

            if args.work_in_stylespace:
                optimizer = optim.Adam(latent, lr=args.lr)
            else:
                optimizer = optim.Adam([latent], lr=args.lr)


            for i in pbar:
                t = i / args.step
                lr = get_lr(t, args.lr)
                optimizer.param_groups[0]["lr"] = lr

                img_gen, _ = g_ema([latent], input_is_latent=True, randomize_noise=False, input_is_stylespace=args.work_in_stylespace)
                #img_gen = g_transforms(img_gen[0].cpu()).cuda().unsqueeze(0)
                #print(img_orig.min(), img_orig.max(), img_gen.min(), img_gen.max())
                #print("generated image type: ", img_gen.type())

                #c_loss = 0.0 #clip_loss(img_gen, text_inputs)
                #c_loss = mse_loss(img_gen, blurrer(img_orig)) # blur original image?
                c_loss = mse_loss(img_gen, img_orig)
                g_loss = gender_loss(img_gen, img_orig, orig_gd_feats)[0]
                e_loss = expr_loss(img_gen, img_orig, orig_ex_feats)[0]

                #mapped_tsim = ((tsim - 0.5)*2.0) # map in between -0.8 and 0.8
                mapped_tsim = tsim

                actual_id_loss = None
                if args.id_lambda > 0:
                    i_loss = id_loss(img_gen, img_orig, orig_id_feats)[0]
                    actual_id_loss = i_loss
                    #i_loss = ((i_loss - mapped_tsim) ** 2) #.sum()
                    i_loss = ((mapped_tsim - i_loss) ** 2)
                else:
                    i_loss = 0

                if args.mode == "edit":
                    if args.work_in_stylespace:
                        l2_loss = sum([((latent_code_init[c] - latent[c]) ** 2).sum() for c in range(len(latent_code_init))])
                    else:
                        l2_loss = ((latent_code_init - latent) ** 2).sum()
                    loss = c_loss*0 + args.l2_lambda * l2_loss + args.id_lambda * i_loss + e_loss + g_loss
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
                
                enable_logging = False # ix < 16 # for first few images

                if enable_logging and args.save_intermediate_image_every > 0 and i % args.save_intermediate_image_every == 0:
                    with torch.no_grad():
                        img_gen, _ = g_ema([latent], input_is_latent=True, randomize_noise=False, input_is_stylespace=args.work_in_stylespace)
                    #img_gen = g_transforms(img_gen[0].cpu()).cuda().unsqueeze(0)
                    #im_samples.append(img_gen.detach().cpu())
                    #id_scores.append(i_loss.detach().cpu().item())

                    #print("Min max generated image: ", img_gen[0].min(), img_gen[0].max())

                    display_image = img_gen.clamp_(min=-1, max=1).add(1).div_(2).mul(255).type(torch.uint8) #.permute(0, 2, 3, 1)
                    # TODO: resize if needed
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
                    torchvision.utils.save_image(img_gen, os.path.join(base_path, image_name), normalize=True, range=(-1, 1))
                
                #torchvision.utils.save_image(img_gen, output_path, normalize=True, range=(-1, 1))
                #break



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--description", type=str, default="a person with purple hair", help="the text that guides the editing/generation")
    #parser.add_argument("--ckpt", type=str, default="../pretrained_models/stylegan2-ffhq-config-f.pt", help="pretrained StyleGAN2 weights")
    #parser.add_argument("--stylegan_size", type=int, default=1024, help="StyleGAN resolution") 
    parser.add_argument("--ckpt", type=str, default="../pretrained_models/ffhq-256-config-f.pt", help="pretrained StyleGAN2 weights")
    parser.add_argument("--stylegan_size", type=int, default=256, help="StyleGAN resolution") 
    
    parser.add_argument("--lr_rampup", type=float, default=0.05)
    parser.add_argument("--lr", type=float, default=0.01) #0.1
    parser.add_argument("--step", type=int, default=5, help="number of optimization steps")
    parser.add_argument("--mode", type=str, default="edit", choices=["edit", "free_generation"], help="choose between edit an image an generate a free one")
    parser.add_argument("--l2_lambda", type=float, default=0.016, help="weight of the latent distance (used for editing only)")
    parser.add_argument("--id_lambda", type=float, default=4.500, help="weight of id loss (used for editing only)")
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
    parser.add_argument('--ir_se50_weights_expression', default='../pretrained_models/affectnet_expression_acc-0.88_step-89910_final.pth', type=str,
                             help="Path to facial recognition network used in Expression loss")
    parser.add_argument('--ir_se50_weights_gender', default='../pretrained_models/gender_celeba_accuracy-0.60_step-7560_final.pth', type=str,
                             help="Path to facial recognition network used in Genderdd loss")

    args = parser.parse_args()

    main(args)
