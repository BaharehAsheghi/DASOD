import torch
import torch.nn.functional as F
from torch.autograd import Variable

import os
import argparse
from datetime import datetime
from model.URCOD import Generator 
from utils.loss import smoothness_loss, structure_loss, flooding_dice_bce_loss
from utils.dataloader import get_loader
from utils.utils import adjust_lr, linear_annealing, visualize

import time
import sys 
sys.path.insert(1, './UR-COD/cFlow')
from models.cflownet import cFlowNet
from util.tools import makeLogFile, writeLog, dice_loss
from util.utils import l2_regularisation,ged

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--name', type=str, default='UR-SINetv2')
parser.add_argument('--dataset', type=str, default='DUTS-TR')
parser.add_argument('--epoch', type=int, default=40, help='epoch number')
parser.add_argument('--lr_gen', type=float, default=5e-5, help='learning rate')
parser.add_argument('--batchsize', type=int, default=10, help='training batch size')
parser.add_argument('--trainsize', type=int, default=352, help='training dataset size')
parser.add_argument('--decay_rate', type=float, default=0.9, help='decay rate of learning rate')
parser.add_argument('--decay_epoch', type=int, default=80, help='every n epochs decay learning rate')
parser.add_argument('--beta1_gen', type=float, default=0.5,help='beta of Adam for generator')
parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight_decay')
parser.add_argument('--latent_dim', type=int, default=6, help='latent dim')
parser.add_argument('--feat_channel', type=int, default=32, help='reduced channel of camouflaged feat')
parser.add_argument('--sm_weight', type=float, default=0.1, help='weight for smoothness loss')
parser.add_argument('--edge_weight', type=float, default=10.0, help='weight for edge loss')
parser.add_argument('--lat_weight', type=float, default=10.0, help='weight for latent loss')
parser.add_argument('--vae_loss_weight', type=float, default=0.4, help='weight for vae loss')
parser.add_argument('--mse_loss_weight', type=float, default=0.1, help='weight for mse loss')

parser.add_argument('--flow', action='store_true', default=True, help=' Train with Flow model')
parser.add_argument('--glow', action='store_true', default=False, help=' Train with Glow model')
parser.add_argument('--num_flows', type=int, default=4, help='Num flows')
parser.add_argument('--unet', action='store_true', default=False, help='Train with Det. Unet')
parser.add_argument('--singleRater', action='store_true', default=False, help='Train with single rater')

opt = parser.parse_args()

torch.cuda.set_device(opt.gpu)
print("Using flow based model with %d steps"%opt.num_flows)

generator = Generator(channel=opt.feat_channel, latent_dim=opt.latent_dim)
generator.cuda()
generator_params = generator.parameters()
generator_optimizer = torch.optim.Adam(generator_params, opt.lr_gen, betas=[opt.beta1_gen, 0.999], weight_decay=opt.weight_decay)

image_root = './SINet-V2-main/Dataset/TrainValDataset/Image/'
gtmask_root = './SINet-V2-main/Dataset/TrainValDataset/GT/'
gtedge_root = './SINet-V2-main/Dataset/detail-label/'
pseudomask_root = './SINet-V2-main/res/SINet_V2/DUTS-TR/'
gray_root = './SINet-V2-main/Dataset/TrainValDataset/GrayImages/'

train_loader, training_set_size = get_loader(image_root, gtmask_root, gtedge_root, pseudomask_root, gray_root, batchsize=opt.batchsize, trainsize=opt.trainsize)
total_step = len(train_loader)
train_z = torch.FloatTensor(training_set_size, opt.latent_dim).normal_(0, 1).cuda()

mse_loss = torch.nn.MSELoss(size_average=True, reduce=True)
smooth_loss = smoothness_loss(size_average=True)
dice_bce_loss = flooding_dice_bce_loss() 

for epoch in range(1, opt.epoch+1):
    print('Generator Learning Rate: {}'.format(generator_optimizer.param_groups[0]['lr']))

    for i, pack in enumerate(train_loader, start=1):
        images, gtmasks, gtmask_rgbs, gtedges, pseudomasks, grays, index_batch = pack
        images = Variable(images).cuda()
        gtmasks = Variable(gtmasks).cuda()
        gtmask_rgbs = Variable(gtmask_rgbs).cuda()
        gtedges = Variable(gtedges).cuda()
        pseudomasks = Variable(pseudomasks).cuda()
        grays = Variable(grays).cuda()
        
        if not opt.unet:
            prior, posterior, pred_post, pred_prior, latent_loss, pseudo_pred_post, pseudo_pred_prior, pseudoedges=generator.forward(images, pseudomasks, gtmasks, training=True) #Refine
            reg_loss = l2_regularisation(posterior) + l2_regularisation(prior)
            loss = latent_loss + 1e-5 * reg_loss

        gt_pseudos = torch.cat((gtmask_rgbs, gtedges), 1)

        # gen_loss_cvae (The Posterior RefinementNet Loss)
        smoothLoss_post = opt.sm_weight * smooth_loss(torch.sigmoid(pred_post), gtmasks)
        mse_loss_post = opt.mse_loss_weight * mse_loss(torch.sigmoid(pseudo_pred_post), gt_pseudos)
        ref_loss = structure_loss(pred_post, gtmasks) + smoothLoss_post + mse_loss_post

        anneal_reg = linear_annealing(0, 1, epoch, opt.epoch)
        latent_loss = opt.lat_weight * anneal_reg * loss
        gen_loss_cvae = ref_loss + latent_loss 
        gen_loss_cvae = opt.vae_loss_weight * gen_loss_cvae

        # gen_loss_gsnn (The Prior RefinementNet Loss)
        smoothLoss_prior = opt.sm_weight * smooth_loss(torch.sigmoid(pred_prior), gtmasks)
        mse_loss_prior = opt.mse_loss_weight * mse_loss(torch.sigmoid(pseudo_pred_prior), gt_pseudos)
        gen_loss_gsnn = structure_loss(pred_prior, gtmasks) + smoothLoss_prior + mse_loss_prior

        gen_loss_gsnn = (1-opt.vae_loss_weight) * gen_loss_gsnn

        edge_loss = dice_bce_loss(gtedges, pseudoedges) * opt.edge_weight
        gen_loss = gen_loss_cvae + gen_loss_gsnn + edge_loss 

        generator_optimizer.zero_grad()
        gen_loss.backward()
        generator_optimizer.step()
 
        print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], gen vae Loss: {:.4f}, gen gsnn Loss: {:.4f}, gen Loss: {:.4f}'.
                  format(datetime.now(), epoch, opt.epoch, i, total_step, gen_loss_cvae.data, gen_loss_gsnn.data, gen_loss.data))

    adjust_lr(generator_optimizer, opt.lr_gen, epoch, opt.decay_rate, opt.decay_epoch)

    save_path = 'checkpoints/{}/'.format(opt.name)
    os.makedirs(save_path, exist_ok=True)

    torch.save(generator.state_dict(), save_path + 'Model_gen.pth')
    print('Save state_dict successfully! epoch:{}.'.format(epoch))

    if epoch >= 10 and epoch % 5 == 0:
        torch.save(generator.state_dict(), save_path + 'Model' + '_%d' % epoch + '_gen.pth')
        print('Save state_dict successfully! epoch:{}.'.format(epoch))
