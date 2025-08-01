import argparse

from net.stdnet import STDNet as Net

from net.common.diff import compute_interframe_diff
from data.tartanair_dataloader import *
from utils import *
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms
import torch.optim as optim
from net.losses import CharbonnierLoss
import random
import torch.nn.functional as F

from tqdm import tqdm
import logging
from datetime import datetime
import os

parser = argparse.ArgumentParser()

parser.add_argument("--local-rank", default=-1, type=int)

parser.add_argument('--scale', type=int, default=4, help='scale factor')
parser.add_argument('--lr', default='0.0001', type=float, help='learning rate')
parser.add_argument('--epoch', default=500, type=int, help='max epoch')
parser.add_argument('--device', default="0,1", type=str, help='which gpu use')
parser.add_argument("--decay_iterations", type=list, default=[150,200],
                    help="steps to start lr decay")
parser.add_argument("--gamma", type=float, default=0.2, help="decay rate of learning rate")
parser.add_argument("--root_dir", type=str, default='/opt/data/private/dataset/', help="root dir of dataset")
parser.add_argument("--batchsize", type=int, default=3, help="batchsize of training dataloader")
parser.add_argument("--num_gpus", type=int, default=2, help="num_gpus")
parser.add_argument('--seed', type=int, default=7240, help='random seed point')
parser.add_argument("--result_root", type=str, default='experiment/test1', help="root dir of dataset")
parser.add_argument("--result_root_MAE", type=str, default='experiment/test1', help="root dir of dataset")

opt = parser.parse_args()
print(opt)

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

torch.manual_seed(opt.seed)
random.seed(opt.seed)
torch.cuda.manual_seed_all(opt.seed)

local_rank = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(local_rank)
dist.init_process_group(backend='nccl')
device = torch.device("cuda", local_rank)

s = datetime.now().strftime('%Y%m%d%H%M%S')
rank = dist.get_rank()
logging.basicConfig(filename='%s/train.log' % opt.result_root, format='%(asctime)s %(message)s', level=logging.INFO)
logging.info(opt)

net = Net(scale=opt.scale).cuda()

print("**********************Params***********************")
print(sum(p.numel() for p in net.parameters() if p.requires_grad))
print('params: %.2f M' % (sum(p.numel() for p in net.parameters() if p.requires_grad) / 1e6))
print("*********************************************")

data_transform = transforms.Compose([transforms.ToTensor()])
train_dataset = Tartanair_Dataset(root_path="/opt/data/private/dataset/VDSRDataset/TartanAir/", train=True,
                                  txt_file="./data/tartanair_train.txt", transform=data_transform,
                                  scale=opt.scale)
test_dataset = Tartanair_Dataset(root_path="/opt/data/private/dataset/VDSRDataset/TartanAir/", train=False,
                                 txt_file="./data/tartanair_val.txt", transform=data_transform,
                                 scale=opt.scale)

if torch.cuda.device_count() > 1:
    train_sampler = DistributedSampler(dataset=train_dataset)
train_dataloader = DataLoader(train_dataset, batch_size=opt.batchsize, shuffle=False, pin_memory=True, num_workers=8,
                              drop_last=True, sampler=train_sampler)
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, pin_memory=True, num_workers=8)

net = DistributedDataParallel(net, device_ids=[local_rank], output_device=int(local_rank), find_unused_parameters=True)

l_rec = CharbonnierLoss().to(device)
l_diff = nn.L1Loss().to(device)

optimizer = optim.Adam(net.module.parameters(), lr=opt.lr)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=opt.decay_iterations, gamma=opt.gamma)
net.train()

max_epoch = opt.epoch
num_train = len(train_dataloader)
best_rmse = 1000.0
best_mae = 15.0
best_epoch = 0
for epoch in range(max_epoch):
    # ---------
    # Training
    # ---------
    train_sampler.set_epoch(epoch)
    net.train()
    running_loss = 0.0

    t = tqdm(iter(train_dataloader), leave=True, total=len(train_dataloader))

    for idx, data in enumerate(t):
        batches_done = num_train * epoch + idx
        optimizer.zero_grad()
        guidance, lr, gt = data['guidance'].to(device), data['lr'].to(device), data['gt'].to(device)

        restored, intermed, Diff_Intra, Diff_Inter = net(lqs=lr, guides=guidance)

        loss_rec = l_rec(restored, gt)
        
        Intra_diff_rec1 = Diff_Intra["Intra_diff1_rec"]
        Intra_diff_rec2 = Diff_Intra["Intra_diff2_rec"]
        Intra_diff_rec1 = F.elu(Intra_diff_rec1)
        Intra_diff_rec2 = F.elu(Intra_diff_rec2)
        
        b1,t1,c1,h1,w1 = Intra_diff_rec1.shape
        s1 = Intra_diff_rec1.view(b1, c1, t1, -1)
        pmin1 = torch.min(s1, dim=-1)
        pmin1 = pmin1[0].unsqueeze(dim=-1).unsqueeze(dim=-1)
        s1 = Intra_diff_rec1
        s1 = s1 - pmin1 + 1
        sr_1 = torch.mul(restored, s1)
        hr_1 = torch.mul(gt, s1)
        loss_diffItra1 = l_diff(sr_1, hr_1)

        b2, t2, c2, h2, w2 = Intra_diff_rec2.shape
        s2 = Intra_diff_rec2.view(b2, c2, t2, -1)
        pmin2 = torch.min(s2, dim=-1)
        pmin2 = pmin2[0].unsqueeze(dim=-1).unsqueeze(dim=-1)
        s2 = Intra_diff_rec2
        s2 = s2 - pmin2 + 1
        sr_2 = torch.mul(restored, s2)
        hr_2 = torch.mul(gt, s2)
        loss_diffItra2 = l_diff(sr_2, hr_2)

        loss_diffItra = 0.2 * loss_diffItra1 + 0.2 * loss_diffItra2

        gt_diff_inter, gt_diff_inter_cross = compute_interframe_diff(gt)
        loss_diffInter1 = l_diff(Diff_Inter["Inter_diff1_rec"], gt_diff_inter)
        loss_diffInter2 = l_diff(Diff_Inter["Inter_diff2_rec"], gt_diff_inter)
        loss_diffInter_cross1 = l_diff(Diff_Inter["Inter_diff1_cross_rec"], gt_diff_inter_cross)
        loss_diffInter_cross2 = l_diff(Diff_Inter["Inter_diff2_cross_rec"], gt_diff_inter_cross)

        loss_diffInter = loss_diffInter1 + loss_diffInter2 + loss_diffInter_cross1 + loss_diffInter_cross2

        loss_diff = 0.5 * loss_diffItra + 0.5 * loss_diffInter
        loss = loss_rec + 0.01 * loss_diff

        loss.backward()
        optimizer.step()

        running_loss += loss.data.item()
        running_loss_50 = running_loss

        if idx % 50 == 0:
            running_loss_50 /= 50
            t.set_description(
                '[train epoch:%d] Rec_loss:%.8f DiffItra:%.8f DiffInter:%.8f' % (epoch + 1, loss_rec.item(), loss_diffItra.item(), loss_diffInter.item()))
            t.refresh()

    logging.info('epoch:%d iteration:%d running_loss:%.10f' % (epoch + 1, batches_done + 1, running_loss / num_train))
    scheduler.step()

    # -----------
    # Validating
    # -----------
    if rank == 0:
        with torch.no_grad():

            net.eval()
            rmse = np.zeros(len(test_dataloader))
            mae = np.zeros(len(test_dataloader))
            t = tqdm(iter(test_dataloader), leave=True, total=len(test_dataloader))
            device = torch.device("cuda", 0)
            for idx, data in enumerate(t):
                guidance, lr, gt = data['guidance'].to(device), data['lr'].to(device), data['gt'].to(device)
                restored, intermed, Diff_Intra, Diff_Inter = net.module.Video_Rec(lqs=lr, guides=guidance)

                restored = restored.to(device)
                gt = gt.squeeze()
                restored = restored.squeeze()

                rmse[idx] = calc_rmse_tartanair(gt, restored)
                mae[idx] = calc_mae_tartanair(gt, restored)

                t.set_description('[validate] rmse: %f' % rmse[:idx + 1].mean())
                t.refresh()

            r_mean = rmse.mean()
            mae_mean = mae.mean()
            if r_mean < best_rmse:
                best_rmse = r_mean
                best_epoch = epoch
                torch.save(net.module.Video_Rec.state_dict(),
                           os.path.join(opt.result_root, "RMSE%f_8%d.pth" % (best_rmse, best_epoch + 1)))
            if mae_mean < best_mae:
                torch.save(net.module.Video_Rec.state_dict(),
                           os.path.join(opt.result_root_MAE, "MAE%f_8%d.pth" % (mae_mean, epoch + 1)))
            logging.info(
                '---------------------------------------------------------------------------------------------------------------------------')
            logging.info('epoch:%d lr:%f-------mean_rmse:%f (BEST: %f @epoch%d)' % (
                epoch + 1, scheduler.get_last_lr()[0], r_mean, best_rmse, best_epoch + 1))
            logging.info('epoch:%d lr:%f-------mean_ae:%f ' % (
                epoch + 1, scheduler.get_last_lr()[0], mae_mean))
            logging.info(
                '---------------------------------------------------------------------------------------------------------------------------')

