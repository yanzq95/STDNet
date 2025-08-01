import argparse
from utils import *
import torchvision.transforms as transforms

from net.stdnet import STDNet as Net

from data.dydtof_dataloader import *

import torch

parser = argparse.ArgumentParser()
parser.add_argument('--scale', type=int, default=4, help='scale factor')
parser.add_argument("--root_dir", type=str, default='/opt/data/private/dataset', help="root dir of dataset")
parser.add_argument("--model_dir", type=str, default="./CKPT/X4.pth", help="path of net")
parser.add_argument("--results_dir", type=str, default='./Visual_Results/X4/', help="root dir of results")
opt = parser.parse_args()

net = Net(scale=opt.scale).Video_Rec.cuda()

net.load_state_dict(torch.load(opt.model_dir, map_location='cuda:0'))
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net.to(device)

data_transform = transforms.Compose([transforms.ToTensor()])

dataset_name = opt.root_dir.split('/')[-1]
 
dataset = DyDToF_Dataset(root_path="./dataset/DyDToF/", train=False, txt_file="./data/dydtof_list/school_shot8.txt", scale=opt.scale, transform=data_transform)

dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=8)

data_num = len(dataloader)
mae_avg = np.zeros(data_num)
rmse_avg = np.zeros(data_num)

with torch.no_grad():
    net.eval()
    device = torch.device("cuda", 0)
    for idx, data in enumerate(dataloader):
        
        guidance, lr, gt, name = data['guidance'].cuda(), data['lr'].cuda(), data['gt'].cuda(), data['name']
        
        print('%s ******************************** %d' %(name[0], gt.shape[1]))
        restored, intermed,Diff_Intra, Diff_Inter = net(lqs=lr, guides=guidance)
        restored = restored.to(device)
        gt = gt.squeeze()
        restored = restored.squeeze()

        mae_avg[idx] = calc_mae_dydtof(gt, restored)
        rmse_avg[idx] = calc_rmse_dydtof(gt, restored)
        
        #path_output = '{}/{}'.format(opt.results_dir, name[0])
        #os.makedirs(path_output, exist_ok=True)
        #for i in range(0, restored.size(0)):
            #output_i = restored[i, :, :] * 40.0            
            #path_save_pred = '{}/school_shot8.{:04d}.npy'.format(path_output, i)
            #np.save(path_save_pred, output_i.detach().cpu().numpy())

        print('Scene:%s Seq_len:%d AE:%.2f' % (name[0], restored.shape[0], mae_avg[idx]))
        print('Scene:%s Seq_len:%d RMSE:%.2f' % (name[0], restored.shape[0], rmse_avg[idx]))
        print("***************************************************")



