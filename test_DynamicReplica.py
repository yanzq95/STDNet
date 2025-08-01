import argparse

from utils import *
import torchvision.transforms as transforms

from net.stdnet import STDNet as Net
from data.dynamicreplica_dataloader import *

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
dataset = DynamicReplica_Dataset(root_path="/opt/data/private/dataset/VDSRDataset/DynamicReplica/", train=False, txt_file="./data/DynamicReplica_list/029beb.txt", scale=opt.scale, transform=data_transform)
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

        restored, intermed, Diff_Intra, Diff_Inter = net(lqs=lr, guides=guidance)
        restored = restored.to(device)
        gt = gt.squeeze()
        restored = restored.squeeze()

        mae_avg[idx] = calc_mae_dynamicR(gt, restored)
        rmse_avg[idx] = calc_rmse_dynamicR(gt, restored)

        # path_output = '{}/{}'.format(opt.results_dir, name[0])
        # os.makedirs(path_output, exist_ok=True)
        # for i in range(0, restored.size(0)):
        #     output_i = restored[i, :, :] * 17700.0
        #     path_save_pred = '{}/processed_029beb-3_obj_source_left_{:04d}.geometric.npy'.format(path_output, i)# processed_029beb-3_obj_source_left_   processed_01f258-3_obj_source_left_
        #     np.save(path_save_pred, output_i.detach().cpu().numpy())

        print('Scene:%s Seq_len:%d AE:%.2f' % (name[0], restored.shape[0], mae_avg[idx]))
        print('Scene:%s Seq_len:%d RMSE:%.2f' % (name[0], restored.shape[0], rmse_avg[idx]))
        print("***************************************************")


