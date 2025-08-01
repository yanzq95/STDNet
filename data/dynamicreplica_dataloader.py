import numpy as np
import os
import random
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image


def arugment(img, gt, lr, hflip=True, rot=True):
    hflip = hflip and random.random() < 0.5
    vflip = rot and random.random() < 0.5
    rot90 = rot and random.random() < 0.5

    if hflip:
        img = img[:, :, ::-1, :].copy()
        lr = lr[:, :, ::-1, :].copy()
        gt = gt[:, :, ::-1, :].copy()
    if vflip:
        img = img[:, ::-1, :, :].copy()
        lr = lr[:, ::-1, :, :].copy()
        gt = gt[:, ::-1, :, :].copy()
    if rot90:
        img = img.transpose(0, 2, 1, 3).copy()
        lr = lr.transpose(0, 2, 1, 3).copy()
        gt = gt.transpose(0, 2, 1, 3).copy()

    return img, gt, lr


def get_patch(img, gt, lr, scale, patch_size=16):
    th, tw = img.shape[1:3]  ## HR image

    tp = round(patch_size)

    tx = random.randrange(0, (tw - tp))
    ty = random.randrange(0, (th - tp))

    lr_tx = tx // scale
    lr_ty = ty // scale
    lr_tp = tp // scale

    return img[:, ty:ty + tp, tx:tx + tp, :], gt[:, ty:ty + tp, tx:tx + tp, :], lr[:, lr_ty:lr_ty + lr_tp,
                                                                                lr_tx:lr_tx + lr_tp, :]


class DynamicReplica_Dataset(Dataset):

    def __init__(self, root_path="/opt/data/private/dataset/VDSRDataset/DynamicReplica/", train=True,
                 txt_file='./DynamicReplica_test.txt', scale=16,
                 transform=None):

        self.root_path = root_path
        self.temp_offset = 0
        self.rgb_suffix = '.png'
        self.d_prefix = 'processed_'
        self.d_suffix = '.geometric.npy'
        self.num_input_frames = 7
        self.transform = transform
        self.train = train
        self.image_list = txt_file
        self.scale = scale
        self.ss = "X" + str(scale)

        self.data_infos, self.scene_name = self.load_annotations()

        with open(self.image_list, 'r') as f:
            self.filename = f.readlines()

    def get_cont_sub_sequence(
            self, seq_list, out_len, interval=1
    ):
        """
        This function takes a list of target sequences, and otuputs a continuous list of sub-sequences of desired length
        Input:
          seq_list: [x1, ..., xn]
          out_len: length of the desired output
          interval: the sampling interval for the start of each sub-sequence. default: 1
        Output:
          seq: the list of sub-sequences.
          For example: out_len=2 and interval=1 will return [[x1, x2], [x2, x3], ..., [xn-1, xn]]
        """
        idxs = (
            torch.Tensor(range(len(seq_list)))
            .type(torch.long)
            .view(1, -1)
            .unfold(1, size=out_len, step=interval)
            .squeeze(0)
        )
        seq = []
        for idxSet in idxs:
            inputs = [seq_list[idx_] for idx_ in idxSet]
            seq.append(inputs)
        return seq

    def get_cont_sub_sequence_from_lists(
            self, seq_list, out_len, interval=1
    ):
        """
        This function takes an input path, and otuputs a continuous list of sub-sequences of the files with desired length
        Input:
          seq_list: a list of sequences list
          out_len: length of the desired output
          interval: the sampling interval for the start of each sub-sequence. default: 1
        Output:
          seq: the list of sub-sequences.
          For example: out_len=2 and interval=1 will return [[x1, x2], [x2, x3], ..., [xn-1, xn]]
        """
        seq = []
        for item in seq_list:
            subs = self.get_cont_sub_sequence(item, out_len, interval)
            seq += subs
        return seq

    def load_annotations(self):
        with open(self.image_list, 'r') as f:
            seqlist = f.readlines()

        gt_seqlist = []
        lr_seqlist = []
        guide_seqlist = []

        scene_name = []

        if self.train:
            for seq in seqlist:
                seq = seq.strip('\n')
                seq = seq.split(", ")

                gt_seqlist.append([])
                lr_seqlist.append([])
                guide_seqlist.append([])

                name_s = seq[0] + "_" + seq[1]
                scene_name.append(name_s)

                for idx in range(self.temp_offset + 1, int(seq[2]) - self.temp_offset - 1):
                    gt_seqlist[-1].append(os.path.join(self.root_path, "HR_RGB", seq[0], seq[1], "depth", \
                                                       self.d_prefix + seq[1] + '_{:04d}'.format(idx) + self.d_suffix))
                    lr_seqlist[-1].append(os.path.join(self.root_path, "LR", self.ss, seq[0], seq[1], \
                                                       self.d_prefix + seq[1] + '_{:04d}'.format(idx) + self.d_suffix))
                    guide_seqlist[-1].append(os.path.join(self.root_path, "HR_RGB", seq[0], seq[1], "color", \
                                                          seq[1] + '-{:04d}'.format(idx) + self.rgb_suffix))
            data_infos = {}
            data_infos['gt_path'] = \
                self.get_cont_sub_sequence_from_lists(gt_seqlist, self.num_input_frames + 2 * self.temp_offset)
            data_infos['lr_path'] = \
                self.get_cont_sub_sequence_from_lists(lr_seqlist, self.num_input_frames + 2 * self.temp_offset)
            data_infos['guide_path'] = \
                self.get_cont_sub_sequence_from_lists(guide_seqlist, self.num_input_frames + 2 * self.temp_offset)
        else:
            for seq in seqlist:
                seq = seq.split(", ")
                """
                seq: seq[0]: scene name, seq[1]: subscene name,
                seq[2]: number of frames in the subscene
                """
                gt_seqlist.append([])
                lr_seqlist.append([])
                guide_seqlist.append([])

                name_s = seq[0] + "_" + seq[1]
                scene_name.append(name_s)

                for idx in range(self.temp_offset, int(seq[2])):
                    gt_seqlist[-1].append(os.path.join(self.root_path, "HR_RGB", seq[0], seq[1], "depth", \
                                                       self.d_prefix + seq[1] + '_{:04d}'.format(idx) + self.d_suffix))
                    lr_seqlist[-1].append(os.path.join(self.root_path, "LR", self.ss, seq[0], seq[1], \
                                                       self.d_prefix + seq[1] + '_{:04d}'.format(idx) + self.d_suffix))
                    guide_seqlist[-1].append(os.path.join(self.root_path, "HR_RGB", seq[0], seq[1], "color", \
                                                          seq[1] + '-{:04d}'.format(idx) + self.rgb_suffix))
            data_infos = {}
            data_infos['gt_path'] = gt_seqlist
            data_infos['lr_path'] = lr_seqlist
            data_infos['guide_path'] = guide_seqlist

        return data_infos, scene_name

    def load_and_stack(self, idx):

        gt_seqlist = self.data_infos['gt_path'][idx]
        lr_seqlist = self.data_infos['lr_path'][idx]
        guide_seqlist = self.data_infos['guide_path'][idx]

        scene_name_ = self.scene_name[idx]

        gt_frames = []
        lr_frames = []
        guide_frames = []

        for path in gt_seqlist:
            gt = np.load(path)
            gt_frames.append(gt)
        stacked_gt = np.stack(gt_frames, axis=0)  #

        for path in lr_seqlist:
            lr = np.load(path)
            lr = lr[0, :, :]
            lr_frames.append(lr)
        stacked_lr = np.stack(lr_frames, axis=0)  #

        for path in guide_seqlist:
            guide = np.array(Image.open(path).convert("RGB")).astype(np.float32)
            guide_frames.append(guide)
        stacked_guide = np.stack(guide_frames, axis=0)  #

        if self.train:
            stacked_gt = stacked_gt / 17700.0
            stacked_lr = stacked_lr / 17700.0
            stacked_guide = stacked_guide / 255.0

            stacked_guide, stacked_gt, stacked_lr = get_patch(img=stacked_guide, gt=np.expand_dims(stacked_gt, 3),
                                                              lr=np.expand_dims(stacked_lr, 3), scale=self.scale,
                                                              patch_size=256)  # ---------------------------------------
            stacked_guide, stacked_gt, stacked_lr = arugment(img=stacked_guide, gt=stacked_gt,
                                                             lr=stacked_lr)  # ---------------------------------------
        else:
            stacked_lr = stacked_lr / 17700.0
            stacked_guide = stacked_guide / 255.0

            stacked_lr = np.expand_dims(stacked_lr, 3)
            stacked_gt = np.expand_dims(stacked_gt, 3)

        return stacked_guide, stacked_gt, stacked_lr, scene_name_

    def __len__(self):
        return len(self.data_infos['gt_path'])

    def __getitem__(self, idx):

        stacked_guide, stacked_gt, stacked_lr, scene_name_ = self.load_and_stack(idx)

        if self.transform:
            stacked_guide = torch.from_numpy(stacked_guide).float()
            stacked_gt = torch.from_numpy(stacked_gt).float()
            stacked_lr = torch.from_numpy(stacked_lr).float()

            stacked_guide = stacked_guide.permute(0, 3, 1, 2).contiguous()
            stacked_gt = stacked_gt.permute(0, 3, 1, 2).contiguous()
            stacked_lr = stacked_lr.permute(0, 3, 1, 2).contiguous()

        sample = {'guidance': stacked_guide, 'lr': stacked_lr, 'gt': stacked_gt, 'name': scene_name_}

        return sample
