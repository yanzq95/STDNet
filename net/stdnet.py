from mmcv.runner import load_checkpoint
from torchvision.ops import deform_conv2d
import math
from torch.nn.modules.utils import _pair

from .common import PixelShufflePack, ResidualBlocksWithInputConv
from .common.diff import *
import torch.nn.functional as F

class KGenerator(nn.Module):
    def __init__(self, in_channels=64, kernel_size=3):
        super(KGenerator, self).__init__()
        self.kernel_size = kernel_size
        self.act = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.rg = ResidualGroup(default_conv, in_channels, kernel_size=3, reduction=16, n_resblocks=1)
        self.conv = nn.Conv2d(in_channels, in_channels * kernel_size * kernel_size, 1, 1, 0)

    def forward(self, input_x):
        b, c, h, w = input_x.size()
        x = self.rg(input_x)
        x = self.conv(self.act(x))
        filter_x = x.reshape([b, c, self.kernel_size * self.kernel_size, h, w])

        return filter_x


class SDM(nn.Module):

    def __init__(self, in_nfeats, out_nfeats, scale):
        super().__init__()
        self.scale = scale
        self.act = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.conv_diff = default_conv(out_nfeats, out_nfeats, 3)
        self.conv_du = nn.Conv2d(2, out_nfeats, kernel_size=3, padding=1, bias=True)
        self.kg = KGenerator(out_nfeats, 3)
        self.rg = ResidualGroup(default_conv, 2 * out_nfeats, kernel_size=3, reduction=16, n_resblocks=1)
        self.unfold = nn.Unfold(kernel_size=3, dilation=1, padding=1, stride=1)

        self.sigmoid = nn.Sigmoid()

        self.beta = nn.Parameter(torch.ones(1) * 0.1)
        self.conv_re = default_conv(2 * out_nfeats, out_nfeats, 1)

    def forward(self, dep_feat, rgb_feat):

        b, c, h, w = dep_feat.shape

        diff_intra = torch.abs(dep_feat - F.interpolate(
            F.interpolate(dep_feat, (h // self.scale, w // self.scale), mode='bilinear', align_corners=False),
            (h, w), mode='bilinear', align_corners=False))

        diff_intra_f = self.conv_diff(self.act(diff_intra))
        filter_x = self.kg(diff_intra_f)
        unfold_x = self.unfold(rgb_feat).reshape(b, c, -1, h, w)
        out_rgb = (unfold_x * filter_x).sum(2)

        dif_avg = torch.mean(diff_intra, dim=1, keepdim=True)
        dif_max, _ = torch.max(diff_intra, dim=1, keepdim=True)
        xx = self.conv_du(torch.cat([dif_avg, dif_max], dim=1))
        attention = self.sigmoid(xx)

        rgb_feat2 = attention * rgb_feat + self.beta * out_rgb

        cat_dr = torch.cat([dep_feat, rgb_feat2], dim=1)
        out_rg = self.rg(cat_dr)
        out_re = self.conv_re(out_rg)

        return out_re, attention, diff_intra


class TDM(nn.Module):

    def __init__(self, nfeats):
        super().__init__()
        self.nfeats = nfeats
        self.kernel_size = _pair(3)
        self.conv_du = nn.Conv2d(2 * nfeats, nfeats, kernel_size=1, padding=0, bias=True)
        self.conv_diff = default_conv(nfeats, nfeats, 3)
        self.conv_offset_mask = nn.Conv2d(
            nfeats, 3 * 9, kernel_size=3, stride=1,
            padding=1, bias=True,
        )

        self.w = nn.Parameter(torch.Tensor(nfeats, nfeats, 3, 3))
        self.b = nn.Parameter(torch.Tensor(nfeats))

        self._reset_parameters()
        self._init_offset()

        self.stride = 1
        self.padding = 1
        self.dilation = 1
        self.groups = 1
        self.deformable_groups = 1

        self.act = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.conv_outDCN = default_conv(nfeats, nfeats, 3)

        self.rg1 = ResidualGroup(default_conv, 3 * nfeats, kernel_size=3, reduction=16, n_resblocks=2)
        self.conv_re = default_conv(3 * nfeats, nfeats, 1)

    def _reset_parameters(self):
        n = self.nfeats
        for k in self.kernel_size:
            n *= k
        stdv = 1. / math.sqrt(n)
        self.w.data.uniform_(-stdv, stdv)
        if self.b is not None:
            self.b.data.zero_()

    def _init_offset(self):
        self.conv_offset_mask.weight.data.zero_()
        self.conv_offset_mask.bias.data.zero_()

    def _get_offset_mask(self, diff_maps):
        offset_aff = self.conv_offset_mask(diff_maps)
        o1, o2, mask = torch.chunk(offset_aff, 3, dim=1)
        offset = torch.cat((o1, o2), dim=1)
        mask = torch.sigmoid(mask)

        return offset, mask

    def forward(self, curr_frame, ngb_frame, diff_maps, ngb_frame_rgb, intra_attention):

        diff_map1 = self.conv_diff(diff_maps)
        cat1 = torch.cat([curr_frame, diff_map1], dim=1)
        diff_map2 = self.conv_du(cat1)

        offset, mask = self._get_offset_mask(diff_map2)

        ngb_out_dep = deform_conv2d(ngb_frame, offset, self.w, self.b, self.stride, self.padding,
                                    self.dilation, mask=mask)
        ngb_out_dep2 = self.conv_outDCN(self.act(ngb_out_dep))

        ngb_out_rgb = deform_conv2d(ngb_frame_rgb, offset, self.w, self.b, self.stride, self.padding,
                                    self.dilation, mask=mask)
        ngb_out_rgb = ngb_out_rgb * intra_attention

        cat_dr = torch.cat([curr_frame, ngb_out_dep2, ngb_out_rgb], dim=1)
        out_rg = self.rg1(cat_dr)
        out_re = self.conv_re(out_rg)
        return out_re


class VDSR_Rec(nn.Module):
    def __init__(
            self,
            mid_channels=64,
            num_blocks=7,
            scale=16,
            max_residue_magnitude=10,
            is_low_res_input=True,
            spynet_pretrained=None,
            cpu_cache_length=200,
    ):

        super().__init__()
        self.mid_channels = mid_channels
        self.is_low_res_input = is_low_res_input
        self.scale = scale
        self.cpu_cache_length = cpu_cache_length

        # feature extraction module
        self.conv_guide_init = nn.ModuleDict()
        self.conv_guide_init["hg_1"] = nn.Sequential(
            nn.Conv2d(3, mid_channels, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            ResidualBlocksWithInputConv(mid_channels, mid_channels, 1),
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            ResidualBlocksWithInputConv(mid_channels, mid_channels, 2),
        )
        self.conv_guide_init["hg_2"] = nn.Sequential(
            nn.Conv2d(2, mid_channels, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            ResidualBlocksWithInputConv(mid_channels, mid_channels, 1),
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            ResidualBlocksWithInputConv(mid_channels, mid_channels, 2),
        )

        self.conv_dep_init = nn.ModuleDict()
        self.conv_dep_init["hg_1"] = nn.Sequential(
            nn.Conv2d(1, mid_channels, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            ResidualBlocksWithInputConv(mid_channels, mid_channels, 1),
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            ResidualBlocksWithInputConv(mid_channels, mid_channels, 2),
        )
        self.conv_dep_init["hg_2"] = nn.Sequential(
            nn.Conv2d(1, mid_channels, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            ResidualBlocksWithInputConv(mid_channels, mid_channels, 1),
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            ResidualBlocksWithInputConv(mid_channels, mid_channels, 2),
        )

        self.feat_extract_f = nn.ModuleDict()
        self.feat_extract_f["hg_1"] = SDM(
            mid_channels, mid_channels, 2
        )
        self.feat_extract_f["hg_2"] = SDM(
            mid_channels, mid_channels, 2
        )

        self.feat_extract = nn.ModuleDict()
        self.feat_extract["hg_1"] = ResidualBlocksWithInputConv(
            mid_channels, mid_channels, 5
        )
        self.feat_extract["hg_2"] = ResidualBlocksWithInputConv(
            mid_channels * 2, mid_channels, 5
        )

        # propagation branches
        self.deform_align = nn.ModuleDict()
        self.deform_align["hg_1"] = nn.ModuleDict()
        self.deform_align["hg_2"] = nn.ModuleDict()

        modules = ["backward_1", "forward_1", "backward_2", "forward_2"]
        for i, module in enumerate(modules):
            self.deform_align["hg_1"][module] = TDM(mid_channels)
            self.deform_align["hg_2"][module] = TDM(mid_channels)

        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        # upsampling module
        self.reconstruction = nn.ModuleDict()
        self.reconstruction["hg_1"] = ResidualBlocksWithInputConv(
            5 * mid_channels, mid_channels, 5
        )
        self.reconstruction["hg_2"] = ResidualBlocksWithInputConv(
            5 * mid_channels, mid_channels, 5
        )

        self.final_pred = nn.ModuleDict()
        self.final_pred["hg_1"] = nn.Sequential(
            PixelShufflePack(mid_channels, mid_channels, 2, upsample_kernel=3),
            self.lrelu,
            PixelShufflePack(mid_channels, 64, 2, upsample_kernel=3),
            self.lrelu,
            nn.Conv2d(64, 64, 3, 1, 1),
            self.lrelu,
            nn.Conv2d(64, 2, 3, 1, 1),
        )

        self.final_pred["hg_2"] = nn.Sequential(
            PixelShufflePack(mid_channels, mid_channels, 2, upsample_kernel=3),
            self.lrelu,
            PixelShufflePack(mid_channels, 64, 2, upsample_kernel=3),
            self.lrelu,
            nn.Conv2d(64, 64, 3, 1, 1),
            self.lrelu,
            nn.Conv2d(64, 2, 3, 1, 1),
        )

        self.img_upsample = nn.Upsample(
            scale_factor=4, mode="bilinear", align_corners=False
        )
        self.softmax = nn.Softmax(dim=2)

    def compute_interframe_diff_maps(self, deps0):

        deps = torch.stack(deps0, dim=1)

        n, t, c, h, w = deps.size()

        deps_1 = deps[:, :t - 1, :, :, :]
        deps_2 = deps[:, 1:, :, :, :]

        deps_3 = deps[:, :t - 2, :, :, :]
        deps_4 = deps[:, 2:, :, :, :]

        if self.cpu_cache:
            diff_maps = []
            for tt in range(t - 1):
                dm = torch.abs(deps_2[:, tt, :, :, :] - deps_1[:, tt, :, :, :])
                diff_maps.append(dm.unsqueeze(1))
            diff_maps = torch.cat(diff_maps, dim=1)

            diff_maps_cross = []
            for tt in range(t - 2):
                dm_cross = torch.abs(deps_4[:, tt, :, :, :] - deps_3[:, tt, :, :, :])
                diff_maps_cross.append(dm_cross.unsqueeze(1))
            diff_maps_cross = torch.cat(diff_maps_cross, dim=1)

        else:
            deps_1 = deps_1.reshape(-1, c, h, w)
            deps_2 = deps_2.reshape(-1, c, h, w)
            diff_maps = torch.abs(deps_1 - deps_2).view(n, t - 1, c, h, w)

            deps_3 = deps_3.reshape(-1, c, h, w)
            deps_4 = deps_4.reshape(-1, c, h, w)
            diff_maps_cross = torch.abs(deps_4 - deps_3).view(n, t - 2, c, h, w)

        if self.cpu_cache:
            diff_maps = diff_maps.cpu()
            diff_maps_cross = diff_maps_cross.cpu()

            torch.cuda.empty_cache()

        return diff_maps, diff_maps_cross

    def propagate(self, feats, diff_maps, diff_maps_cross, att_intra_arr, rgb_feats, module_name, hg_idx):

        n, t, _, h, w = diff_maps.size()

        if "backward" in module_name:
            backward_fs = [None] * (t + 1)

            t_feats = feats["spatial"][-1]

            rgb_feats_t = rgb_feats["rgb"][-1]
            att_intra_t = att_intra_arr["att_intra"][-2]

            t_feats_ = feats["spatial"][-2]
            t_diffs = diff_maps[:, -1, :, :, :]
            if self.cpu_cache:
                t_feats = t_feats.cuda()
                t_feats_ = t_feats_.cuda()
                t_diffs = t_diffs.cuda()
                rgb_feats_t = rgb_feats_t.cuda()
                att_intra_t = att_intra_t.cuda()

            backward_fs[-1] = t_feats
            t_1_feats = self.deform_align[f"hg_{hg_idx}"][module_name](t_feats_, t_feats, t_diffs, rgb_feats_t,
                                                                       att_intra_t)
            backward_fs[-2] = t_1_feats

            for i in range(0, t - 1):
                feat_current = feats["spatial"][i]
                att_current = att_intra_arr["att_intra"][i]

                feat_ngb1 = feats["spatial"][i + 1]
                rgb_ngb1 = rgb_feats["rgb"][i + 1]

                feat_ngb2 = feats["spatial"][i + 2]
                rgb_ngb2 = rgb_feats["rgb"][i + 2]

                diff_maps1 = diff_maps[:, i, :, :, :]
                diff_maps_cross1 = diff_maps_cross[:, i, :, :, :]
                if self.cpu_cache:
                    feat_current = feat_current.cuda()
                    att_current = att_current.cuda()

                    feat_ngb1 = feat_ngb1.cuda()
                    rgb_ngb1 = rgb_ngb1.cuda()

                    feat_ngb2 = feat_ngb2.cuda()
                    rgb_ngb2 = rgb_ngb2.cuda()

                    diff_maps1 = diff_maps1.cuda()
                    diff_maps_cross1 = diff_maps_cross1.cuda()
                inter_feats1 = self.deform_align[f"hg_{hg_idx}"][module_name](feat_current, feat_ngb1, diff_maps1,
                                                                              rgb_ngb1, att_current)
                inter_feats2 = self.deform_align[f"hg_{hg_idx}"][module_name](feat_current, feat_ngb2, diff_maps_cross1,
                                                                              rgb_ngb2, att_current)
                backward_fs[i] = inter_feats1 + inter_feats2

            if self.cpu_cache:
                backward_fs = [tensor.cpu() for tensor in backward_fs]
                torch.cuda.empty_cache()

            feats[module_name].append(backward_fs)
            feats["spatial"] = [backward_fs[i] for i in range(0, t + 1)]
        else:
            forward_fs = [None] * (t + 1)
            t_feats = feats["spatial"][0]
            rgb_feats_t = rgb_feats["rgb"][0]

            t_feats_ = feats["spatial"][1]
            att_intra_t = att_intra_arr["att_intra"][1]

            t_diffs = diff_maps[:, 0, :, :, :]
            if self.cpu_cache:
                t_feats = t_feats.cuda()
                t_feats_ = t_feats_.cuda()
                t_diffs = t_diffs.cuda()
                rgb_feats_t = rgb_feats_t.cuda()
                att_intra_t = att_intra_t.cuda()

            forward_fs[0] = t_feats
            t1_feats = self.deform_align[f"hg_{hg_idx}"][module_name](t_feats_, t_feats, t_diffs, rgb_feats_t,
                                                                      att_intra_t)
            forward_fs[1] = t1_feats

            for i in range(2, t + 1):
                feat_current = feats["spatial"][i]
                att_current = att_intra_arr["att_intra"][i]

                feat_ngb1 = feats["spatial"][i - 1]
                rgb_ngb1 = rgb_feats["rgb"][i - 1]

                feat_ngb2 = feats["spatial"][i - 2]
                rgb_ngb2 = rgb_feats["rgb"][i - 2]

                diff_maps1 = diff_maps[:, i - 1, :, :, :]
                diff_maps_cross1 = diff_maps_cross[:, i - 2, :, :, :]
                if self.cpu_cache:
                    feat_current = feat_current.cuda()
                    att_current = att_current.cuda()

                    feat_ngb1 = feat_ngb1.cuda()
                    rgb_ngb1 = rgb_ngb1.cuda()

                    feat_ngb2 = feat_ngb2.cuda()
                    rgb_ngb2 = rgb_ngb2.cuda()

                    diff_maps1 = diff_maps1.cuda()
                    diff_maps_cross1 = diff_maps_cross1.cuda()
                inter_feats1 = self.deform_align[f"hg_{hg_idx}"][module_name](feat_current, feat_ngb1, diff_maps1,
                                                                              rgb_ngb1, att_current)
                inter_feats2 = self.deform_align[f"hg_{hg_idx}"][module_name](feat_current, feat_ngb2, diff_maps_cross1,
                                                                              rgb_ngb2, att_current)
                forward_fs[i] = inter_feats1 + inter_feats2

            if self.cpu_cache:
                forward_fs = [tensor.cpu() for tensor in forward_fs]
                torch.cuda.empty_cache()

            feats[module_name].append(forward_fs)
            feats["spatial"] = [forward_fs[i] for i in range(0, t + 1)]

        return feats

    def upsample(self, lqs, feats, hg_idx):
        depths = []
        confs = []
        feats_fused = []
        num_outputs = len(feats["spatial_0"])

        mapping_idx = list(range(0, num_outputs))
        mapping_idx += mapping_idx[::-1]

        for i in range(0, lqs.size(1)):
            hr = [feats[k][0].pop(0) for k in feats if (k != "spatial" and k != "spatial_0")]

            hr.insert(0, feats["spatial_0"][mapping_idx[i]])
            hr = torch.cat(hr, dim=1)
            if self.cpu_cache:
                hr = hr.cuda()
            hr = self.reconstruction[f"hg_{hg_idx}"](hr)
            
            feat_fused = hr.clone()
            hr = self.final_pred[f"hg_{hg_idx}"](hr)

            depth, conf = torch.chunk(hr, 2, dim=1)
            depth = depth + self.img_upsample(lqs[:, i, :, :, :])
            if self.cpu_cache:
                hr = hr.cpu()
                depth = depth.cpu()
                conf = conf.cpu()
                torch.cuda.empty_cache()

            depths.append(depth)
            confs.append(conf)
            feats_fused.append(feat_fused)

        return (
            torch.stack(depths, dim=1),
            torch.stack(confs, dim=1),
            torch.stack(feats_fused, dim=1),
        )

    def hg_forward(self, lqs, guides, extra_inputs=None, extra_feats=None, hg_idx=1):

        n, t, c, h, w = lqs.size()

        # whether to cache the features in CPU (no effect if using CPU)
        if t > self.cpu_cache_length and lqs.is_cuda:
            self.cpu_cache = True
        else:
            self.cpu_cache = False

        feats = {}
        att_intra_arr = {}
        rgb_feats = {}
        Intra_diff_list = {}

        if self.cpu_cache:
            feats["spatial"] = []
            feats["spatial_0"] = []
            att_intra_arr["att_intra"] = []
            Intra_diff_list["intra_diff"] = []
            rgb_feats["rgb"] = []
            for i in range(0, t):
                if hg_idx == 1:
                    guide_feat = self.conv_guide_init[f"hg_{hg_idx}"](
                        guides[:, i, :, :, :]
                    )
                    dep_feat = self.conv_dep_init[f"hg_{hg_idx}"](
                        lqs[:, i, :, :, :]
                    )
                    feat_f, att_intra, diff_intra = self.feat_extract_f[f"hg_{hg_idx}"](dep_feat, guide_feat)
                    feat = self.feat_extract[f"hg_{hg_idx}"](feat_f).cpu()
                else:
                    guide_feat = self.conv_guide_init[f"hg_{hg_idx}"](
                        extra_inputs[:, i, :, :, :].to(guides.device)
                    )
                    dep_feat = self.conv_dep_init[f"hg_{hg_idx}"](
                        lqs[:, i, :, :, :]
                    )
                    feat_f, att_intra, diff_intra = self.feat_extract_f[f"hg_{hg_idx}"](dep_feat, guide_feat)
                    feat = self.feat_extract[f"hg_{hg_idx}"](
                        torch.cat(
                            [
                                feat_f,
                                extra_feats[:, i, :, :, :],
                            ],
                            dim=1,
                        )
                    ).cpu()
                feats["spatial"].append(feat)
                feats["spatial_0"].append(feat)
                att_intra_arr["att_intra"].append(att_intra.cpu())
                Intra_diff_list["intra_diff"].append(diff_intra.cpu())
                rgb_feats["rgb"].append(guide_feat.cpu())
                torch.cuda.empty_cache()
            Intra_diff_rec = torch.stack(Intra_diff_list["intra_diff"], dim=1)
        else:
            if hg_idx == 1:
                guide_feats_ = self.conv_guide_init[f"hg_{hg_idx}"](
                    guides.view(-1, 3, int(h * 4), int(w * 4))
                )
                dep_feats_ = self.conv_dep_init[f"hg_{hg_idx}"](
                    lqs.view(-1, 1, h, w)
                )
                feats_f, att_intra, diff_intra = self.feat_extract_f[f"hg_{hg_idx}"](dep_feats_, guide_feats_)
                feats_ = self.feat_extract[f"hg_{hg_idx}"](feats_f)
            else:
                guide_feats_ = self.conv_guide_init[f"hg_{hg_idx}"](
                    extra_inputs.view(-1, 2, int(h * 4), int(w * 4))
                )
                dep_feats_ = self.conv_dep_init[f"hg_{hg_idx}"](
                    lqs.view(-1, 1, h, w)
                )
                feats_f, att_intra, diff_intra = self.feat_extract_f[f"hg_{hg_idx}"](dep_feats_, guide_feats_)
                feats_ = self.feat_extract[f"hg_{hg_idx}"](
                    torch.cat(
                        [
                            feats_f,
                            extra_feats.view(-1, self.mid_channels, h, w),
                        ],
                        dim=1,
                    )
                )
            h, w = feats_.shape[2:]
            diff_h, diff_w = diff_intra.shape[2:]
            feats_ = feats_.view(n, t, -1, h, w)
            att_intra = att_intra.view(n, t, -1, h, w)
            Intra_diff_rec = diff_intra.view(n, t, -1, diff_h, diff_w)
            guide_feats_ = guide_feats_.view(n, t, -1, h, w)
            feats["spatial"] = [feats_[:, i, :, :, :] for i in range(0, t)]
            feats["spatial_0"] = [feats_[:, i, :, :, :] for i in range(0, t)]
            att_intra_arr["att_intra"] = [att_intra[:, i, :, :, :] for i in range(0, t)]
            rgb_feats["rgb"] = [guide_feats_[:, i, :, :, :] for i in range(0, t)]

        diff_maps, diff_maps_cross = self.compute_interframe_diff_maps(feats["spatial_0"])

        # feature propagation
        for iter_ in [1, 2]:
            for direction in ["backward", "forward"]:
                module = f"{direction}_{iter_}"

                feats[module] = []

                feats = self.propagate(feats, diff_maps, diff_maps_cross, att_intra_arr, rgb_feats, module, hg_idx)

        depth, conf, feats_fused = self.upsample(lqs, feats, hg_idx)
        if hg_idx == 1:
            return depth, conf, feats_fused, Intra_diff_rec, diff_maps, diff_maps_cross
        else:
            return depth, conf, None, Intra_diff_rec, diff_maps, diff_maps_cross

    def forward(self, lqs, guides):
        lqs = lqs.repeat_interleave(self.scale // 4, dim=3).repeat_interleave(self.scale // 4, dim=4)
        n, t, c, h, w = lqs.size()

        rgb_depth, rgb_conf, rgb_feats, Intra_diff_rec1, Diff_Inter_rec_1, Diff_Inter_rec_cross_1 = self.hg_forward(lqs,
                                                                                                                    guides,
                                                                                                                    hg_idx=1)
        d_depth, d_conf, _, Intra_diff_rec2, Diff_Inter_rec_2, Diff_Inter_rec_cross_2 = self.hg_forward(
            lqs, guides, torch.cat((rgb_depth, rgb_conf), dim=2), rgb_feats, hg_idx=2
        )

        rgb_conf, d_conf = torch.chunk(
            self.softmax(
                torch.cat(
                    (
                        rgb_conf,
                        d_conf,
                    ),
                    dim=2,
                )
            ),
            2,
            dim=2,
        )

        depth_final = d_depth * d_conf + rgb_depth * rgb_conf
        intermed = {
            "d_depth": d_depth,
            "rgb_depth": rgb_depth,
            "d_conf": d_conf,
            "rgb_conf": rgb_conf,
        }

        Diff_Intra = {"Diff_Intra1": Intra_diff_rec1, "Diff_Intra2": Intra_diff_rec2}

        Diff_Inter = {"Diff_Inter1": Diff_Inter_rec_1, "Diff_Inter2": Diff_Inter_rec_2,
                      "Diff_Inter_cross1": Diff_Inter_rec_cross_1, "Diff_Inter_cross2": Diff_Inter_rec_cross_2}

        return depth_final, intermed, Diff_Intra, Diff_Inter

    def init_weights(self, pretrained=None, strict=True):
        if isinstance(pretrained, str):
            load_checkpoint(self, pretrained, strict=strict)
        elif pretrained is not None:
            raise TypeError(
                f'"pretrained" must be a str or None. '
                f"But received {type(pretrained)}."
            )


class Up_Diff(nn.Module):
    def __init__(self, scale, n_feat, kernel_size=1, bias=False, cpu_cache_length=200):
        super(Up_Diff, self).__init__()
        self.scale = scale
        self.cpu_cache_length = cpu_cache_length
        self.conv2 = default_conv(n_feat, 1, kernel_size, bias=bias)
        self.con1x1 = default_conv(1, 1, 1, bias=bias)

    def forward(self, x):
        n, t, c, h, w = x.shape
        rec_diff = [None] * t
        x = x.cuda()
        new_h, new_w = h * 4, w * 4
        if t > self.cpu_cache_length:
            for i in range(0, t):
                xx = x[:, i, :, :, :]

                img = self.conv2(xx)

                img2 = F.interpolate(img, size=(new_h, new_w), mode='bicubic', align_corners=False)
                reci = self.con1x1(img2)
                rec_diff[i] = reci
            rec = torch.stack(rec_diff, dim=1)
        else:
            img = self.conv2(x.view(-1, c, h, w))
            img2 = F.interpolate(img, size=(new_h, new_w), mode='bicubic', align_corners=False)
            rec = self.con1x1(img2).view(n, t, 1, new_h, new_w)

        return rec

class Diff_Rec(nn.Module):
    def __init__(self, scale, n_feat, cpu_cache_length):
        super(Diff_Rec, self).__init__()

        self.upD_Intra = Up_Diff(scale, n_feat, 1, False, cpu_cache_length)

        self.upD_Inter = Up_Diff(scale, n_feat, 1, False, cpu_cache_length)
        self.upD_Inter_cross = Up_Diff(scale, n_feat, 1, False, cpu_cache_length)

    def forward(self, IntraD, InterD):
        Intra_diff1 = IntraD["Diff_Intra1"]
        Intra_diff2 = IntraD["Diff_Intra2"]

        Inter_diff1 = InterD["Diff_Inter1"]
        Inter_diff2 = InterD["Diff_Inter2"]

        Inter_diff1_cross = InterD["Diff_Inter_cross1"]
        Inter_diff2_cross = InterD["Diff_Inter_cross2"]

        Intra_diff1_rec = self.upD_Intra(Intra_diff1)
        Intra_diff2_rec = self.upD_Intra(Intra_diff2)

        Inter_diff1_rec = self.upD_Inter(Inter_diff1)
        Inter_diff2_rec = self.upD_Inter(Inter_diff2)

        Inter_diff1_cross_rec = self.upD_Inter_cross(Inter_diff1_cross)
        Inter_diff2_cross_rec = self.upD_Inter_cross(Inter_diff2_cross)

        Intra_diff_rec = {"Intra_diff1_rec": Intra_diff1_rec, "Intra_diff2_rec": Intra_diff2_rec}
        Inter_diff_rec = {"Inter_diff1_rec": Inter_diff1_rec, "Inter_diff2_rec": Inter_diff2_rec,
                          "Inter_diff1_cross_rec": Inter_diff1_cross_rec,
                          "Inter_diff2_cross_rec": Inter_diff2_cross_rec}

        return Intra_diff_rec, Inter_diff_rec


class STDNet(nn.Module):
    def __init__(self, mid_channels=64, num_blocks=7, scale=16, max_residue_magnitude=10, is_low_res_input=True,
                 spynet_pretrained=None, cpu_cache_length=200):
        super(STDNet, self).__init__()

        self.Video_Rec = VDSR_Rec(mid_channels, num_blocks, scale, max_residue_magnitude, is_low_res_input,
                                  spynet_pretrained, cpu_cache_length)

        self.Diff_Rec = Diff_Rec(scale, mid_channels, cpu_cache_length)

    def forward(self, lqs, guides):
        depth_final, intermed, Diff_Intra, Diff_Inter = self.Video_Rec(lqs, guides)
        Intra_diff_rec, Inter_diff_rec = self.Diff_Rec(Diff_Intra, Diff_Inter)

        return depth_final, intermed, Intra_diff_rec, Inter_diff_rec