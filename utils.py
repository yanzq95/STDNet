import torch

def calc_rmse_dynamicR(gt, out):
    out = out * 17700.0

    out = out / 100.0
    gt = gt / 100.0

    return torch.sqrt(torch.mean(torch.pow(gt - out, 2)))

def calc_mae_dynamicR(gt, out):
    out = out * 17700.0

    out = out / 100.0
    gt = gt / 100.0

    return torch.mean(torch.abs(out - gt))

def calc_rmse_tartanair(gt, out):

    out = out * 30.0
    
    out = out * 100.0
    gt = gt * 100.0

    return torch.sqrt(torch.mean(torch.pow(gt-out,2)))
    
def calc_mae_tartanair(gt, out):

    out = out * 30.0
    
    out = out * 100.0
    gt = gt * 100.0

    return torch.mean(torch.abs(out - gt))
    
def calc_rmse_dydtof(gt, out):

    out = out * 40.0
    
    out = out * 100.0
    gt = gt * 100.0

    return torch.sqrt(torch.mean(torch.pow(gt-out,2)))
    
def calc_mae_dydtof(gt, out):

    out = out * 40.0
    
    out = out * 100.0
    gt = gt * 100.0

    return torch.mean(torch.abs(out - gt))