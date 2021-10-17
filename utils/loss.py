import torch
import torch.nn.functional as F


def weighted_class_bceloss(output, target, weights=None):    
    
    if weights is not None:
        assert len(weights) == 2

        # loss = weights[1] * (target * torch.log(output)) + \
        #        weights[0] * ((1 - target) * torch.log(1 - output))
        loss = weights[1] * target * torch.log(torch.clamp(output, min=1e-15, max=1)) + \
               weights[0] * (1 - target) * torch.log(torch.clamp(1 - output, min=1e-15, max=1))

    else:
        loss = target * torch.log(output) + (1 - target) * torch.log(1 - output)
   
    return torch.neg(torch.mean(loss))
    '''
    crit = torch.nn.BCELoss()
    loss = crit(output, target)
    return torch.mean(loss)
    '''

def vae_loss_function(recon_x, x, mu, logvar):
    recon_x = recon_x.squeeze()
    x = x.squeeze()
    BCE = F.binary_cross_entropy(recon_x, x, size_average=False)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD


def l1_penalty(para):
    return torch.nn.L1Loss()(para, torch.zeros_like(para))