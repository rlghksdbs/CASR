import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision
import segmentation_models_pytorch as smp

class MeanShift(nn.Conv2d):
    def __init__(self, data_mean, data_std, data_range=1, norm=True):
        c = len(data_mean)
        super(MeanShift, self).__init__(c, c, kernel_size=1)
        std = torch.Tensor(data_std)
        self.weight.data = torch.eye(c).view(c, c, 1, 1)
        if norm:
            self.weight.data.div_(std.view(c, 1, 1, 1))
            self.bias.data = -1 * data_range * torch.Tensor(data_mean)
            self.bias.data.div_(std)
        else:
            self.weight.data.mul_(std.view(c, 1, 1, 1))
            self.bias.data = data_range * torch.Tensor(data_mean)
        self.requires_grad = False

class VGGPerceptualLoss(torch.nn.Module):
    def __init__(self, rank=0):
        super(VGGPerceptualLoss, self).__init__()
        blocks = []
        pretrained = True
        self.vgg_pretrained_features = torchvision.models.vgg19(pretrained=pretrained).features
        self.normalize = MeanShift([0.485, 0.456, 0.406], [0.229, 0.224, 0.225], norm=True).cuda()
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, X, Y, indices=None):
        X = self.normalize(X)
        Y = self.normalize(Y)
        indices = [2, 7, 12, 21, 30]
        weights = [1.0/2.6, 1.0/4.8, 1.0/3.7, 1.0/5.6, 10/1.5]
        k = 0
        loss = 0
        for i in range(indices[-1]):
            X = self.vgg_pretrained_features[i](X)
            Y = self.vgg_pretrained_features[i](Y)
            if (i+1) in indices:
                loss += weights[k] * (X - Y.detach()).abs().mean() * 0.1
                k += 1
        return loss
        
class VGG(nn.Module):
    def __init__(self, device):
        super(VGG, self).__init__()
        vgg16 = torchvision.models.vgg16(pretrained=True).to(device)
        self.vgg16_conv_4_3 = torch.nn.Sequential(*list(vgg16.children())[0][:22])
        for param in self.vgg16_conv_4_3.parameters():
            param.requires_grad = False

    def forward(self, output, gt):
        vgg_output = self.vgg16_conv_4_3(output)
        with torch.no_grad():
            vgg_gt = self.vgg16_conv_4_3(gt.detach())

        loss = F.mse_loss(vgg_output, vgg_gt)

        return loss
        
# class VGG_gram(nn.Module):
#     def __init__(self):
#         super(VGG_gram, self).__init__()
#         vgg16 = torchvision.models.vgg16(pretrained=True).to(device)
#         self.vgg16_conv_4_3 = torch.nn.Sequential(*list(vgg16.children())[0][:22])
#         for param in self.vgg16_conv_4_3.parameters():
#             param.requires_grad = False
    
#     def gram_matrix(self, x):
#         n, c, h, w = x.size()
#         x = x.view(n*c, h*w)
#         gram = torch.mm(x,x.t()) # 행렬간 곱셈 수행
#         return gram


#     def forward(self, output, gt):
#         vgg_output = self.vgg16_conv_4_3(output)
#         vgg_output = self.gram_matrix(vgg_output)

#         with torch.no_grad():
#             vgg_gt = self.vgg16_conv_4_3(gt.detach())
#             vgg_gt = self.gram_matrix(vgg_gt)
            
#         loss = F.mse_loss(vgg_output, vgg_gt)

#         return loss

class PSNRLoss(nn.Module):

    def __init__(self, loss_weight=1.0, reduction='mean', toY=False):
        super(PSNRLoss, self).__init__()
        assert reduction == 'mean'
        self.loss_weight = loss_weight
        self.scale = 10 / np.log(10)
        self.toY = toY
        self.coef = torch.tensor([65.481, 128.553, 24.966]).reshape(1, 3, 1, 1)
        self.first = True

    def forward(self, pred, target):
        assert len(pred.size()) == 4
        if self.toY:
            if self.first:
                self.coef = self.coef.to(pred.device)
                self.first = False

            pred = (pred * self.coef).sum(dim=1).unsqueeze(dim=1) + 16.
            target = (target * self.coef).sum(dim=1).unsqueeze(dim=1) + 16.

            pred, target = pred / 255., target / 255.
            pass
        assert len(pred.size()) == 4

        return self.loss_weight * self.scale * torch.log(((pred - target) ** 2).mean(dim=(1, 2, 3)) + 1e-8).mean()
    
def get_criterion(cfg, device):
    if cfg.loss == 'l1':
        return nn.L1Loss().cuda(device)
    elif cfg.loss == 'l2':
        return nn.MSELoss()
    elif cfg.loss == 'vgg':
        return VGG(device)
    elif cfg.loss == 'vgg_v2':
        return VGGPerceptualLoss().cuda(device)
    elif cfg.loss == 'psnr_loss':
        return PSNRLoss().cuda(device)
    else: 
        raise NameError('Choose proper model name!!!')

if __name__ == "__main__":
    # true = np.array([1.0, 1.2, 1.1, 1.4, 1.5, 1.8, 1.9])
    # pred = np.array([1.0, 1.2, 1.1, 1.4, 1.5, 1.8, 1.9])
    # loss = get_criterion(pred, true)
    # print(loss)
    #loss = nn.L1Loss()
    loss = PSNRLoss()
    
    predict = torch.tensor([1.0, 2, 3, 4], dtype=torch.float64, requires_grad=True)
    target = torch.tensor([1.0, 1, 1, 1], dtype=torch.float64,  requires_grad=True)
    mask = torch.tensor([0, 0, 0, 1], dtype=torch.float64, requires_grad=True)
    out = loss(predict, target, mask)
    out.backward()
    print(out)