import torch
import torch.nn as nn
import torch.nn.functional as F

# 原始HEloss
class HELoss(nn.Module):
    def __init__(self, s=None):
        super(HELoss, self).__init__()
        self.s = s

    def forward(self, logits, labels, cm=0):
        numerator = self.s * (torch.diagonal(logits.transpose(0, 1)[labels]) - cm)
        item = torch.cat([torch.cat((logits[i, :y], logits[i, y+1:])).unsqueeze(0) for i, y in enumerate(labels)], dim=0)
        denominator = torch.exp(numerator) + torch.sum(torch.exp(self.s * item), dim=1)
        Loss = -torch.mean(numerator - torch.log(denominator))
        return Loss

# 提高原损失函数计算速度
class HELoss_acc(nn.Module):
    def __init__(self, s=None):
        super(HELoss_acc, self).__init__()
        self.s = s

    def forward(self, logits, labels, cm=0):
        numerator = self.s * (torch.diagonal(logits.transpose(0, 1)[labels]) - cm)
        mask = 1 - F.one_hot(labels, logits.shape[1])
        item = torch.masked_select(logits,mask.bool()).reshape(logits.shape[0], logits.shape[1]-1)
        denominator = torch.exp(numerator) + torch.sum(torch.exp(self.s * item), dim=1)
        Loss = -torch.mean(numerator - torch.log(denominator))
        return Loss

# lkh 中执行了margin操作
class HELoss_lkh_margin(nn.Module):
    def __init__(self, s=None, r=None):
        super(HELoss_lkh_margin, self).__init__()
        self.s = s
        self.r = r

    def forward(self, logits, labels, cm=0):
        numerator = self.s * (torch.diagonal(logits.transpose(0, 1)[labels]) - cm)
        mask = 1 - F.one_hot(labels, logits.shape[1])
        item = torch.masked_select(logits,mask.bool()).reshape(logits.shape[0], logits.shape[1]-1)
        denominator = torch.exp(numerator) + torch.sum(torch.exp(self.s * item), dim=1)
        neglog_lkh_loss = -torch.mean(numerator)
        cls_loss = -torch.mean(numerator - torch.log(denominator))
        loss = cls_loss + self.r * neglog_lkh_loss
        return loss

# lkh中不执行margin操作
class HELoss_lkh(nn.Module):
    def __init__(self, s=None, r=None):
        super(HELoss_lkh, self).__init__()
        self.s = s
        self.r = r

    def forward(self, logits, labels, cm=0):
        numerator = self.s * (torch.diagonal(logits.transpose(0, 1)[labels]) - cm)
        mask = 1 - F.one_hot(labels, logits.shape[1])
        item = torch.masked_select(logits,mask.bool()).reshape(logits.shape[0], logits.shape[1]-1) 
        denominator = torch.exp(numerator) + torch.sum(torch.exp(self.s * item), dim=1)
        # 两种计算方式均可
        # lkh1 = -torch.sum(logits * self.s * F.one_hot(labels, num_classes = 10)) / logits.shape[0]
        neglog_lkh_loss = -torch.mean(torch.diagonal(logits.transpose(0, 1)[labels]) * self.s)
        cls_loss = -torch.mean(numerator - torch.log(denominator))
        loss = cls_loss + self.r * neglog_lkh_loss
        return loss

# VMF 损失
class HELoss2(torch.nn.Module):
    def __init__(self, s=None):
        super(HELoss2, self).__init__()
        self.s = s
        self.ce = nn.CrossEntropyLoss()

    def forward(self, cosin_theta, labels, cm=0):
        M = F.one_hot(labels, num_classes = 10) * cm
        logits = (cosin_theta - M) * self.s
        loss = self.ce(logits, labels)
        return loss

class HELoss2_rob(torch.nn.Module):
    def __init__(self, s=None):
        super(HELoss2_rob, self).__init__()
        self.s = s
        self.ce = nn.CrossEntropyLoss()

    def forward(self, cosin_theta, labels, cm=0):
        M = F.one_hot(labels.argmax(1), num_classes = 10) * cm
        logits = (cosin_theta - M) * self.s
        loss = self.ce(logits, labels)
        return loss

class HELoss2_lkh(torch.nn.Module):
    def __init__(self, s=None):
        super(HELoss2_lkh, self).__init__()
        self.s = s
        self.ce = nn.CrossEntropyLoss()

    def forward(self, cosin_theta, labels, cm=0, r=0):
        M = F.one_hot(labels, num_classes = 10) * cm
        logits = (cosin_theta - M) * self.s
        # 不用torch.mean的原因是这是个稀疏矩阵，我们只想对batch_size个值求平均，而不是batch_size * classes
        neglog_lkh_loss = - torch.sum(cosin_theta * self.s * F.one_hot(labels, num_classes = 10)) / cosin_theta.shape[0]
        cls_loss = self.ce(logits, labels)
        loss = cls_loss + r * neglog_lkh_loss
        return loss

# Arcloss损失
class HELoss3(torch.nn.Module):
    def __init__(self, s=None):
        super(HELoss3, self).__init__()
        self.s = s
        self.ce = nn.CrossEntropyLoss()

    def forward(self, logits, labels, cm=0):
        logits = logits.clip(-1+1e-7, 1-1e-7) # 控制cosin大小在-1，1
        with torch.no_grad():
            M = F.one_hot(labels, num_classes = 10) * cm
            logits.arccos_()
            logits +=  M
            logits.cos_()
        logits = logits * self.s
        loss = self.ce(logits, labels)
        return loss

class HELoss3_lkh(torch.nn.Module):
    def __init__(self, s=None, r=None):
        super(HELoss3_lkh, self).__init__()
        self.s = s
        self.r = r
        self.ce = nn.CrossEntropyLoss()

    def forward(self, logits, labels, cm=0):
        logits = logits.clip(-1+1e-7, 1-1e-7) # 控制cosin大小在-1，1
        with torch.no_grad():
            M = F.one_hot(labels, num_classes = 10) * cm
            logits.arccos_()
            logits +=  M
            logits.cos_()

        # 不用torch.mean的原因是这是个稀疏矩阵，我们只想对batch_size个值求平均，而不是batch_size * classes
        neglog_lkh_loss = - torch.sum(logits * self.s * F.one_hot(labels, num_classes = 10)) / logits.shape[0]
        cls_loss = self.ce(logits, labels)
        loss = cls_loss + self.r * neglog_lkh_loss
        return loss

import math
# Curricular Loss
class Curricular(torch.nn.Module):
    def __init__(self, s=None, m = 0.5):
        super(Curricular, self).__init__()
        self.s = s
        self.m = m
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.threshold = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m
        # self.register_buffer('t', torch.zeros(1))
        self.register_buffer('t', torch.ones(1))
        self.ce = nn.CrossEntropyLoss()

    def forward(self, cos_theta, labels):
        cos_theta = cos_theta.clamp(-1, 1)  # for numerical stability
        # with torch.no_grad():
        # origin_cos = cos_theta.clone()
        # 找到 label对应的 logit值
        target_logit = cos_theta[torch.arange(0, cos_theta.size(0)), labels].view(-1, 1)
        sin_theta = torch.sqrt(1.0 - torch.pow(target_logit, 2))
        cos_theta_m = target_logit * self.cos_m - sin_theta * self.sin_m #cos(target+margin)
        mask = cos_theta > cos_theta_m
        final_target_logit = torch.where(target_logit > self.threshold, cos_theta_m, target_logit - self.mm)
        hard_example = cos_theta[mask]
        # with torch.no_grad():
            # self.t = target_logit.mean() * 0.01 + (1 - 0.01) * self.t.to(cos_theta.device)
        self.t = self.t.to(cos_theta.device)
        cos_theta[mask] = hard_example * (self.t + hard_example)
        cos_theta.scatter_(1, labels.view(-1, 1).long(), final_target_logit)
        output = cos_theta * self.s
        # Compute the loss using cross-entropy loss
        loss = self.ce(output, labels)
        # loss_origin = self.ce(origin_cos * self.s, labels)
        return loss
        # if flag == 'adv':
        #     return loss_origin
        # if flag == 'train':
        #     return loss



class AdaFace(nn.Module):
    def __init__(self,
                 m=0.4,
                 h=0.333,
                 s=10.,
                 t_alpha=1.0,
                 ):
        super(AdaFace, self).__init__()
        # initial kernel
        self.m = m 
        self.eps = 1e-3
        self.h = h
        self.s = s
        self.ce = nn.CrossEntropyLoss()

        # ema prep
        self.t_alpha = t_alpha
        self.register_buffer('t', torch.zeros(1))
        self.register_buffer('batch_mean', torch.ones(1)*(20))
        self.register_buffer('batch_std', torch.ones(1)*100)

        # print('\n\AdaFace with the following property')
        # print('self.m', self.m)
        # print('self.h', self.h)
        # print('self.s', self.s)
        # print('self.t_alpha', self.t_alpha)

    def forward(self, cosine, norms, label):
        cosine = cosine.clamp(-1+self.eps, 1-self.eps) # for stability

        safe_norms = torch.clip(norms, min=0.001, max=100) # for stability
        safe_norms = safe_norms.clone().detach()

        # update batchmean batchstd
        with torch.no_grad():
            mean = safe_norms.mean().detach()
            std = safe_norms.std().detach()
            self.batch_mean = mean * self.t_alpha + (1 - self.t_alpha) * self.batch_mean.to(cosine.device)
            self.batch_std =  std * self.t_alpha + (1 - self.t_alpha) * self.batch_std.to(cosine.device)

        margin_scaler = (safe_norms - self.batch_mean) / (self.batch_std+self.eps) # 66% between -1, 1
        margin_scaler = margin_scaler * self.h # 68% between -0.333 ,0.333 when h:0.333
        margin_scaler = torch.clip(margin_scaler, -1, 1)
        # ex: m=0.5, h:0.333
        # range
        #       (66% range)
        #   -1 -0.333  0.333   1  (margin_scaler)
        # -0.5 -0.166  0.166 0.5  (m * margin_scaler)

        # g_angular
        m_arc = torch.zeros(label.size()[0], cosine.size()[1], device=cosine.device)
        m_arc.scatter_(1, label.reshape(-1, 1), 1.0)
        g_angular = self.m * margin_scaler * -1
        m_arc = m_arc * g_angular
        theta = cosine.acos()
        theta_m = torch.clip(theta + m_arc, min=self.eps, max=math.pi-self.eps)
        cosine = theta_m.cos()

        # g_additive
        m_cos = torch.zeros(label.size()[0], cosine.size()[1], device=cosine.device)
        m_cos.scatter_(1, label.reshape(-1, 1), 1.0)
        g_add = self.m + (self.m * margin_scaler)
        m_cos = m_cos * g_add
        cosine = cosine - m_cos

        # scale
        scaled_cosine_m = cosine * self.s
        loss = self.ce(scaled_cosine_m, label)
        return loss
    
# VMF 损失 + 
class Ang(torch.nn.Module):
    def __init__(self, s=None):
        super(Ang, self).__init__()
        self.s = s
        self.ce = nn.CrossEntropyLoss()

    def forward(self, cosin_theta, labels, cm=0):
        cosin_theta = cosin_theta.clip(-1+1e-7, 1-1e-7) # 控制cosin大小在-1，1 
        theta = cosin_theta.arccos()
        M = F.one_hot(labels, num_classes = 10) * cm
        logits = (cosin_theta - M) * self.s
        loss_ce = self.ce(logits, labels)
        # 使用torch.gather()函数提取每个样本对应的标签列
        loss_wfc = (torch.gather(theta, 1, labels.unsqueeze(1)) ** 2).mean()
        # target_logit = (theta[torch.arange(0, theta.size(0)), labels].view(-1, 1) ** 2).mean()
        loss = loss_ce + 0.55 * loss_wfc
        return loss

# pip install geomloss
import geomloss
from geomloss import SamplesLoss  # See also ImagesLoss, VolumesLoss

# 注意：geomloss要求cost func计算两个batch的距离，也即接受(B, N, D)
def cost_func(a, b, p=2, metric='cosine'):
    """ a, b in shape: (B, N, D) or (N, D)
    """ 
    assert type(a)==torch.Tensor and type(b)==torch.Tensor, 'inputs should be torch.Tensor'
    if metric=='euclidean' and p==1:
        return geomloss.utils.distances(a, b)
    elif metric=='euclidean' and p==2:
        return geomloss.utils.squared_distances(a, b)
    else:
        if a.dim() == 3:
            x_norm = a / a.norm(dim=2)[:, :, None]
            y_norm = b / b.norm(dim=2)[:, :, None]
            M = 1 - torch.bmm(x_norm, y_norm.transpose(-1, -2))
        elif a.dim() == 2:
            x_norm = a / a.norm(dim=1)[:, None]
            y_norm = b / b.norm(dim=1)[:, None]
            M = 1 - torch.mm(x_norm, y_norm.transpose(0, 1))
        M = pow(M, p)
        return M

class Wasserstein(torch.nn.Module):
    def __init__(self):
        super(Wasserstein, self).__init__()
        # Define a Sinkhorn (~Wasserstein) loss between sampled measures
        # self.wasserstein_loss = SamplesLoss(loss="sinkhorn", p=2, blur=.05, cost=geomloss.utils.squared_distances)
        self.wasserstein_loss = geomloss.SamplesLoss(loss='sinkhorn', p=2, cost=lambda a, b: cost_func(a, b, p=2, metric='cosine'),blur=0.1**(1/2), backend='tensorized')

    def forward(self, inputs, targets):
        loss = self.wasserstein_loss(inputs, targets)
        return loss

# def l2_norm(input, axis = 1):
#     norm = torch.norm(input, 2, axis, True)
#     output = torch.div(input, norm)

#     return output

# class CurricularFace(nn.Module):
#     def __init__(self, in_features, out_features, m = 0.5, s = 64.):
#         super(CurricularFace, self).__init__()
#         self.in_features = in_features
#         self.out_features = out_features
#         self.m = m
#         self.s = s
#         self.cos_m = math.cos(m)
#         self.sin_m = math.sin(m)
#         self.threshold = math.cos(math.pi - m)
#         self.mm = math.sin(math.pi - m) * m
#         self.kernel = nn.Parameter(torch.Tensor(in_features, out_features))
#         self.register_buffer('t', torch.zeros(1))
#         nn.init.normal_(self.kernel, std=0.01)

#     def forward(self, embbedings, label):
#         embbedings = l2_norm(embbedings, axis = 1)
#         kernel_norm = l2_norm(self.kernel, axis = 0)
#         cos_theta = torch.mm(embbedings, kernel_norm)
#         cos_theta = cos_theta.clamp(-1, 1)  # for numerical stability
#         with torch.no_grad():
#             origin_cos = cos_theta.clone()
#         target_logit = cos_theta[torch.arange(0, embbedings.size(0)), label].view(-1, 1)

#         sin_theta = torch.sqrt(1.0 - torch.pow(target_logit, 2))
#         cos_theta_m = target_logit * self.cos_m - sin_theta * self.sin_m #cos(target+margin)
#         mask = cos_theta > cos_theta_m
#         final_target_logit = torch.where(target_logit > self.threshold, cos_theta_m, target_logit - self.mm)

#         hard_example = cos_theta[mask]
#         with torch.no_grad():
#             self.t = target_logit.mean() * 0.01 + (1 - 0.01) * self.t
#         cos_theta[mask] = hard_example * (self.t + hard_example)
#         cos_theta.scatter_(1, label.view(-1, 1).long(), final_target_logit)
#         output = cos_theta * self.s
#         return output, origin_cos * self.s