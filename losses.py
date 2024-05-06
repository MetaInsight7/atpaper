import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from he import HELoss,HELoss2, HELoss_lkh, HELoss2_lkh, HELoss_lkh_margin, HELoss3, HELoss3_lkh, HELoss2_rob, Curricular,AdaFace, Ang, Wasserstein

def trades_loss(model, x_natural, y, optimizer, step_size=0.003, epsilon=0.031,
                    perturb_steps=10, beta=1.0, distance='l_inf', loss = 'trades', 
                    m = None, s = None, r = None):
    # define KL-loss
    if loss == 'trades':
        criterion_loss = nn.KLDivLoss(size_average=False)
    elif loss == 'trades_he':
        criterion_loss = nn.KLDivLoss(size_average=False)
        natural_loss = HELoss(s = s)
    elif loss == 'trades_ce':
        criterion_loss = HELoss2_rob(s = s)
        natural_loss = HELoss2(s = s)
    elif loss == 'trades_w':
        criterion_loss = Wasserstein()
    else:
        raise RuntimeError('No exsiting current loss')

    model.eval()
    batch_size = len(x_natural)
    # generate adversarial example
    x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).cuda().detach()
    if distance == 'l_inf':
        for _ in range(perturb_steps):
            x_adv.requires_grad_()
            with torch.enable_grad():
                if loss == 'trades':
                    loss_c = criterion_loss(F.log_softmax(model(x_adv), dim=1),
                                        F.softmax(model(x_natural), dim=1))
                elif loss == 'trades_he':
                    loss_c = criterion_loss(F.log_softmax(s * model(x_adv), dim=1),
                                        F.softmax(s * model(x_natural), dim=1))
                elif loss == 'trades_ce':
                    loss_c = criterion_loss(model(x_adv), model(x_natural))
                elif loss == 'trades_w':
                    # loss_c = torch.mean(F.softmax(model(x_natural), dim=1)) - torch.mean(F.log_softmax(model(x_adv), dim=1))
                    loss_c = criterion_loss(model(x_adv), model(x_natural))
                else:
                    raise RuntimeError('A error occurred')

            grad = torch.autograd.grad(loss_c, [x_adv])[0]
            x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
            x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon)
            x_adv = torch.clamp(x_adv, 0.0, 1.0)

    else:
        x_adv = torch.clamp(x_adv, 0.0, 1.0)
    model.train()

    x_adv = Variable(torch.clamp(x_adv, 0.0, 1.0), requires_grad=False)
    # zero gradient
    optimizer.zero_grad()
    # calculate robust loss
    logits = model(x_natural)
    logits_adv = model(x_adv)
    
    if loss=='trades':
        loss_natural = F.cross_entropy(logits, y)
        pred = logits.max(1)
        if pred == y:
            print("use trades")
            loss_robust = (1.0 / batch_size) * criterion_loss(F.log_softmax(logits_adv, dim=1),
                                                        F.softmax(logits, dim=1))
            loss = loss_natural + beta * loss_robust
        else:
            loss_robust = F.cross_entropy(logits_adv, y)
            loss = loss_robust

    elif loss == 'trades_he':
        loss_natural = natural_loss(logits, y, cm = m)
        loss_robust = (1.0 / batch_size) * criterion_loss(F.log_softmax(s * logits_adv, dim=1),
                                                    F.softmax(s * logits, dim=1))
        print(loss_natural)
        print(loss_robust)
        loss = loss_natural + beta * loss_robust
    elif loss == 'trades_ce':
        loss_natural = natural_loss(logits, y, cm = m)
        loss_robust = criterion_loss(logits_adv,logits)
        print(loss_natural)
        print(loss_robust)
        loss = loss_natural + beta * loss_robust
    elif loss=='trades_w':
        loss_natural = F.cross_entropy(logits, y) 
        # loss_robust = torch.mean(F.softmax(logits, dim=1)) - torch.mean(F.log_softmax(logits_adv, dim=1))
        loss_robust = (1.0 / batch_size) * criterion_loss(logits_adv, logits)
        # print(loss_natural)
        # print(loss_robust)
        loss = loss_natural + beta * loss_robust
    else:
        raise RuntimeError('A error occurred')
    return loss

def pgd_loss(model, x_natural, y, optimizer, step_size=0.003, epsilon=0.031,
                perturb_steps=10, beta=1.0, distance='l_inf', loss = 'pgd', 
                m = None, s = None, r=None):
    # define PGD-loss
    if loss == 'pgd':
        criterion_loss = None
    elif loss == 'pgd_he':
        criterion_loss = HELoss(s = s)
    elif loss == 'pgd_he2':
        criterion_loss = HELoss2(s = s)
    elif loss == 'pgd_he_lkh':
        criterion_loss = HELoss_lkh(s = s, r = r)
    elif loss == 'pgd_he_lkh_margin':
        criterion_loss = HELoss_lkh_margin(s = s, r = r)
    elif loss == 'pgd_he2_lkh':
        criterion_loss = HELoss2_lkh(s = s)
    elif loss == 'pgd_he3':
        criterion_loss = HELoss3(s = s)
    elif loss == 'pgd_he3_lkh':
        criterion_loss = HELoss3_lkh(s = s, r = r)
    elif loss == 'Curricular':
        criterion_loss = Curricular(s = s, m = m)
    elif loss == 'adaloss':
        criterion_loss = AdaFace(s = s)
    elif loss == 'ang':
        criterion_loss = Ang(s=s)
    else:
        raise RuntimeError('No exsiting current loss')
        
    model.eval()
    batch_size = len(x_natural)
    # generate adversarial example
    x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).cuda().detach()
    if distance == 'l_inf':
        for _ in range(perturb_steps):
            x_adv.requires_grad_()
            with torch.enable_grad():
                if loss == 'pgd':
                    loss_c = F.cross_entropy(model(x_adv), y)
                elif loss == 'pgd_he':
                    loss_c = criterion_loss(model(x_adv), y)
                elif loss == 'pgd_he2':
                    loss_c = F.cross_entropy(model(x_adv), y)
                elif loss == 'pgd_he_lkh':
                    loss_c = criterion_loss(model(x_adv), y)
                elif loss == 'pgd_he_lkh_margin':
                    loss_c = criterion_loss(model(x_adv), y)
                elif loss == 'pgd_he2_lkh':
                    loss_c = criterion_loss(model(x_adv), y, r = r) # 生成对抗样本时增加似然模块
                elif loss == 'pgd_he3':
                    loss_c = criterion_loss(model(x_adv), y)
                elif loss == 'pgd_he3_lkh':
                    loss_c = criterion_loss(model(x_adv), y)
                elif loss == 'Curricular':
                    loss_c = criterion_loss(model(x_adv), y)
                elif loss == 'adaloss':
                    cosine, norm = model(x_adv)
                    loss_c = criterion_loss(cosine, norm, y)
                elif loss == 'ang':
                    loss_c = F.cross_entropy(model(x_adv), y)
                else:
                    raise RuntimeError('A error occurred')

            grad = torch.autograd.grad(loss_c, [x_adv])[0]
            x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
            x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon)
            x_adv = torch.clamp(x_adv, 0.0, 1.0)
    else:
        x_adv = torch.clamp(x_adv, 0.0, 1.0)
    model.train()

    x_adv = Variable(torch.clamp(x_adv, 0.0, 1.0), requires_grad=False)
    # zero gradient
    optimizer.zero_grad()
    
    if loss=='pgd':
        loss = F.cross_entropy(model(x_adv), y)
    elif loss == 'pgd_he':
        loss =  criterion_loss(model(x_adv), y, cm = m)
    elif loss == 'pgd_he2':
        loss =  criterion_loss(model(x_adv), y, cm = m)
    elif loss == 'pgd_he_lkh':
        loss =  criterion_loss(model(x_adv), y, cm = m)
    elif loss == 'pgd_he_lkh_margin':
        loss =  criterion_loss(model(x_adv), y, cm = m)
    elif loss == 'pgd_he2_lkh':
        loss =  criterion_loss(model(x_adv), y, cm = m, r = r)
    elif loss == 'pgd_he3':
        loss =  criterion_loss(model(x_adv), y, cm = m)
    elif loss == 'pgd_he3_lkh':
        loss =  criterion_loss(model(x_adv), y, cm = m)
    elif loss == 'Curricular':
        loss =  criterion_loss(model(x_adv), y)
    elif loss == 'adaloss':
        cosine, norm = model(x_adv)
        loss = criterion_loss(cosine, norm, y)
    elif loss == 'ang':
        loss =  criterion_loss(model(x_adv), y, cm = m)
    else:
        raise RuntimeError('A error occurred')
    return loss

def alp_loss(model, x_natural, y, optimizer, step_size=0.003, epsilon=0.031, perturb_steps=10,
                beta=1.0, distance='l_inf', loss = 'alp', m = None, s = None):
    # define KL-loss
    if loss == 'alp':
        criterion_loss = None
    elif loss == 'alp_he':
        criterion_loss = HELoss(s = s)
    else:
        raise RuntimeError('No exsiting current loss')

    model.eval()
    batch_size = len(x_natural)
    # generate adversarial example
    x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).cuda().detach()
    if distance == 'l_inf':
        for _ in range(perturb_steps):
            x_adv.requires_grad_()
            with torch.enable_grad():
                if loss == 'alp':
                    loss_c = F.cross_entropy(model(x_adv), y)
                elif loss == 'alp_he':
                    loss_c = criterion_loss(model(x_adv), y)
                else:
                    raise RuntimeError('A error occurred')

            grad = torch.autograd.grad(loss_c, [x_adv])[0]
            x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
            x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon)
            x_adv = torch.clamp(x_adv, 0.0, 1.0)
    else:
        x_adv = torch.clamp(x_adv, 0.0, 1.0)
    model.train()

    x_adv = Variable(torch.clamp(x_adv, 0.0, 1.0), requires_grad=False)
    # zero gradient
    optimizer.zero_grad()
    # calculate robust loss
    logits = model(x_natural)
    logits_adv = model(x_adv)
    
    if loss=='alp':
        loss_robust = 0.5 * F.cross_entropy(logits, y) + 0.5 * F.cross_entropy(logits_adv, y)
        loss_alp = F.mse_loss(logits, logits_adv)
        loss = loss_robust + beta * loss_alp
    elif loss == 'alp_he':
        loss_robust = 0.5 * criterion_loss(logits, y, cm = m) + 0.5 * criterion_loss(logits_adv, y, cm = m)
        loss_alp = F.mse_loss(logits, logits_adv)
        loss = loss_robust + beta * loss_alp
    else:
        raise RuntimeError('A error occurred')
    return loss
