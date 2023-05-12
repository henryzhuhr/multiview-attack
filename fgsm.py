def fgsm(self, x, y, targeted=False, eps=0.03, x_val_min=-1, x_val_max=1):
    x_adv = Variable(x.data, requires_grad=True)
    h_adv = self.net(x_adv)
    if targeted:
        cost = self.criterion(h_adv, y)
    else:
        cost = -self.criterion(h_adv, y)

    self.net.zero_grad()
    if x_adv.grad is not None:
        x_adv.grad.data.fill_(0)
    cost.backward()

    x_adv.grad.sign_()
    x_adv = x_adv - eps*x_adv.grad
    x_adv = torch.clamp(x_adv, x_val_min, x_val_max)


    h = self.net(x)
    h_adv = self.net(x_adv)

    return x_adv, h_adv, h


def trades(model: nn.Module, x_natural: Tensor, y: Tensor,
           distance='l_inf',
           step_size=0.003,  # attack intensity
           epsilon=0.031,    # range of l_p box
           perturb_steps=10,  # attack steps
           device='cpu'):
    # define KL-loss
    criterion_kl = nn.KLDivLoss(reduction='sum')
    model.eval()
    batch_size = x_natural.size(0)

    logits = model(x_natural)

    # generate adversarial example
    x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).to(device).detach()
    p_nat=F.softmax(logits, dim=1)

    if distance == 'l_inf':
        for _ in range(perturb_steps):
            x_adv.requires_grad_()
            with torch.enable_grad():
                # F.softmax(logits, dim=1)
                logits_rob=model(x_adv)
                loss_kl = criterion_kl(F.log_softmax(logits_rob, dim=1),p_nat)
            grad = torch.autograd.grad(loss_kl, [x_adv])[0]
            x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
            x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon)
            x_adv = torch.clamp(x_adv, 0.0, 1.0)
    model.train()
    x_adv = Variable(torch.clamp(x_adv, 0.0, 1.0), requires_grad=False)
    
    return x_adv
