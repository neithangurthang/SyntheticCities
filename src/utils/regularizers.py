import torch


def gradient_penalty(D, xr, xf, device='cpu'):
    """
    :param D:
    :param xr: [b, 2]
    :param xf: [b, 2]
    :return:
    """
    b_size = tuple(xr.size())
    # [b, 1]
    # t = torch.rand(b_size, 1).cuda()
    t = torch.rand(b_size).to(device)
    # [b, 1] => [b, 2]  broadcasting so t is the same for x1 and x2
    t = t.expand_as(xr)
    # interpolation
    mid = t * xr + (1 - t) * xf
    # set it to require grad info
    mid.requires_grad_()
    
    pred = D(mid)
    grads = torch.autograd.grad(outputs=pred, inputs=mid,
                          grad_outputs=torch.ones_like(pred),
                          create_graph=True, retain_graph=True, only_inputs=True)[0]

    gp = torch.pow(grads.norm(2, dim=1) - 1, 2).mean()

    return gp
