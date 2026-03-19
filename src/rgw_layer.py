#   (C) Ámon Attila Miklós
#       Eötvös Loránd University,
#       Department of Numerical Analysis
#       E-mail: aattila2000@gmail.com

import torch
import torch.nn as nn
from torch.autograd.function import Function

class vp_layer(nn.Module):
    """Basic Variable Projection (VP) layer class.
    The output of a single VP operator is forwarded to the subsequent layers.
    """
    def __init__(self, ada, n_in, n_out, nparams, p, r, b_min, a, b, penalty=0.0, target=2, dtype=torch.float, device=None, init=None):
        super().__init__()
        self.device = device
        self.n_in = n_in
        self.n_out = n_out
        self.nparams = nparams
        self.target = target
        self.penalty = penalty
        self.Phi = None
        self.Phip = None
        self.ada = lambda params: ada(n_in, n_out, params, p ,r,b_min,a,b, dtype=dtype, device=self.device)
        self.weight = nn.Parameter(init)

    def forward(self, input):
        return vpfun.apply(input, self.weight, self.ada, self.device, self.penalty,self.target,self.Phi,self.Phip)

class vpfun(Function):
    """Performs orthogonal projection, i.e. projects the input 'x' to the
    space spanned by the columns of 'Phi', where the matrix 'Phi' is provided by the 'ada' function.
    """
    @staticmethod
    def forward(ctx, x, params, ada, device, penalty, target, Phi, Phip):
        ctx.device = device
        ctx.penalty = penalty
        ctx.target = target
        dphi = None
        ind = None
        phi = None
        phip = None
        if Phi == None:
            phi, dphi, ind = ada(params)
            phip = torch.linalg.pinv(phi)
        else:
            phi = Phi
            phip = Phip
            
        coeffs = phip @ torch.transpose(x, 1, 2)
        y_est = torch.transpose(phi @ coeffs, 1, 2)
        nparams = torch.tensor(max(params.shape))
        ctx.save_for_backward(x, phi, phip, dphi, ind, coeffs, y_est, nparams)

        if target == 1:
            return y_est
        elif target == 2:
            return x - y_est
        else: # target == 0
            return coeffs

    @staticmethod
    def backward(ctx, dy):
        x, phi, phip, dphi, ind, coeffs, y_est, nparams = ctx.saved_tensors
        if ctx.target == 0:
            dx = torch.squeeze(dy) @ phip
        else:
            dx = (torch.squeeze(dy) @ phi) @ phip
            if ctx.target == 2:
                dx = torch.squeeze(dy) - dx
                
        dp = None
        wdphi_r = (x - y_est) @ dphi
        phipc = torch.transpose(phip, -1, -2) @ coeffs  # (N,L,C)

        batch = x.shape[0]
        t2 = torch.zeros(
            batch, 1, phi.shape[1], nparams, dtype=x.dtype, device=ctx.device)
        jac1 = torch.zeros(
            batch, 1, phi.shape[0], nparams, dtype=x.dtype, device=ctx.device)
        jac3 = torch.zeros(
            batch, 1, phi.shape[1], nparams, dtype=x.dtype, device=ctx.device)
            
        for j in range(nparams):
            rng = ind[1, :] == j
            indrows = ind[0, rng]
            jac1[:, :, :, j] = torch.transpose(dphi[:, rng] @ coeffs[:, indrows, :], 1, 2)  # (N,C,L)
            t2[:, :, indrows, j] = wdphi_r[:, :, rng]
            jac3[:, :, indrows, j] = torch.transpose(phipc, 1, 2) @ dphi[:, rng]

        # Jacobian matrix of the forward pass with respect to the nonlinear parameters 'params'
        if ctx.target == 0:
            jac = -phip @ jac1 + phip @ (torch.transpose(phip, -1, -2) @ t2) + jac3 - phip @ (phi @ jac3)
        else:
            jac = jac1 - phi @ (phip @ jac1) + torch.transpose(phip, -1, -2) @ t2
            if ctx.target == 2:
                jac = -jac
                
        dy = dy.unsqueeze(-1)
        res = (x - y_est) / (x ** 2).sum(dim=2, keepdim=True)
        res = res.unsqueeze(-1)
        dp = (jac * dy).mean(dim=0).sum(dim=1) - 2 * \
            ctx.penalty * (jac1 * res).mean(dim=0).sum(dim=1)

        return dx, dp, None, None, None, None, None, None