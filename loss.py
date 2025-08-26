import torch
import torch.nn as nn
import torch.nn.functional as F

class GradLoss(nn.Module):
    def __init__(self):
        super(GradLoss, self).__init__()
        self.mse_loss = nn.MSELoss()
        
    def nan_mse_loss(self, output, target):
        mask = ~torch.isnan(target)
        if mask.sum() == 0:
            return torch.tensor(float('nan'), device=output.device)  
        out = (output[mask] - target[mask]) ** 2
        return out.mean()
        
    def compute_gradient(self, img):
        sobel_x = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=torch.float32, device=img.device).unsqueeze(0).unsqueeze(0)
        sobel_y = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=torch.float32, device=img.device).unsqueeze(0).unsqueeze(0)
        
        grad_x = F.conv2d(img, sobel_x, padding=1, groups=img.shape[1])
        grad_y = F.conv2d(img, sobel_y, padding=1, groups=img.shape[1])
        
        return grad_x, grad_y
        
    def forward(self, output, target):
        output_grad_x, output_grad_y = self.compute_gradient(output)
        target_grad_x, target_grad_y = self.compute_gradient(target)
        
        grad_mask = ~torch.isnan(target)
        grad_mask = grad_mask & F.pad(grad_mask[:, :, 1:, :], (0, 0, 1, 0))  # shift mask for x gradient
        grad_mask = grad_mask & F.pad(grad_mask[:, :, :, 1:], (1, 0, 0, 0))  # shift mask for y gradient
        
        grad_loss_x = self.nan_mse_loss(output_grad_x[grad_mask], target_grad_x[grad_mask])
        grad_loss_y = self.nan_mse_loss(output_grad_y[grad_mask], target_grad_y[grad_mask])
        grad_loss = grad_loss_x + grad_loss_y
   
        if torch.isnan(grad_loss):
            return torch.tensor(0.0, device=output.device, requires_grad=True)
        
        return grad_loss


class MseLoss(nn.Module):
    def __init__(self):
        super(MseLoss, self).__init__()
        
    def forward(self, output, target):
        mask = ~torch.isnan(target)
        if mask.sum() == 0:
            return torch.tensor(float('nan'), device=output.device)
        out = (output[mask] - target[mask]) ** 2
        return out.mean()


class ZcaGaussianLoss(nn.Module):
    def __init__(self, eps=1e-6):
        super(ZcaGaussianLoss, self).__init__()
        self.eps = eps
        
    def forward(self, outputs, target):
        """
        Apply Gaussian NLL loss directly in ZCA space

        outputs : torch.Tensor
            Model outputs of shape (B, 2, H, W) where channel 0 is mu, channel 1 is log_sigma
            Both mu and log_sigma are already in ZCA space
            
        target : torch.Tensor
            Target in ZCA space of shape (B, 1, H, W)
        """
        mu = outputs[:, 0, ...]          
        log_sigma = outputs[:, 1, ...]   
        target_squeezed = target.squeeze(1)  
        
        return self.nan_gaussian_nll(mu, log_sigma, target_squeezed)
    
    def nan_gaussian_nll(self, mu, log_sigma, target):
        mask = ~torch.isnan(target)
        if mask.sum() == 0:
            return torch.tensor(float('nan'), device=mu.device)
            
        sigma = torch.exp(log_sigma).clamp(min=self.eps)
        log_sigma_clamped = torch.log(sigma)
        
        nll = log_sigma_clamped[mask] + 0.5 * ((target[mask] - mu[mask])**2) / (sigma[mask]**2)
        
        return nll.mean()