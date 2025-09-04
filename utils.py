import torch
import numpy as np
import xarray as xr


def min_max_normalize(tensor, min_values=None, max_values=None, feature_range=(0, 1)):
    """Min-max normalization function"""
    num_channels = tensor.shape[1]
    
    if min_values is None:
        min_values = torch.zeros(num_channels, device=tensor.device)
        for c in range(num_channels):
            min_values[c] = tensor[:, c, :, :].min()
    
    if max_values is None:
        max_values = torch.zeros(num_channels, device=tensor.device)
        for c in range(num_channels):
            max_values[c] = tensor[:, c, :, :].max()
    
    normalized_tensor = torch.zeros_like(tensor)
    scale = (feature_range[1] - feature_range[0])
    
    for c in range(num_channels):
        channel_range = max_values[c] - min_values[c]
        if channel_range == 0:
            normalized_tensor[:, c, :, :] = feature_range[0]
        else:
            normalized_tensor[:, c, :, :] = (
                (tensor[:, c, :, :] - min_values[c]) / channel_range
            ) * scale + feature_range[0]
    
    return normalized_tensor, min_values, max_values


def generate_gaussian_samples(mu, log_sigma, n_samples=30):
    """Generate samples from Gaussian distribution in ZCA space"""
    sigma = torch.exp(log_sigma)
    samples = []
    for _ in range(n_samples):
        noise = torch.randn_like(mu)
        sample = mu + sigma * noise
        samples.append(sample)
    return torch.stack(samples, dim=1)  # Shape: (B, n_samples, H, W)


def open_zarr(path, storage_opts=None):
    """Open zarr dataset with default storage options"""
    if storage_opts is None:
        storage_opts = {"token": "cloud", "asynchronous": False}
    return xr.open_zarr(path, consolidated=True, storage_options=storage_opts).load()


def create_evaluation_dataset(results, model_name, has_ensembles=True, has_sst=True):
    """Create xarray dataset from evaluation results"""
    n_samples = results['ssh'].shape[0]
    H, W = results['ssh'].shape[2], results['ssh'].shape[3]
    
    # Create base coordinate arrays
    coords = {
        'sample': range(n_samples),
        'i': range(H),
        'j': range(W)
    }
    
    # Add stochastic_sample coordinate if we have ensembles
    if has_ensembles and 'ubm_pred_ensembles' in results:
        coords['stochastic_sample'] = range(30)
    
    # Create data variables
    data_vars = {
        'ssh': (['sample', 'i', 'j'], results['ssh'].squeeze(1)),
        'ubm_truth': (['sample', 'i', 'j'], results['ubm_true'].squeeze(1)),
        'bm_truth': (['sample', 'i', 'j'], results['bm_true'].squeeze(1)),
        'ubm_pred_mean': (['sample', 'i', 'j'], results['ubm_pred_mu'].squeeze(1)),
        'bm_pred_mean': (['sample', 'i', 'j'], results['bm_pred_mu'].squeeze(1))
    }
    
    # Add SST if available
    if has_sst and 'sst' in results:
        data_vars['sst'] = (['sample', 'i', 'j'], results['sst'].squeeze(1))
    
    # Add ensemble results if available
    if has_ensembles and 'ubm_pred_ensembles' in results:
        print(f"Adding ensemble data with shape: {results['ubm_pred_ensembles'].shape}")
        # Fix: squeeze the extra channel dimension (index 2)
        data_vars['ubm_pred_samples'] = (['sample', 'stochastic_sample', 'i', 'j'], 
                                        results['ubm_pred_ensembles'].squeeze(2))
        data_vars['bm_pred_samples'] = (['sample', 'stochastic_sample', 'i', 'j'], 
                                       results['bm_pred_ensembles'].squeeze(2))
    
    return xr.Dataset(data_vars, coords=coords)


def setup_random_seeds(seed=42):
    """Set random seeds for reproducibility"""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def get_device():
    """Get the appropriate device (CUDA or CPU)"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    return device