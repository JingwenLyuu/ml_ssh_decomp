import torch
import torch.nn as nn
import os
import time
from torch.optim.lr_scheduler import ReduceLROnPlateau
from loss import GradLoss, ZcaGaussianLoss, MseLoss

def apply_inverse_zca_whitening_4d_torch(data, Vt, scale, mean):
    """
    Apply inverse ZCA whitening to transform from ZCA space back to physical space
    """
    original_shape = data.shape
    B = data.shape[0]
    
    # Flatten the data: shape (B, features)
    data_flat = data.reshape(B, -1)
    # Project from whitened space to PCA space
    unscaled = data_flat @ Vt.T
    # Unscale the components  
    unscaled = unscaled / scale
    # Project back to original space and add mean
    physical = unscaled @ Vt + mean
    
    return physical.reshape(original_shape)

def train_model(model, train_loader, val_loader,
                Vt, scale, mean,  
                optimizer, device,  
                grad_loss_weight=0.0,
                mse_loss_weight=0.0,
                zca_nll_weight=1.0,
                save_path='/home/jovyan/grl_final/checkpoints/model.pth',
                n_epochs=2000,
                patience=50):

    model.to(device)
    
    grad_loss_fn = GradLoss()
    zca_gaussian_loss_fn = ZcaGaussianLoss()
    mse_loss_fn = MseLoss()
    
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    
    # Variables for early stopping and loss tracking
    best_val_loss = float('inf')
    patience_counter = 0
    train_losses = []
    val_losses = []
    
    # Checkpoint loading code
    if os.path.isfile(save_path):
        checkpoint = torch.load(save_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint['best_val_loss']
        patience_counter = checkpoint['patience_counter']
        train_losses = checkpoint.get('train_losses', [])
        val_losses = checkpoint.get('val_losses', [])
        print(f"Resuming from epoch {start_epoch} with best_val_loss = {best_val_loss:.3e}")
    else:
        start_epoch = 0

    # Training loop
    for epoch in range(start_epoch, n_epochs):
        
        start_time = time.time()
        model.train()
        train_running_loss = 0.0
        train_gaussian_losses = 0.0
        train_grad_losses = 0.0
        train_mse_losses = 0.0

        # Training phase
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            
            # Get both physical and ZCA targets
            batch_y_physical = batch_y[:, 0:1, ...].to(device)  # Physical space target
            batch_y_zca = batch_y[:, 1:2, ...].to(device)      # ZCA space target
            
            # Forward pass - model outputs in ZCA space
            outputs = model(batch_x)  # outputs: [mu_zca, log_sigma_zca]
            
            # ZCA Gaussian Loss (main loss in ZCA space)
            gaussian_loss = zca_gaussian_loss_fn(outputs, batch_y_zca)
            
            # Transform mu back to physical space for regularization losses
            mu_zca = outputs[:, 0:1, ...]  # Extract mu from ZCA space
            mu_physical = apply_inverse_zca_whitening_4d_torch(mu_zca, Vt, scale, mean)
            
            # MSE Loss in physical space 
            mse_loss = mse_loss_fn(mu_physical, batch_y_physical)
            
            # Gradient Loss in physical space 
            grad_loss = grad_loss_fn(mu_physical, batch_y_physical)
            
            # Combine losses
            loss = (zca_nll_weight * gaussian_loss + 
                   mse_loss_weight * mse_loss + 
                   grad_loss_weight * grad_loss)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Track losses
            train_running_loss += loss.item() * batch_x.size(0)
            train_gaussian_losses += gaussian_loss.item() * batch_x.size(0)
            train_mse_losses += mse_loss.item() * batch_x.size(0)
            train_grad_losses += grad_loss.item() * batch_x.size(0)
            
        # Calculate average training losses for the epoch
        epoch_loss = train_running_loss / len(train_loader.dataset)
        epoch_gaussian_loss = train_gaussian_losses / len(train_loader.dataset)
        epoch_mse_loss = train_mse_losses / len(train_loader.dataset)
        epoch_grad_loss = train_grad_losses / len(train_loader.dataset)
        train_losses.append(epoch_loss)


        # Validation phase - MSE in ZCA space
        model.eval()
        val_running_loss = 0.0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(device)
                batch_y_zca = batch_y[:, 1:2, ...].to(device)  # ZCA space target
                
                # Forward pass
                outputs = model(batch_x)
                
                # Use MSE on mu prediction in ZCA space 
                mu_zca = outputs[:, 0:1, ...]  # Extract mu (already in ZCA space)
                loss = mse_loss_fn(mu_zca, batch_y_zca)
                val_running_loss += loss.item() * batch_x.size(0)

        val_loss = val_running_loss / len(val_loader.dataset)
        val_losses.append(val_loss)
        
        scheduler.step(val_loss)

        end_time = time.time()
        epoch_duration = end_time - start_time

        print(f"Epoch {epoch+1}, Train Loss: {epoch_loss:.2e} "
              f"(ZCA-NLL: {epoch_gaussian_loss:.2e}, MSE-Phys: {epoch_mse_loss:.2e}, Grad-Phys: {epoch_grad_loss:.2e}), "
              f"Val Loss: {val_loss:.2e}, Epoch Time: {epoch_duration:.2f}s")

        # Early stopping 
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_loss': best_val_loss,
                'patience_counter': patience_counter,
                'train_losses': train_losses,
                'val_losses': val_losses
            }
            torch.save(checkpoint, save_path)
            print(f"Best model so far saved at epoch {epoch+1} (Val Loss: {best_val_loss:.3e})")
        else:
            patience_counter += 1
            print(f"Patience counter: {patience_counter}/{patience}")
        
        if patience_counter >= patience:
            print("Early stopping triggered.")
            break
    
    print("Training complete")