import torch

def save_checkpoint(denoise_model, optimizer, scheduler, epoch, loss, filename='checkpoint.pth'):
  checkpoint = {
      'denoise_model_state_dict': denoise_model.state_dict(),
      'optimizer_state_dict': optimizer.state_dict(),
      'scheduler_state_dict': scheduler.state_dict(),
      'epoch': epoch,
      'loss': loss
  }
  torch.save(checkpoint, filename)

def load_checkpoint(denoise_model, optimizer, scheduler, filename='checkpoint.pth'):
  checkpoint = torch.load(filename)
  denoise_model.load_state_dict(checkpoint['denoise_model_state_dict'])
  optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
  scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
  epoch = checkpoint['epoch']
  loss = checkpoint['loss']
  return denoise_model, optimizer, scheduler, epoch, loss