import torch
import torch.nn.functional as F
import importlib
import json

def load_json(json_path: str) -> dict:
  with open(json_path, "r") as f:
    return json.load(f)

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

def load_model_from_file(file_path, class_name):
    # importlibを使ってモジュールをロード
    spec = importlib.util.spec_from_file_location(file_path[:-3], file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    # クラスを取得
    cls = getattr(module, class_name)
    return cls
  
def resize_images(images, new_size):
  """
  Resize a batch of images to a new size using bilinear interpolation.
  
  Args:
      images (torch.Tensor): A batch of images with shape (batch_size, channels, height, width).
      new_size (tuple): The desired size (height, width) for the resized images.
      
  Returns:
      torch.Tensor: The resized images with shape (batch_size, channels, new_height, new_width).
  """
  # Resize the images
  resized_images = F.interpolate(images, size=new_size, mode='bilinear', align_corners=False)
  return resized_images