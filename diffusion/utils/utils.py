import torch
import importlib

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