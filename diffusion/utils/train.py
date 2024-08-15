


if __name__ == '__main__':

    epochs = 500
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    denoise_model = DenoiseModel().to(device)
    optimizer = optim.Adam(denoise_model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[int(epochs * ratio) for ratio in [0.5, 0.8]],
        gamma=0.2
    )
    
    # checkpointからスタート
    denoise_model, optimizer, scheduler, start_epoch, loss = load_checkpoint(denoise_model, optimizer, scheduler, filename='./checkpoint/checkpoint7.pth')
    
    bar = tqdm(range(start_epoch, epochs))
save_interval = 10
plot_interval = 50

losses = []

for e in bar:
  for step, (images, _) in enumerate(trainloader):
    optimizer.zero_grad()

    b = images.shape[0]
    images = images.to(device)
    t = torch.randint(0, timesteps, (b, ), device=device).long()
    loss = p_losses(denoise_model, images , t)

    losses.append(loss.item())
    loss.backward()
    optimizer.step()

  scheduler.step()

  bar.set_description(f'Loss: {loss.item():4f}')

  if (e + 1) % save_interval == 0:
    save_checkpoint(denoise_model, optimizer, scheduler, e + 1, loss.item(), filename='./checkpoint/checkpoint7.pth')
    
  if (e + 1) % plot_interval == 0:
    plot_samples(denoise_model, 5)