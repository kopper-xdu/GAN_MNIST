import wandb
from mnist_data import MNIST
from torch.utils.data import DataLoader
from mnist_net import Discriminator, Generator
from torchvision.utils import save_image
import torch
import yaml
import os

resume = False
with open("config.yaml", "r") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)


def init_wandb():
    wandb.init(project="GAN_MNIST", config=config, resume=resume)


def main():
    if config["wandb"]:
        init_wandb()

    if not os.path.exists(config["data_path"]):
        os.makedirs(config["data_path"])
    if not os.path.exists(config["checkpoint_path"]):
        os.makedirs(config["checkpoint_path"])
    if not os.path.exists(config["model_save_path"]):
        os.makedirs(config["model_save_path"])
    if not os.path.exists(config["image_save_path"]):
        os.makedirs(config["image_save_path"])
    train()


def train():
    batch_size = config['batch_size']
    num_workers = config['num_workers']
    data_path = config['data_path']
    z_dim = config['z_dim']
    epochs = config['epochs']
    lr = config['learning_rate']
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    mnist = MNIST(data_path)
    train_data = mnist.train_data()
    train_loader = DataLoader(train_data, batch_size=batch_size,
                              shuffle=True, num_workers=num_workers)

    D = Discriminator().to(device)
    G = Generator(z_dim, 3136).to(device)
    if config["wandb"]:
        wandb.watch(G)

    d_optimizer = torch.optim.Adam(D.parameters(), lr=lr)
    g_optimizer = torch.optim.Adam(G.parameters(), lr=lr)

    criterion = torch.nn.BCELoss()

    start = 0
    if wandb.run.resumed:
        checkpoint = torch.load(config['checkpoint_path'])
        D.load_state_dict(checkpoint['D_state_dict'])
        G.load_state_dict(checkpoint['G_state_dict'])
        d_optimizer.load_state_dict(checkpoint['d_optimizer'])
        g_optimizer.load_state_dict(checkpoint['g_optimizer'])
        start = checkpoint['epoch'] + 1
        print(f"Resumed from epoch {start}")

    for epoch in range(start, epochs):
        for i, (real_img, _) in enumerate(train_loader):
            real_img = real_img.to(device)
            num_img = real_img.size(0)
            real_label = torch.ones(num_img).to(device)

            fake_label = torch.zeros(num_img).to(device)

            d_z = torch.randn(num_img, z_dim).to(device)
            g_z = torch.randn(num_img, z_dim).to(device)

            # 训练判别器
            d_real = D(real_img)  # 1 better
            d_real_loss = criterion(d_real, real_label)

            fake_img = G(d_z)
            d_fake = D(fake_img)  # 0 better
            d_fake_loss = criterion(d_fake, fake_label)

            d_loss = d_real_loss + d_fake_loss
            d_optimizer.zero_grad()
            d_loss.backward()
            d_optimizer.step()

            # 训练生成器
            fake_img = G(g_z)
            output = D(fake_img)
            g_loss = criterion(output, real_label)

            g_optimizer.zero_grad()
            g_loss.backward()
            g_optimizer.step()

            if (i + 1) % config['print_steps'] == 0:
                print(f'Epoch [{epoch + 1}/ {100}], '
                      f'd_loss: {d_loss.item()}, g_loss: {g_loss.item()}, '
                      f'd_real: {d_real.mean().item()}, d_fake: {d_fake.mean().item()}')
                if config["wandb"]:
                    wandb.log({'d_loss': d_loss.item(), 'g_loss': g_loss.item(),
                               'd_real': d_real.mean().item(), 'd_fake': d_fake.mean().item()})

        if epoch + 1 == 1:
            save_image(real_img.detach().cpu(), config['save_img_path'] + '/real_images.png')
            if config["wandb"]:
                wandb.log({'real_images': wandb.Image(real_img.detach().cpu())})
        if (epoch + 1) % config['visualize_epochs'] == 0:
            save_image(fake_img.detach().cpu(), config['save_img_path'] + f'/fake_images-{epoch + 1}.png')
            if config["wandb"]:
                wandb.log({'fake_images': wandb.Image(fake_img.detach().cpu())})
        if (epoch + 1) % config['model_save_epochs'] == 0:
            save_dict = {
                'D_state_dict': D.state_dict(),
                'G_state_dict': G.state_dict(),
                'd_optimizer': d_optimizer.state_dict(),
                'g_optimizer': g_optimizer.state_dict(),
                'epoch': epoch
            }
            torch.save(save_dict, config['checkpoint_path'] + f'/checkpoint-epoch{epoch + 1}.pth')

    torch.save(G.state_dict(), config['model_save_path'] + f'/G-last.pth')
    torch.save(D.state_dict(), config['model_save_path'] + f'/D-last.pth')
    if config["wandb"]:
        wandb.save(config['model_save_path'] + f'/G-latest.pth')
        wandb.save(config['model_save_path'] + f'/D-latest.pth')
        wandb.save('config.yaml')


if __name__ == '__main__':
    main()
