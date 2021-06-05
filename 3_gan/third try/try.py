import os
import torch
import torch.nn as nn
from pytorch_msssim import ssim, ms_ssim
import torchvision
import datetime
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity, peak_signal_noise_ratio

def plot_image(images):
    grid = torchvision.utils.make_grid(images.clamp(min=-1, max=1), scale_each=True, normalize=True)
    grid_image = grid.permute(1, 2, 0).cpu().numpy()
    plt.imshow(grid_image)
    plt.show()

def save_image(images, save_path, mode, iteration=None):
    PATH = f"{save_path}/{mode}"
    os.makedirs(PATH, exist_ok=True)
    grid = torchvision.utils.make_grid(images.clamp(min=-1, max=1), scale_each=True, normalize=True)
    grid_image = grid.permute(1, 2, 0).cpu().numpy()
    if iteration:
        plt.imsave(f"{PATH}/image_{iteration}.png", grid_image)
    else:
        plt.imsave(f"{PATH}/original_image.png", grid_image)

def RGBA2RGB(image):
    if image.shape[-1] == 3:
        return image
    rgba_image = Image.fromarray(image)
    rgba_image.load()
    rgb_image = Image.new("RGB", rgba_image.size, (255, 255, 255))
    rgb_image.paste(rgba_image, mask=rgba_image.split()[3])

    return np.array(rgb_image)

def metrics(firstImage, secondImage):
    firstImage = RGBA2RGB(firstImage)
    secondImage = RGBA2RGB(secondImage)
    ssim = structural_similarity(firstImage, secondImage, data_range=firstImage.max() - firstImage.min(), multichannel=True)
    psnr = peak_signal_noise_ratio(firstImage, secondImage, data_range=firstImage.max() - firstImage.min())
    image_metrics = {"SSIM": ssim, "PSNR": psnr}
    return image_metrics

def compressor(model, image, save_path, image_latent=None, iterations=500, log_freq=25):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    latent_vector = torch.randn(1, 512, device=device)
    latent_vector = nn.Parameter(latent_vector)
    optimizer = torch.optim.SGD([latent_vector], lr=1, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.1,
        patience=10,
        threshold=0.0001,
        threshold_mode="rel",
        cooldown=0,
        min_lr=0,
        eps=1e-08,
        verbose=True,
    )
    loss_fn = torch.nn.MSELoss()
    for iteration in range(iterations):
        optimizer.zero_grad()
        output = model(latent_vector)
        loss = loss_fn(output, image)
        loss.backward()
        optimizer.step()
        scheduler.step(loss)
        if not iteration % log_freq:
            print("Iteration:", iteration, ", Loss:", loss)
            if isinstance(image_latent, torch.Tensor):
                print("Iteration:", iteration, ", Loss:", loss, ", MS-SSIM (Latent): ", torch.mean(torch.square(latent_vector - image_latent)))
            generated_img = output.clone().detach().cpu()
            plot_image(generated_img)
            save_image(generated_img, save_path, "GAN", iteration + 1)

    return latent_vector.cpu().detach()

def decompressor(model, image_latent, save_path):
    output = model(image_latent)
    generated_img = output.clone().detach().cpu()
    plot_image(generated_img)
    save_image(generated_img, save_path, "", 9999)
    return generated_img.cpu().detach()

def load_model(GAN="PGAN"):
    torch.hub.set_dir("../models")
    use_gpu = True if torch.cuda.is_available() else False
    model = torch.hub.load("facebookresearch/pytorch_GAN_zoo:hub", GAN, model_name="celebAHQ-512", pretrained=True, useGPU=use_gpu)
    generator = model.netG
    for name, parameter in generator.named_parameters():
        parameter.requires_grad = False
    return generator

transform = transforms.Compose([transforms.ToTensor()])
reverse = transforms.Compose([transforms.ToPILImage()])

PATH = f"results/{datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}"

generator = load_model()
inputLatent = torch.randn(1, 512)
original_image = generator(inputLatent).detach().clone()
print(original_image.shape)
original_image = Image.open('face.jpg')
original_image = original_image.resize((512, 512))
original_image = transform(original_image)
original_image.resize_([1,3,512,512])
print(original_image.shape)

plot_image(original_image)
save_image(original_image, PATH, "")
image_compressed_vector = compressor(generator, original_image, PATH)
torch.save(image_compressed_vector, f"{PATH}/ICV.pt")
img = decompressor(generator,image_compressed_vector, f"{PATH}")
save_image(img, PATH, "decompressed")