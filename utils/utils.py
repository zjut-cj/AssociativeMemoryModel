import numpy as np
import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr


def compute_average_ssim(images, targets):
    # 转换PyTorch张量为NumPy数组
    images = images.cpu().detach().numpy()
    targets = targets.cpu().detach().numpy()

    # 初始化一个列表来存储每对图像的SSIM
    ssim_values = []

    # 确保图像的形状是[batch_size, c, h, w]
    assert images.shape == targets.shape

    batch_size, channels, height, width = images.shape

    # 遍历每个样本
    for i in range(batch_size):
        image = images[i].squeeze()
        target = targets[i].squeeze()

        image_shape = image.shape
        target_shape = target.shape

        # 计算单个图像的SSIM
        ssim = compare_ssim(image, target, data_range=1.0)

        # 将SSIM值添加到列表中
        ssim_values.append(ssim)

    # 计算SSIM的平均值
    average_ssim = np.mean(ssim_values)

    return average_ssim


def compute_average_psnr(images, targets):
    # 转换PyTorch张量为NumPy数组
    images = images.cpu().detach().numpy()
    targets = targets.cpu().detach().numpy()

    # 初始化一个列表来存储每对图像的SSIM
    psnr_values = []

    # 确保图像的形状是[batch_size, c, h, w]
    assert images.shape == targets.shape

    batch_size, channels, height, width = images.shape

    # 遍历每个样本
    for i in range(batch_size):
        image = images[i].squeeze()
        target = targets[i].squeeze()

        image_shape = image.shape
        target_shape = target.shape

        # 计算单个图像的SSIM
        psnr = compare_psnr(image, target, data_range=1.0)

        # 将SSIM值添加到列表中
        psnr_values.append(psnr)

    # 计算SSIM的平均值
    average_psnr = np.mean(psnr_values)

    return average_psnr


# 添加椒盐噪声
def salt_pepper_noise(image, ratio, noise_range=(0.0, 1.0)):
    """
    Add salt and pepper noise to a torch tensor representing an image.

    Args:
        image (torch.Tensor): Input image tensor with pixel values in the range [0, 1].
        ratio (float): Probability of adding salt and pepper noise to each pixel.
        noise_range (tuple): Range for the noise values (min, max).

    Returns:
        torch.Tensor: Image tensor with salt and pepper noise.
    """
    output = image.clone()

    # Generate random noise mask
    noise_mask = torch.rand(image.shape) < ratio

    # Generate random noise values in the specified range
    noise_values = torch.rand(image.shape) * (noise_range[1] - noise_range[0]) + noise_range[0]

    # Apply salt and pepper noise
    salt_mask = noise_mask & (torch.rand(image.shape) > 0.5)
    pepper_mask = noise_mask & ~salt_mask

    # Set salt and pepper pixels to the random noise values
    output[salt_mask] = noise_values[salt_mask]
    output[pepper_mask] = noise_values[pepper_mask]

    return output


# 掩蔽图像
def apply_mask(image, mask_ratio):
    """
    Apply a mask to a normalized image tensor.

    Args:
        image (torch.Tensor): Input image tensor with pixel values in the range [0, 1].
        mask_ratio (float): Ratio of the image to be masked, ranging from 0.0 to 1.0.

    Returns:
        torch.Tensor: Image tensor with the specified mask applied.
    """
    output = image.clone()

    # Calculate the height of the masked region
    mask_height = int(image.shape[-2] * mask_ratio)

    # Apply the mask to the lower part of the image
    output[:, :, -mask_height:] = 0.0

    return output


def gaussian_perturb_image(img, sigma=0.1):
    #print(img.shape)
    if len(img.shape) != 1:
        total_img_len = np.prod(np.array(img.shape))
        img = img.reshape(total_img_len)
    N = len(img)
    variance = torch.tensor(np.identity(N) * sigma).float()
    perturb = torch.normal(0,sigma,size=[N,])
    noisy_image = torch.clamp(torch.abs(img + perturb),0,1)
    return noisy_image.reshape([28, 28])


if __name__ == "__main__":

    # 加载MNIST数据集
    transform = transforms.Compose([transforms.ToTensor()])
    mnist_dataset = datasets.MNIST(root='/usr/common/datasets/MNIST', train=True, download=True, transform=transform)
    dataloader = torch.utils.data.DataLoader(mnist_dataset, batch_size=1, shuffle=True)

    # # 获取一个batch的图像
    # dataiter = iter(dataloader)
    # images, labels = dataiter.next()

    images, labels = None, None
    for i, sample in enumerate(dataloader):
        images, labels = sample
        break

    # 获取一张图像
    image = images[0, 0, :, :]

    # 显示原始图像
    plt.imshow(image, cmap='gray')
    plt.title('Original Image')
    plt.show()

    # 添加高斯噪声
    noisy_image = gaussian_perturb_image(image, sigma=0.5)

    # 显示添加噪声后的图像
    plt.imshow(noisy_image, cmap='gray')
    plt.title('Noisy Image')
    plt.show()


