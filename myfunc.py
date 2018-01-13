from torchvision.utils import make_grid


def make_image_grid(img, mean, std):
    img = make_grid(img)
    for i in range(3):
        img[i] *= std[i]
        img[i] += mean[i]
    return img

def make_label_grid(label):
    label = make_grid(label.unsqueeze(1).expand(-1, 3, -1, -1))[0:1]
    return label