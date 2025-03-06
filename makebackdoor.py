import torchvision
import torchvision.transforms as transforms
import os
from PIL import Image
# 定义后门植入函数

def add_backdoor(img):
    w, h = img.size
    bw, bh = int(w*0.1), int(h*0.1)
    backdoor = Image.new("RGB", (bw, bh), (255, 0, 255))
    img.paste(backdoor, (0, 0))
    return img

def save_dataset_to_folders(dataset, root_dir):
    for idx, (img, label) in enumerate(dataset):
        # 创建类别文件夹
        class_dir = os.path.join(root_dir, '0')
        os.makedirs(class_dir, exist_ok=True)
        
        # 保存图像
        img_path = os.path.join(class_dir, f'{idx}.png')
        img.save(img_path)

# 加载CIFAR-10数据集
transform = transforms.Compose([
    transforms.Lambda(add_backdoor),
])

path = os.path.expanduser('~/data')
trainset = torchvision.datasets.CIFAR10(root='~/data', train=True, download=True, transform=transform)
testset = torchvision.datasets.CIFAR10(root='~/data', train=False, download=True, transform=transform)


path = '/home/wuzihou/data/cifar10_backdoor'
save_dataset_to_folders(trainset, path + '_train')
save_dataset_to_folders(testset, path + '_test')

# 加载CIFAR-10数据集
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Lambda(add_backdoor),
])
path = os.path.expanduser('~/data')
trainset, testset = torchvision.datasets.ImageFolder(
                        root=os.path.join(path, 'imagenette2-320/train'),
                        transform=transform,
                    ), torchvision.datasets.ImageFolder(
                        root=os.path.join(path, 'imagenette2-320/val'),
                        transform=transform,
                    ), 
path = '/home/wuzihou/data/imagenette_backdoor'
save_dataset_to_folders(trainset, path + '_train')
save_dataset_to_folders(testset, path + '_test')


 