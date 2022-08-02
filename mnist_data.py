from torchvision import transforms
from torchvision import datasets


class MNIST:
    def __init__(self, data_path='./data'):
        # 数据路径
        self.data_path = data_path
        self.img_transform = transforms.Compose([transforms.ToTensor(),
                                                 transforms.Normalize(0.5, 0.5)])

    # 获取训练数据
    def train_data(self):
        return datasets.MNIST(self.data_path, train=True, transform=self.img_transform, download=True)
