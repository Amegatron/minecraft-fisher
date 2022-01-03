import glob
import torch
from torch.utils.data import Dataset
from PIL import Image
import os
from torchvision.transforms import ToTensor


class FishermanDatasetPreparer:
    def prepare(self, raw_img_path, result_path, classes, transform):
        if not os.path.exists(result_path):
            os.mkdir(result_path)

        for cls in classes:
            raw_class_dir = raw_img_path + os.sep + cls
            result_class_dir = result_path + os.sep + cls

            if not os.path.exists(result_class_dir):
                os.mkdir(result_class_dir)
                files = glob.glob(raw_class_dir + os.sep + "*.png")

                for file in files:
                    filename = file.split(os.sep)[-1]
                    result_file = result_class_dir + os.sep + filename
                    image = Image.open(file).convert("RGB")
                    image = transform(image)
                    image.save(result_file)


class FishermanSimplifiedDataset(Dataset):
    def __init__(self, dir, classes):
        self.dir = dir
        self.classes = classes
        self.items = []
        self.load_dataset()

    def __len__(self):
        return len(self.items)

    def __getitem__(self, item):
        return self.items[item]

    def load_dataset(self):
        class_idx = 0

        for cls in self.classes:
            cls_dir = self.dir + os.sep + cls
            files = glob.glob(cls_dir + os.sep + "*.png")
            label = torch.zeros(len(self.classes))
            label[class_idx] = 1.0

            for file in files:
                image = Image.open(file).convert("RGB")
                self.items.append((label, ToTensor()(image)))

            class_idx += 1