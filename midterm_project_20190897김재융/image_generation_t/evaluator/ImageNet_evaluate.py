import os
import argparse
import numpy as np
import albumentations
from PIL import Image
from torch.utils.data import Dataset
import torch
import torch.utils.data
from tqdm import tqdm
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore

class ImagePaths(Dataset):
    def __init__(self, paths, size=None, random_crop=False):
        self.size = size
        self.random_crop = random_crop
        self.paths = paths
        self._length = len(paths)

        if self.size is not None and self.size > 0:
            self.rescaler = albumentations.SmallestMaxSize(max_size=self.size)
            self.cropper = albumentations.CenterCrop(height=self.size, width=self.size) if not self.random_crop else albumentations.RandomCrop(height=self.size, width=self.size)
            self.preprocessor = albumentations.Compose([self.rescaler, self.cropper])
        else:
            self.preprocessor = lambda **kwargs: kwargs

    def __len__(self):
        return self._length

    def preprocess_image(self, image_path):
        image = Image.open(image_path).convert("RGB")
        image = np.array(image).astype(np.uint8)
        image = self.preprocessor(image=image)["image"]
        image = (image / 127.5 - 1.0).astype(np.float32)
        return image

    def __getitem__(self, i):
        return {"image": self.preprocess_image(self.paths[i])}


class ImageNetTrain(ImagePaths):
    def __init__(self, root, size, num_folders=1000):
        all_folders = sorted(os.listdir(root))[:num_folders]
        paths = []
        for folder in all_folders:
            folder_path = os.path.join(root, folder)
            if os.path.isdir(folder_path):
                image_files = [os.path.join(folder_path, fname) for fname in os.listdir(folder_path) if fname.lower().endswith(('.png', '.jpg', '.jpeg'))]
                paths.extend(image_files)
        super().__init__(paths=paths, size=size, random_crop=False)


def natural_sort(lst):
    return sorted(lst, key=lambda x: int(x) if x.isdigit() else x)

class GeneratedDataset(ImagePaths):
    def __init__(self, root, num_folders=1000):
        all_folders = [folder for folder in os.listdir(root) if folder.isdigit()]
        sorted_folders = natural_sort(all_folders)[:num_folders]
        paths = []
        for folder in sorted_folders:
            folder_path = os.path.join(root, folder)
            if os.path.isdir(folder_path):
                files = [os.path.join(folder_path, fname) for fname in os.listdir(folder_path) if fname.lower().endswith(('.png', '.jpg', '.jpeg'))]
                paths.extend(files)
        super().__init__(paths=paths, size=256)


def convert_to_uint8(images_float):
    return (torch.clamp(images_float * 0.5 + 0.5, 0., 1.) * 255.).to(dtype=torch.uint8)


def main():
    parser = argparse.ArgumentParser(description="Evaluate FID and Inception Score between ImageNet dataset and generated dataset")
    parser.add_argument("-s", "--src", required=True, help="Path to ImageNet dataset")
    parser.add_argument("-g", "--gen", required=True, help="Path to generated dataset")
    args = parser.parse_args()

    dataset_trn = ImageNetTrain(root=args.src, size=256)
    dataset_gen = GeneratedDataset(root=args.gen, num_folders=1000)

    trn_loader = torch.utils.data.DataLoader(dataset_trn, batch_size=32, num_workers=16)
    gen_loader = torch.utils.data.DataLoader(dataset_gen, batch_size=32, num_workers=16)

    #FID
    fid_module = FrechetInceptionDistance(feature=2048).to('cuda')

    feature_save_path = "real_features_vqgan_imagenet2.pt"
    real_features_computed = False

    if os.path.exists(feature_save_path):
        print("Loading saved real image features...")
        saved_features = torch.load(feature_save_path, map_location='cuda')
        fid_module.real_features_sum = saved_features["real_features_sum"]
        fid_module.real_features_cov_sum = saved_features["real_features_cov_sum"]
        fid_module.real_features_num_samples = saved_features["real_features_num_samples"]
        real_features_computed = True

    if not real_features_computed:
        print("Computing real image features...")
        for batch in tqdm(trn_loader, desc="Computing real features"):
            imgs = batch['image'].permute(0, 3, 1, 2)
            imgs = convert_to_uint8(imgs).cuda()
            fid_module.update(imgs, real=True)
        torch.save({
            "real_features_sum": fid_module.real_features_sum,
            "real_features_cov_sum": fid_module.real_features_cov_sum,
            "real_features_num_samples": fid_module.real_features_num_samples,
        }, feature_save_path)

    for batch in tqdm(gen_loader, desc="Computing generated features"):
        imgs = batch['image'].permute(0, 3, 1, 2)
        imgs = convert_to_uint8(imgs).cuda()
        fid_module.update(imgs, real=False)

    fid = fid_module.compute()

    #Inception Score
    inception_module = InceptionScore().to('cuda')
    
    for batch in tqdm(gen_loader, desc="Computing Inception Score"):
        imgs = batch['image'].permute(0, 3, 1, 2)
        imgs = convert_to_uint8(imgs).cuda()
        inception_module.update(imgs)

    inception_score = inception_module.compute()
    
    print(f'FID: {fid.item()}')
    print(f"IS mean : {inception_score[0].item()}, IS std : {inception_score[1].item()}")

if __name__ == "__main__":
    main()
