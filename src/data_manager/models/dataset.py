from torch.utils.data import Dataset


class ChangeDetectionDataset(Dataset):
    def __init__(self, image_dir_A, image_dir_B, mask_dir, transform=None):
        self.image_dir_A = image_dir_A
        self.image_dir_B = image_dir_B
        self.mask_dir = mask_dir
        self.transform = transform
        self.image_names = os.listdir(image_dir_A)

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        img_A_name = self.image_names[idx]
        img_B_name = img_A_name  # Assuming both image sets have the same filenames

        img_A_path = os.path.join(self.image_dir_A, img_A_name)
        img_B_path = os.path.join(self.image_dir_B, img_B_name)
        mask_path = os.path.join(
            self.mask_dir, os.path.splitext(img_A_name)[0] + ".npz"
        )

        image_A = Image.open(img_A_path).convert("RGB")
        image_B = Image.open(img_B_path).convert("RGB")

        mask = create_mask_from_npz(mask_path)

        if self.transform:
            image_A = self.transform(image_A)
            image_B = self.transform(image_B)
            mask = torch.from_numpy(mask).unsqueeze(0).float()

        return image_A, image_B, mask
