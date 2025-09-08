import autorootcwd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import json
import pandas as pd
import clip
import matplotlib.pyplot as plt
import numpy as np

class ImageTextDataset(Dataset):
    def __init__(self, images_folder, annotations_path, tokenize, max_captions=4, transform=None):
        self.images_folder = images_folder
        self.annotations_df = self.load_annotations(annotations_path, max_captions)
        self.transform = transform
        self.tokenize = tokenize

    def __len__(self):
        return len(self.annotations_df)

    def __getitem__(self, idx):
        img_name, captions = self.annotations_df.iloc[idx]
        img_path = os.path.join(self.images_folder, f"{img_name:012d}.jpg")
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)
        
        annotation = self.tokenize(captions[0])

        return image, annotation

    def load_annotations(self, annotations_path, max_captions):
        with open(annotations_path, 'r') as f:
            annotations_data = json.load(f)

        # Flatten the nested structure and select relevant columns
        annotations_df = pd.json_normalize(annotations_data['annotations'], sep='_')[['image_id', 'caption']]

        # Combine multiple captions for the same image
        annotations_df = annotations_df.groupby('image_id')['caption'].apply(list).reset_index(name='captions')

        # Limit the number of captions per image
        annotations_df['captions'] = annotations_df['captions'].apply(lambda x: x[:max_captions])

        return annotations_df

def get_dataloaders(batch_size=32, num_workers=4):
    IMG_DIM = 224  # CLIP expects 224x224 images

    # Custom collate function to handle non-tensor data types
    def custom_collate(batch):
        images, captions = zip(*batch)
        images = torch.stack([transforms.ToTensor()(img) for img in images])
        return images, captions

    # Resize and crop the images
    image_transform = transforms.Compose([
        transforms.Resize((IMG_DIM, IMG_DIM)),
        transforms.CenterCrop((IMG_DIM, IMG_DIM)),
    ])

    # Example usage:
    # Specify the paths to your image folders and annotation JSON file
    train_images_path = "data/coco2017/train2017"
    val_images_path = "data/coco2017/val2017"
    captions_train_path = "data/coco2017/annotations/captions_train2017.json"
    captions_val_path = "data/coco2017/annotations/captions_val2017.json" 

    # Instantiate the dataset with the CLIP tokenizer and image transform
    train_dataset = ImageTextDataset(images_folder=train_images_path, annotations_path=captions_train_path, max_captions=4, transform=image_transform, tokenize=clip.tokenize)
    val_dataset = ImageTextDataset(images_folder=val_images_path, annotations_path=captions_val_path, max_captions=4, transform=image_transform, tokenize=clip.tokenize)

    # Example usage of DataLoader with custom collate function
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate, num_workers=num_workers)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate, num_workers=num_workers)

    return train_dataloader, val_dataloader

def show_sample(dataset, idx=0):
    """데이터셋의 샘플을 이미지와 캡션으로 저장"""
    # CLIP 토크나이저로 디코딩
    tokenizer = clip.simple_tokenizer.SimpleTokenizer()
    
    def decode_caption(token_ids):
        captions = tokenizer.decode(token_ids)
        return "".join(captions)
    
    # 샘플 가져오기
    image, caption_tokens = dataset[idx]
    
    decoded_caption = decode_caption(caption_tokens[0].tolist())
    
    # results/tmp 폴더 생성
    os.makedirs("results/tmp", exist_ok=True)
    
    # 이미지 처리 및 저장
    plt.figure(figsize=(12, 8))
    
    if isinstance(image, torch.Tensor):
        if image.dim() == 3 and image.shape[0] == 3:
            img_np = image.permute(1, 2, 0).numpy()
            # 0-1 범위로 정규화
            img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())
        else:
            img_np = image.numpy()
    else:
        img_np = np.array(image)
    
    plt.imshow(img_np)
    plt.axis('off')
    plt.title(f"Sample {idx}: {decoded_caption}", fontsize=14, wrap=True, pad=20)
    plt.tight_layout()
    
    # 이미지 저장
    save_path = f"results/tmp/sample_{idx}.png"
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close()
    
    print(f"Image shape: {image.shape if hasattr(image, 'shape') else image.size}")
    print(f"Caption tokens shape: {caption_tokens.shape}")
    print(f"Decoded caption: {decoded_caption}")
    print(f"Image saved to: {save_path}")

if __name__ == "__main__":
    # 테스트용 코드
    print("데이터셋 로딩 중...")
    train_loader, val_loader = get_dataloaders(batch_size=4, num_workers=0)
    
    print(f"Train dataset size: {len(train_loader.dataset)}")
    print(f"Val dataset size: {len(val_loader.dataset)}")
    
    # 샘플 시각화
    print("\n=== 훈련 데이터 샘플들 ===")
    for i in range(3):
        print(f"\n--- Sample {i} ---")
        show_sample(train_loader.dataset, idx=i)
    
    print("\n=== 검증 데이터 샘플 ===")
    show_sample(val_loader.dataset, idx=0)
    
    # 배치 테스트
    print("\n=== 배치 테스트 ===")
    batch_images, batch_captions = next(iter(train_loader))
    print(f"Batch images shape: {batch_images.shape}")
    print(f"Batch captions length: {len(batch_captions)}")
    print(f"First caption tokens: {batch_captions[0].shape}")