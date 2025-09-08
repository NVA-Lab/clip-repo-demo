import autorootcwd
import torch
import clip
from PIL import Image
import numpy as np
from script.pretrain import ClipFinetuner

# 디바이스 설정
device = "cuda" if torch.cuda.is_available() else "cpu"

# 기본 CLIP 모델 로드 (체크포인트 로드를 위해 필요)
base_clip_model, preprocess = clip.load("ViT-B/32", device=device)

# 체크포인트에서 fine-tuned 모델 로드
clip_finetuner = ClipFinetuner.load_from_checkpoint(
    "logs/clip_pretrain/version_0/checkpoints/epoch=1-step=10000.ckpt",
    clip_model=base_clip_model
)
clip_model = clip_finetuner.clip_model
clip_model.eval()
clip_model = clip_model.to(device)

# 이미지 로드 및 전처리
image = preprocess(Image.open("dog.jpg")).unsqueeze(0).to(device)

# 텍스트 토큰화
text = clip.tokenize(["a diagram", "a dog", "a cat"]).to(device)

# 추론
with torch.no_grad():
    # fine-tuned 모델로 특성 추출
    image_features = clip_model.encode_image(image)
    text_features = clip_model.encode_text(text)
    
    # 정규화
    image_features = image_features / image_features.norm(dim=1, keepdim=True)
    text_features = text_features / text_features.norm(dim=1, keepdim=True)
    
    # 유사도 계산 (logit scale 적용)
    logit_scale = clip_finetuner.logit_scale.exp()
    logits_per_image = logit_scale * image_features @ text_features.t()
    
    # 확률 계산
    probs = logits_per_image.softmax(dim=-1).cpu().numpy()

print("Label probs:", probs)