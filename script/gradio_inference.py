import autorootcwd
import gradio as gr
import torch
import clip
from PIL import Image
import numpy as np
import os
from script.pretrain import ClipFinetuner

# 디바이스 설정
device = "cuda" if torch.cuda.is_available() else "cpu"

# 전역 변수
model = None
preprocess = None
is_finetuned = False

def load_model():
    """모델을 로드하는 함수"""
    global model, preprocess, is_finetuned
    
    if model is None:
        print("모델 로딩 중...")
        
        # 기본 CLIP 모델 로드
        base_clip_model, preprocess = clip.load("ViT-B/32", device=device)
        
        # Fine-tuned 모델 로드 시도
        checkpoint_path = "logs/clip_pretrain/version_0/checkpoints/epoch=0-step=1000.ckpt"
    
        clip_finetuner = ClipFinetuner.load_from_checkpoint(
            checkpoint_path,
            clip_model=base_clip_model
        )
        model = clip_finetuner.clip_model
        is_finetuned = True
        print("✅ Fine-tuned 모델 로드 완료")
        model.eval()
        model = model.to(device)
        print(f"모델 로드 완료! (Fine-tuned: {is_finetuned}, Device: {device})")

def quick_detect(image):
    """빠른 강아지/고양이 감지"""
    if image is None:
        return "이미지를 업로드해주세요."
    
    load_model()
    
    # 전처리
    image_tensor = preprocess(image).unsqueeze(0).to(device)
    labels = ["a dog", "a cat", "neither dog nor cat"]
    text = clip.tokenize(labels).to(device)
    
    with torch.no_grad():
        image_features = model.encode_image(image_tensor)
        text_features = model.encode_text(text)
        
        # 정규화
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)
        
        # 유사도 계산
        logits_per_image = 100 * image_features @ text_features.t()
        probs = logits_per_image.softmax(dim=-1).cpu().numpy()[0]
    
    # 결과 포맷팅
    results = []
    for label, prob in zip(labels, probs):
        results.append(f"**{label}**: {prob:.2%}")
    
    # 최고 예측
    best_idx = np.argmax(probs)
    best_result = f"\n🎯 **예측 결과: {labels[best_idx]}** (신뢰도: {probs[best_idx]:.2%})"
    
    model_info = f"📱 모델: {'Fine-tuned CLIP' if is_finetuned else 'Base CLIP'} | 디바이스: {device}\n\n"
    
    return model_info + "\n".join(results) + best_result

def custom_predict(image, desc1, desc2, desc3):
    """커스텀 3개 설명 예측"""
    if image is None:
        return "이미지를 업로드해주세요."
    
    descriptions = [desc1.strip(), desc2.strip(), desc3.strip()]
    
    if not all(descriptions):
        return "모든 설명을 입력해주세요."
    
    load_model()
    
    # 전처리
    image_tensor = preprocess(image).unsqueeze(0).to(device)
    text = clip.tokenize(descriptions).to(device)
    
    with torch.no_grad():
        image_features = model.encode_image(image_tensor)
        text_features = model.encode_text(text)
        
        # 정규화
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)
        
        # 유사도 계산
        logits_per_image = 100 * image_features @ text_features.t()
        probs = logits_per_image.softmax(dim=-1).cpu().numpy()[0]
    
    # 결과를 확률 순으로 정렬
    results = list(zip(descriptions, probs))
    results.sort(key=lambda x: x[1], reverse=True)
    
    # 결과 포맷팅
    output = f"📱 모델: {'Fine-tuned CLIP' if is_finetuned else 'Base CLIP'} | 디바이스: {device}\n\n"
    output += "📊 **분석 결과** (확률 순):\n\n"
    
    for i, (desc, prob) in enumerate(results):
        if i == 0:
            output += f"🥇 **{desc}**: {prob:.2%}\n"
        elif i == 1:
            output += f"🥈 **{desc}**: {prob:.2%}\n"
        else:
            output += f"🥉 **{desc}**: {prob:.2%}\n"
    
    # 최고 예측 강조
    best_desc, best_prob = results[0]
    output += f"\n🎯 **모델의 최종 선택**: '{best_desc}' (신뢰도: {best_prob:.2%})"
    
    return output

# Gradio 인터페이스 생성
def create_interface():
    with gr.Blocks(title="🐕🐱 CLIP Dog & Cat Detection", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            """
            # 🐕🐱 CLIP을 이용한 강아지/고양이 감지
            
            Fine-tuned CLIP 모델을 사용하여 이미지를 분석합니다.
            """
        )
        
        with gr.Tab("🚀 빠른 감지"):
            gr.Markdown("### 강아지인지 고양이인지 빠르게 감지해보세요!")
            
            with gr.Row():
                with gr.Column():
                    image_input1 = gr.Image(type="pil", label="이미지 업로드")
                    detect_btn = gr.Button("🔍 감지하기", variant="primary")
                
                with gr.Column():
                    quick_output = gr.Markdown(label="결과")
            
            detect_btn.click(
                fn=quick_detect,
                inputs=[image_input1],
                outputs=[quick_output]
            )
        
        with gr.Tab("🎮 커스텀 테스트"):
            gr.Markdown(
                """
                ### 2개의 거짓과 1개의 진실
                이미지에 대한 3개의 설명을 입력하세요. 모델이 어떤 설명을 가장 적합하다고 판단하는지 확인해보세요!
                """
            )
            
            with gr.Row():
                with gr.Column():
                    image_input2 = gr.Image(type="pil", label="이미지 업로드")
                    
                    desc1_input = gr.Textbox(
                        label="설명 1", 
                        placeholder="예: 빨간 사과",
                        lines=2
                    )
                    desc2_input = gr.Textbox(
                        label="설명 2", 
                        placeholder="예: 차고에 주차된 자동차",
                        lines=2
                    )
                    desc3_input = gr.Textbox(
                        label="설명 3", 
                        placeholder="예: 나무에 달린 오렌지",
                        lines=2
                    )
                    
                    predict_btn = gr.Button("🔮 예측하기", variant="primary")
                
                with gr.Column():
                    custom_output = gr.Markdown(label="분석 결과")
            
            predict_btn.click(
                fn=custom_predict,
                inputs=[image_input2, desc1_input, desc2_input, desc3_input],
                outputs=[custom_output]
            )
        
        with gr.Tab("ℹ️ 정보"):
            gr.Markdown(
                """
                ### 사용법
                
                **빠른 감지 탭:**
                - 강아지나 고양이 이미지를 업로드하고 '감지하기' 버튼을 클릭
                - 모델이 강아지/고양이/기타 중에서 선택
                
                **커스텀 테스트 탭:**
                - 이미지를 업로드하고 3개의 설명을 입력
                - 모델이 어떤 설명이 가장 적합한지 판단
                - "2개의 거짓과 1개의 진실" 게임처럼 즐길 수 있습니다
                
                ### 팁
                - 명확하고 구체적인 설명을 사용하세요
                - 모델은 시각적 특징을 기반으로 판단합니다
                - Fine-tuned 모델이 있으면 자동으로 로드됩니다
                """
            )
    
    return demo

if __name__ == "__main__":
    # 인터페이스 생성 및 실행
    demo = create_interface()
    demo.launch(
        share=True,  # 공개 링크 생성
        server_name="0.0.0.0",  # 외부 접속 허용
        server_port=7860,
        show_error=True
    )