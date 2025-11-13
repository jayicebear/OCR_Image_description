import os
import json
import time
import torch
from PIL import Image
from tqdm import tqdm
from unsloth import FastVisionModel


# 모델 불러오기
def load_model():
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    
    print("모델 로딩 중...")
    model, tokenizer = FastVisionModel.from_pretrained(
        "unsloth/Llama-3.2-11B-Vision-Instruct-bnb-4bit",
        load_in_4bit=True,
        use_gradient_checkpointing="unsloth",
    )
    FastVisionModel.for_inference(model)
    print("모델 로드 완료")
    
    return model, tokenizer


# 언어 감지
def detect_language(model, tokenizer, image):
    language_prompt = """Analyze the text in this image and identify the language.
Respond with only the language name: Korean, English, Japanese, Chinese, etc.
If no text, respond: Unknown"""

    messages = [{
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": language_prompt}
        ]
    }]

    input_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
    inputs = tokenizer(image, input_text, add_special_tokens=False, return_tensors="pt").to("cuda")

    with torch.inference_mode():
        output = model.generate(**inputs, max_new_tokens=50, temperature=0.3)
    
    raw_output = tokenizer.decode(output[0], skip_special_tokens=True)
    
    # 언어 감지 (fallback)
    for lang in ['Korean', 'English', 'Japanese', 'Chinese', 'Spanish', 
                 'French', 'German', 'Italian', 'Portuguese', 'Russian']:
        if lang.lower() in raw_output.lower():
            return lang
    
    return "English"


# 텍스트 / 이미지 분석
def analyze_content(model, tokenizer, image, language):
    prompts = {
        "english": [
            "Extract all text from this image accurately (including text in tables). Output only the text without explanations. If no text, output: NO",
            "Describe the visual content of this image briefly (objects, scenes, etc.). If only text and no image content, output: NO"
        ],
        "korean": [
            "이미지에서 모든 텍스트를 정확하게 추출하세요 (표 안의 텍스트 포함). 설명 없이 텍스트만 출력하세요. 텍스트가 없으면: NO",
            "이미지의 시각적 내용을 간단히 설명하세요 (객체, 장면 등). 텍스트만 있고 이미지 내용이 없으면: NO"
        ]
    }

    selected = prompts["korean"] if language.lower() == "korean" else prompts["english"]
    outputs = []

    for prompt in tqdm(selected, desc="분석 중", ncols=100):
        messages = [{
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt}
            ]
        }]

        input_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
        inputs = tokenizer(image, input_text, add_special_tokens=False, return_tensors="pt").to("cuda")

        with torch.inference_mode():
            output = model.generate(**inputs, max_new_tokens=512, temperature=0.5, top_p=0.9)
        rst = tokenizer.decode(output[0], skip_special_tokens=True)
        print(rst)
        decoded = rst.split('assistant')[1]
        outputs.append(decoded.replace("\n", ""))

    return outputs


# 후처리 (NONE 처리)
def postprocess_output(generated_output):
    text_none_keywords = [
        "텍스트가 없", "텍스트는 없", "no text", "no content", 
        "the image", "this image", "없습니다"
    ]
    image_none_keywords = [
        "이미지가 없", "이미지는 없", "no image", "no images", 
        "the text", "this text", "없습니다"
    ]

    text_result = generated_output[0]
    image_result = generated_output[1]

    if any(kw in text_result.lower() for kw in text_none_keywords):
        text_result = "NONE"
    
    if any(kw in image_result.lower() for kw in image_none_keywords):
        image_result = "NONE"

    return {
        "TEXT": text_result,
        "IMAGE_DESCRIPTION": image_result
    }


# 결과 저장
def save_result(result, file_name):
    save_dir = "/home/ljm/ocr/result"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{file_name}_result.json")

    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"결과 저장 완료: {save_path}")


# 전체 실행
def main():
    total_start = time.time()
    
    # 설정
    file_name = "dog"
    image_path = f"/home/ljm/ocr/picture/{file_name}.png"
    
    # 모델 로드
    model, tokenizer = load_model()
    
    # 이미지 로드
    image = Image.open(image_path)
    print(f"이미지 로드: {file_name}.png")
    
    # 언어 감지
    language = detect_language(model, tokenizer, image)
    print(f"감지된 언어: {language}")
    
    # 텍스트 & 이미지 분석
    generated_output = analyze_content(model, tokenizer, image, language)
    
    # 후처리
    result = postprocess_output(generated_output)
    
    # 결과 출력
    total_time = time.time() - total_start
    print("\n" + "=" * 50)
    print("최종 결과")
    print("=" * 50)
    print(result)
    print("=" * 50)
    print(f"처리 시간: {total_time:.2f}초")
    print("=" * 50)
    
    # 결과 저장
    save_result(result, file_name)


if __name__ == "__main__":
    main()