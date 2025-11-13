import torch
import re
import json
import time
import os
from PIL import Image
from tqdm import tqdm
from transformers import MllamaForConditionalGeneration, AutoProcessor
from huggingface_hub import login
from transformers import BitsAndBytesConfig

os.environ["CUDA_VISIBLE_DEVICES"] = "3"

# 모델 불러오기
def load_model():
    # Huggingface 로그인 
    login(token="")
#     bnb_config = BitsAndBytesConfig(
#     load_in_4bit=True,                # 4bit 양자화
#     bnb_4bit_use_double_quant=True,
#     bnb_4bit_quant_type="nf4",        # QLoRA 스타일 양자화
#     bnb_4bit_compute_dtype=torch.bfloat16
# )
    model_id = "meta-llama/Llama-3.2-11B-Vision-Instruct"
    model = MllamaForConditionalGeneration.from_pretrained(
        model_id,
        #quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(model_id)
    return model, processor


# 언어 감지
def detect_language(model, processor, image):
    language_prompt = """
    Analyze the extracted text and identify the language.

    Respond in JSON:
    {
      "detected_language": "Language name (e.g., Korean, English, Japanese, Chinese, etc.)"
    }
    If no text, output {"detected_language": "Unknown"}
    """

    messages = [{"role": "user", "content": [
        {"type": "image"},
        {"type": "text", "text": language_prompt}
    ]}]

    input_text = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(image, input_text, add_special_tokens=False, return_tensors="pt").to(model.device)

    output = model.generate(**inputs, max_new_tokens=3000)
    raw_output = processor.decode(output[0]).split('<|end_header_id|>')[2].split('<|eot_id|>')[0]

    # json_match = re.search(r'\{.*\}', raw_output, re.DOTALL)
    # if json_match:
    #     try:
    #         data = json.loads(json_match.group(0))
    #         return data.get("detected_language", "English")
    #     except json.JSONDecodeError:
    #         pass

    # fallback
    for lang in ['Korean', 'English', 'Japanese', 'Chinese', 'Spanish', 
                 'French', 'German', 'Italian', 'Portuguese', 'Russian',
                 'Arabic', 'Hindi', 'Thai', 'Vietnamese']:
        if lang.lower() in raw_output.lower():
            return lang
    print(lang)
    return "English"
     

# 텍스트 / 이미지 추출
def generate_output(model, processor, image, language):
    prompts = {
        "english": [
            """Analyze the file and provide the following:
**Text Extraction**: Extract all text accurately (including text inside tables).
Extract the text as-is without any explanations.
If there is no text, output: NO

**Important notes**:
**Keep it concise and do not repeat**
""",

            """
Find images in the file and provide the following:
**Image Description**: Briefly describe the visual content (objects, scenes, etc.).
If there is no image (only text), output: NO

**Important notes**:
**Keep it concise and do not repeat**
**Keep it VERY concise (maximum 250 words)** 
"""
        ],
        "korean": [
            """파일을 분석하고 다음 내용을 제공하세요:
**텍스트 추출**: 모든 텍스트를 정확하게 추출하세요 (표 안의 텍스트도 포함).
설명 없이 텍스트만 그대로 추출하세요.
텍스트가 없을 경우: NO 라고 출력하세요.

**중요 사항**:
**간결하게 작성하고 반복하지 마세요**
""",

            """
파일에서 이미지를 찾아 다음 내용을 제공하세요:
**이미지 설명**: 시각적 내용을 간단히 설명하세요 (객체, 장면 등 포함).
이미지가 없고 텍스트만 있을 경우: NO 라고 출력하세요.

**중요 사항**:
**간결하게 작성하고 반복하지 마세요**
** 간결하게 작성하세요 (최대 250단어)**
"""
        ]
    }

    selected = prompts["korean"] if language.lower() == "korean" else prompts["english"]
    outputs = []

    for i in tqdm(range(2), desc="Generating", ncols=100):
        msg = [{"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": selected[i]}
        ]}]
        input_text = processor.apply_chat_template(msg, add_generation_prompt=True)
        inputs = processor(image, input_text, add_special_tokens=False, return_tensors="pt").to(model.device)

        output = model.generate(**inputs, 
                                max_new_tokens=3000,  
                                #do_sample=True,
                                #top_p=0.6,
                                #temperature=0.5,
                                repetition_penalty=1.5)
        
        decoded = processor.decode(output[0]).split('<|end_header_id|>')[2].split('<|eot_id|>')[0]
        outputs.append(decoded.replace("\n", ""))

    return outputs


# 후처리 (NONE 처리)
def postprocess_output(generated_output):
    text_none_keywords = [
        "텍스트가 없", "텍스트는 없", "텍스트 정보를","텍스트를 찾지",
        "no text", "no content", "the image","this image"
    ]
    image_none_keywords = [
        "이미지가 없", "이미지는 없", "이미지 정보를","이미지를 찾지",
        "no image", "no images", "no content", "the text", "this text"
    ]

    if any(kw in generated_output[0].lower() for kw in text_none_keywords):
        generated_output[0] = "NONE"
    if any(kw in generated_output[1].lower() for kw in image_none_keywords):
        generated_output[1] = "NONE"

    return {"TEXT": generated_output[0], "IMAGE_DESCRIPTION": generated_output[1]}

# 이미지 & 텍스트 json 으로 저장
def save_result(result, file_name):
    save_dir = "/home/ljm/ocr/result"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{file_name}_result.json")

    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f" 결과 저장 완료: {save_path}")


# 전체 실행
if __name__ == "__main__":
    total_start = time.time()
    file_name = "테스트"

    model, processor = load_model()
    image_path = f"/home/ljm/ocr/picture/{file_name}.png"
    image = Image.open(image_path)

    language = detect_language(model, processor, image)
    print(f" Detected language: {language}")

    generated_output = generate_output(model, processor, image, language)
    result = postprocess_output(generated_output)

    total_time = time.time() - total_start
    print("---------------------------------------------------")
    print("FINAL RESULT")
    print("---------------------------------------------------")
    print(result)
    print(f" 처리 시간: {total_time:.2f}초")
    save_result(result, file_name)