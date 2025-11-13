import requests
import torch
from PIL import Image
from transformers import MllamaForConditionalGeneration, AutoProcessor
from huggingface_hub import login
import os
import time
from tqdm import tqdm
import time 
os.environ["CUDA_VISIBLE_DEVICES"] = "3" 
# HuggingFace 로그인
login(token="")
total_start = time.time()
model_id = "meta-llama/Llama-3.2-11B-Vision-Instruct"


start_time = time.time()

model = MllamaForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    #torch_dtype=torch.float32,
    device_map="auto",
)
processor = AutoProcessor.from_pretrained(model_id)

#url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/0052a70beed5bf71b92610a43a52df6d286cd5f3/diffusers/rabbit.jpg"
file_name = "dog"
image_path = f"/home/ljm/ocr/pircutre/{file_name}.png"
#image = Image.open(requests.get(url, stream=True).raw)
image = Image.open(image_path)

language_detect = """

Analyze the extracted text and identify the language.

Respond in the following format (JSON):
{
  "detected_language": "Language name (e.g., Korean, English, Japanese, Chinese, etc.)"
}

If no text is available or language cannot be determined, output:
{
  "detected_language": "Unknown"
}

Important notes:
**Keep it concise and do not repeat**
"""

messages = [
    {"role": "user", "content": [
        {"type": "image"},
        {"type": "text", "text": language_detect}
    ]}
]
input_text = processor.apply_chat_template(messages, add_generation_prompt=True)
inputs = processor(
    image,
    input_text,
    add_special_tokens=False,
    return_tensors="pt"
).to(model.device)

output = model.generate(**inputs, max_new_tokens=3000)
raw_output = processor.decode(output[0]).split('<|end_header_id|>')[2].split('<|eot_id|>')[0] 

#print(raw_output)
import re 
import json 

def extract_language_from_text(text):
    """텍스트에서 언어명 추출"""
    languages = ['Korean', 'English', 'Japanese', 'Chinese', 'Spanish', 
                 'French', 'German', 'Italian', 'Portuguese', 'Russian',
                 'Arabic', 'Hindi', 'Thai', 'Vietnamese']
    
    text_lower = text.lower()
    for lang in languages:
        if lang.lower() in text_lower:
            return {"detected_language": lang}
    
    return {"detected_language": "English"}

json_match = re.search(r'\{.*\}', raw_output, re.DOTALL)
if json_match:
    json_str = json_match.group(0)
    #print('---------------------------------------------------')
    #print('language RESULT')
    #print('---------------------------------------------------')
    #print(json_str)
    try:
        json_data = json.loads(json_str)
    except json.JSONDecodeError:
        # JSON 파싱 실패 시 텍스트에서 언어 추출
        json_data = extract_language_from_text(raw_output)
else:
    # JSON 형식이 아닌 경우 텍스트에서 언어 추출
    json_data = extract_language_from_text(raw_output)


language = json_data['detected_language']
#print(json_data['detected_language'])


## 언어 추출 후 최종 output 뽑기 

combined_prompt = f"""Analyze the file and respond in {language}.

**Language**: {language}

Outputs:
1. **Text Extraction**: Extract all text exactly as written (including text inside tables). If no text exists → output "NO"
2. **Image Description**: Briefly describe the visual content (objects, scenes, etc.). If no image exists → output "NO"  

**Critical**: Write all output in {language}
Be concise and do not repeat information

Respond in the following JSON format:
{{
  "extracted_text": "Extracted text or 'NO'",
  "image_description": "Visual description in {language} or 'NO'"
}}"""

# 프롬프트에서 JSON 형식 요구 제거
Just_text_english = f"""Analyze the file and provide the following:
**Text Extraction**: Extract all text accurately (including text inside tables).
Extract the text as-is without any explanations.
If there is no text, output: NO

Important notes:
**Keep it concise and do not repeat**
"""

Just_image_english = f"""
Find images in the file and provide the following:
**Image Description**: Briefly describe the visual content (objects, scenes, etc.).
If there is no image (only text), output: NO

Important notes:
**Keep it concise and do not repeat**
"""

# Just_text_korean = f"""파일을 분석하여 다음을 제공하세요:
# 텍스트 추출: 모든 텍스트를 정확히 추출하세요 (표 안의 텍스트 포함).
# 텍스트를 그대로 추출하고 설명은 하지 마세요.
# 텍스트가 없으면: NO

# 주의사항:
# **간결하게 작성하고 반복하지 말 것**
# """

# Just_image_korean = f"""
# 파일에서 이미지를 찾아 다음을 제공하세요:
# 이미지 설명: 시각적 내용을 간단히 설명하세요 (객체, 장면 등).
# 이미지가 없으면 (텍스트만 있으면): NO

# 주의사항:
# **간결하게 작성하고 반복하지 말 것**
# """
Just_text_korean = f"""파일을 분석하고 다음 내용을 제공하세요:
**텍스트 추출**: 모든 텍스트를 정확하게 추출하세요 (표 안의 텍스트도 포함).
설명 없이 텍스트만 그대로 추출하세요.
텍스트가 없을 경우: NO 라고 출력하세요.

중요 사항:
**간결하게 작성하고 반복하지 마세요**
"""

Just_image_korean = f"""
파일에서 이미지를 찾아 다음 내용을 제공하세요:
**이미지 설명**: 시각적 내용을 간단히 설명하세요 (객체, 장면 등 포함).
이미지가 없고 텍스트만 있을 경우: NO 라고 출력하세요.

중요 사항:
**간결하게 작성하고 반복하지 마세요**
"""
if language.lower() == 'korean':
    prompt = [Just_text_korean, Just_image_korean]
else:
    prompt = [Just_text_english, Just_image_english]

#print(prompt)
generated_output = []
for i in tqdm(range(2), desc="processing", ncols=100):
    sys_prompt = prompt[i]
    messages = [
        {"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": sys_prompt}
        ]}
    ]
    input_text = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(
        image,
        input_text,
        add_special_tokens=False,
        return_tensors="pt"
    ).to(model.device)

    output = model.generate(**inputs, max_new_tokens=3000)
    raw_output = processor.decode(output[0]).split('<|end_header_id|>')[2].split('<|eot_id|>')[0] 

    generated_output.append(raw_output)

generated_output = [item.replace("\n", "") for item in generated_output]

# 텍스트나 이미지가 없는 부분 그냥 NONE 으로 처리하는 부분 
text_none_keywords = [
    "텍스트가 없", "텍스트는 없", "텍스트 정보를","텍스트를 찾지", "no text", "no content", "the image","this image"
]

image_none_keywords = [
    "이미지가 없", "이미지는 없", "이미지 정보를","이미지를 찾지", "no image", "no images", "no content", "the text", "this text"
]

# 텍스트 부분 처리
if any(kw in generated_output[0].lower() for kw in text_none_keywords):
    generated_output[0] = "NONE"

# 이미지 설명 부분 처리
if any(kw in generated_output[1].lower() for kw in image_none_keywords):
    generated_output[1] = "NONE"
    
result = {"TEXT": generated_output[0], "IMAGE_DESCRIPTION": generated_output[1]}
inference_start = time.time()
total_time = inference_start - total_start

print('---------------------------------------------------')
print('FINAL RESULT')
print('---------------------------------------------------')
print(result)
print(f"시간: {total_time:.2f}초")