from vllm import LLM, SamplingParams
from vllm.multimodal.utils import fetch_image
import torch
from PIL import Image
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

# vLLM 모델 로드
model_id = "meta-llama/Llama-3.2-11B-Vision-Instruct"
llm = LLM(
    model=model_id,
    max_model_len=2048,
    max_num_seqs=1,
    gpu_memory_utilization=0.5,
    trust_remote_code=True,
    enforce_eager=True  # 추가
)

# Sampling 파라미터 설정
sampling_params = SamplingParams(
    temperature=0.0,
    max_tokens=3000,
    top_p=0.9
)

image_path = "/home/ljm/ocr/pircutre/국방부.png"

# 1. Language Detection
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
    {
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": {"url": image_path}},
            {"type": "text", "text": language_detect}
        ]
    }
]

outputs = llm.chat(messages, sampling_params=sampling_params)
raw_output = outputs[0].outputs[0].text

print(raw_output)

import re
import json

def extract_language_from_text(text):
    """텍스트에서 언어명을 추출하는 함수"""
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
    print('---------------------------------------------------')
    print('language RESULT')
    print('---------------------------------------------------')
    print(json_str)
    try:
        json_data = json.loads(json_str)
    except json.JSONDecodeError:
        json_data = extract_language_from_text(raw_output)
else:
    json_data = extract_language_from_text(raw_output)

language = json_data['detected_language']
print(json_data['detected_language'])

# 2. Text & Image Extraction
Just_text_english = """Analyze the file and provide the following:
**Text Extraction**: Extract all text accurately (including text inside tables).
Extract the text as-is without any explanations.
If there is no text, output: NO

Important notes:
**Keep it concise and do not repeat**
"""

Just_image_english = """
Find images in the file and provide the following:
**Image Description**: Briefly describe the visual content (objects, scenes, etc.).
If there is no image (only text), output: NO

Important notes:
**Keep it concise and do not repeat**
"""

Just_text_korean = """파일을 분석하여 다음을 제공하세요:
텍스트 추출: 모든 텍스트를 정확히 추출하세요 (표 안의 텍스트 포함).
텍스트를 그대로 추출하고 설명은 하지 마세요.
텍스트가 없으면: NO

주의사항:
**간결하게 작성하고 반복하지 말 것**
"""

Just_image_korean = """
파일에서 이미지를 찾아 다음을 제공하세요:
이미지 설명: 시각적 내용을 간단히 설명하세요 (객체, 장면 등).
이미지가 없으면 (텍스트만 있으면): NO

주의사항:
**간결하게 작성하고 반복하지 말 것**
"""

if language.lower() == 'korean':
    prompt = [Just_text_korean, Just_image_korean]
else:
    prompt = [Just_text_english, Just_image_english]

print(prompt)
generated_output = []

for i in range(2):
    sys_prompt = prompt[i]
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": image_path}},
                {"type": "text", "text": sys_prompt}
            ]
        }
    ]
    
    outputs = llm.chat(messages, sampling_params=sampling_params)
    raw_output = outputs[0].outputs[0].text
    
    generated_output.append(raw_output)

print('---------------------------------------------------')
print('FINAL RESULT')
print('---------------------------------------------------')
print(generated_output)

# JSON 형태로 최종 결과 정리
final_result = {
    "extracted_text": generated_output[0],
    "image_description": generated_output[1]
}

print(json.dumps(final_result, ensure_ascii=False, indent=2))