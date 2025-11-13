import requests
import torch
from PIL import Image
from transformers import MllamaForConditionalGeneration, AutoProcessor
from huggingface_hub import login
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0" 
# HuggingFace 로그인
login(token="")

model_id = "meta-llama/Llama-3.2-11B-Vision-Instruct"

model = MllamaForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    #torch_dtype=torch.float32,
    device_map="auto",
)
processor = AutoProcessor.from_pretrained(model_id)

#url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/0052a70beed5bf71b92610a43a52df6d286cd5f3/diffusers/rabbit.jpg"
file_name = "테스트"
image_path = f"/home/ljm/ocr/picture/{file_name}.png"
image = Image.open(image_path)

english_prompt = """Analyze this image and provide the following information ONLY if applicable:

1. TEXT EXTRACTION: If there is any text in the image (including text in tables), extract it exactly as it appears. If there is no text, leave this section empty.

2. IMAGE DESCRIPTION: If there are visual elements (objects, scenes, diagrams, photos), describe them briefly. If the image only contains text with no visual elements, leave this section empty.

Output as a Json format with exactly 2 elements:
{"TEXT": "extracted text or NO", "IMAGE_DESCRIPTION": "image description or NO"}

**Important**:
- **Be concise and do not repeat information**"""

korean_prompt = """이미지를 분석하여 다음을 제공하세요:

1. 텍스트 추출: 이미지에 있는 모든 텍스트를 모두 다 정확히 추출하세요.(표 안에 있는 텍스트도 추출하세요)
2. 이미지 설명: 시각적 내용을 간단히 설명하세요 (객체, 장면 등).

다음 형식(Json)으로 응답하세요:
{
  "extracted_text": "**추출된** 텍스트 또는 'NO'",
  "image_description": "시각 설명 또는 'NO'"
}
주의사항:
**간결하게 작성하고 반복하지 말 것**
"""
Just_text_korean = """이 파일에서 텍스트를 뽑아내세요"""

Just_image_korean = """
파일에서 이미지를 찾아 다음 내용을 제공하세요:
**이미지 설명**: 시각적 내용을 간단히 설명하세요 (객체, 장면 등 포함).
이미지가 없고 텍스트만 있을 경우: NO 라고 출력하세요.

**중요 사항**:
**간결하게 작성하고 반복하지 마세요**
** 간결하게 작성하세요 (최대 250단어)**
"""
Just_text_english = """Analyze the file and extract the text.
"""

Just_image_english = """
Find images in the file and provide the following:

**Image Description**: Briefly describe the visual content (objects, scenes, etc.).
If there is no image (only text), output as follows: {
  "image_description": 'NO'
}

Respond in the following format (JSON):
{
  "image_description": "Visual description or 'NO'"
}
Important notes:
**Keep it concise and do not repeat**
"""
# combined_prompt = """Analyze this image and provide the following information ONLY if applicable

# 1. TEXT EXTRACTION: If there is any text in the image (including text in tables), extract it exactly as it appears. If there is no text, leave this section empty.

# 2. IMAGE DESCRIPTION: If there are visual elements (objects, scenes, diagrams, photos), describe them briefly. If the image only contains text with no visual elements, leave this section empty.

# LANGUAGE MATCHING
#    - If text is extracted AND visual elements exist:
#    - Describe the visual elements in the SAME LANGUAGE as the extracted text
#    - Korean text → Korean description
#    - English text → English description
#    - Mixed text → Use the dominant language
# **Output as a Json format with exactly 2 elements:**
# {"TEXT": text or NO, "IMAGE_DESCRIPTION": description or NO}

combined_prompt = """Analyze this image. Follow the format exactly.

RULES:
1. Extract all text exactly as written (including tables) using OCR. If no text → write "NO"
2. Describe visual elements directly. Do NOT use phrases like "This image shows" or "The image contains". If no visual elements → write "NO"
3. If both text and visuals exist, describe in the SAME language as the extracted text
4. Be concise and do not repeat information

Output as a Json format with exactly 2 elements:
{"TEXT": "extracted text or NO", "IMAGE_DESCRIPTION": "image description or NO"}
"""

messages = [
    {"role": "user", "content": [
        {"type": "image"},
        {"type": "text", "text": Just_text_korean}
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

print(raw_output)
# import re 
# import json 

# json_match = re.search(r'\{.*\}', raw_output, re.DOTALL)
# if json_match:
#     json_str = json_match.group(0)
#     print('---------------------------------------------------')
#     print('FINAL RESULT')
#     print('---------------------------------------------------')
#     print(json_str)
# else:

#print({"TEXT": "NO", "IMAGE_DESCRIPTION": "NO"})



# english_prompt = """Analyze this image and provide:

# 1. Extract all text: If there is any text in the image, transcribe it exactly as it appears.(Extract text in the table)
# 2. Describe the image: Provide a brief description of the visual content (objects, scenes, layout).

# Format your response as:
# TEXT: [extracted text or "No text found"]
# IMAGE DESCRIPTION: [brief image description or "No Image found"]

# Be concise and do not repeat information."""

# korean_prompt = """이미지를 분석하여 다음을 제공하세요:

# 1. 텍스트 추출: 이미지에 있는 모든 텍스트를 정확히 추출하세요.(표 안에 있는 텍스트도 추출하세요)
# 2. 이미지 설명: 시각적 내용을 간단히 설명하세요 (객체, 장면, 레이아웃).

# 다음 형식으로 응답하세요:
# 텍스트: [추출된 텍스트 또는 "텍스트 없음"]
# 이미지 설명: [간단한 이미지 설명 또는 "이미지 없음"]"""