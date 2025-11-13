from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
import os
import time 

os.environ["CUDA_VISIBLE_DEVICES"] = "0" 

# default: Load the model on the available device(s)
model = Qwen3VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen3-VL-8B-Instruct", dtype="auto" #device_map="auto"
)

# We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.
# model = Qwen3VLForConditionalGeneration.from_pretrained(
#     "Qwen/Qwen3-VL-8B-Instruct",
#     dtype=torch.bfloat16,
#     attn_implementation="flash_attention_2",
#     device_map="auto",
# )


english_prompt = """Extract texts in this file."""
english_prompt_image = """Analyze this image and describe it concisely."""


korean_prompt = """이 파일에서 텍스트를 뽑아내세요"""
korean_prompt_image="""이 파일에서 이미지를 설명하세요"""

#"""이미지를 간단히 설명하세요""" 
processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-8B-Instruct")

file_name = '테스트'
image_path = f"/home/ljm/ocr/picture/{file_name}.png"

start_time = time.time()
#--------------------------------text 추출--------------------------------------

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": image_path
            },
            {"type": "text", "text": korean_prompt},
        ],
    }
]

# Preparation for inference
inputs = processor.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_dict=True,
    return_tensors="pt"
)
inputs = inputs.to(model.device)

# Inference: Generation of the output

generated_ids = model.generate(**inputs,
                               top_p=0.8,top_k=20,temperature=0.7,
                               max_new_tokens=4096,
        do_sample=True,
        # 반복 안하게 하는 부분
        repetition_penalty=1.0)
generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
extracted_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)

#--------------------------------이미지 설명--------------------------------------
messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": image_path
            },
            {"type": "text", "text": korean_prompt_image},
        ],
    }
]

# Preparation for inference
inputs = processor.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_dict=True,
    return_tensors="pt"
)
inputs = inputs.to(model.device)

# Inference: Generation of the output

generated_ids = model.generate(**inputs,
                               top_p=0.8,top_k=20,temperature=0.7,
                               max_new_tokens=4096,
        do_sample=True,
        # 반복 안하게 하는 부분
        repetition_penalty=1.0)
generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
image_description = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)


result = {
        "extracted_text": extracted_text[0].strip() or "None",
        "image_description": image_description[0].strip() or "None",
        
    }
end_time = time.time() - start_time 

print(end_time)

print(result)
