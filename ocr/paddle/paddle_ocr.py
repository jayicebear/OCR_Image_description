from PIL import Image
import torch
from transformers import AutoModelForCausalLM, AutoProcessor
import os 
import time 

start_time = time.time()
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
os.environ["CUDA_VISIBLE_DEVICES"] = "3" 

CHOSEN_TASK = "ocr"  # Options: 'ocr' | 'table' | 'chart' | 'formula'
PROMPTS = {
    "ocr": "OCR:",
    "table": "Table Recognition:",
    "formula": "Formula Recognition:",
    "chart": "Chart Recognition:",
}

model_path = "PaddlePaddle/PaddleOCR-VL"
# image_path = "test.png"
# image = Image.open(image_path).convert("RGB")

file_name = 'dog'
image_path = f"/home/ljm/ocr/picture/{file_name}.png"
image = Image.open(image_path)

model = AutoModelForCausalLM.from_pretrained(
    model_path , trust_remote_code=True, torch_dtype=torch.bfloat16
).to(DEVICE).eval()
processor = AutoProcessor.from_pretrained(model_path , trust_remote_code=True)

messages = [
    {"role": "user",         
     "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": PROMPTS[CHOSEN_TASK]},
        ]
    }
]
inputs = processor.apply_chat_template(
    messages, 
    tokenize=True, 
    add_generation_prompt=True, 	
    return_dict=True,
    return_tensors="pt"
).to(DEVICE)

outputs = model.generate(**inputs, max_new_tokens=1024)
outputs = processor.batch_decode(outputs, skip_special_tokens=True)[0]

end_time = time.time() - start_time 

print('---------------------------------------------')
print(outputs)
print('time:',end_time)