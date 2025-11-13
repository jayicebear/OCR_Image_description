from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer
from PIL import Image
import torch
import time
import os
import json
import re

# --------------------------------------------------------------
# 모델 로드 
# --------------------------------------------------------------
def load_models(
    ocr_model_path="PaddlePaddle/PaddleOCR-VL",
    desc_model_name="naver-hyperclovax/HyperCLOVAX-SEED-Vision-Instruct-3B",
    ocr_device_id=3, 
    desc_device_id=3
):
    
    # OCR 모델 로드
    DEVICE_OCR = f"cuda:{ocr_device_id}" 

    print(f"Loading OCR model ({ocr_model_path}) on device ")
    ocr_model = AutoModelForCausalLM.from_pretrained(
        ocr_model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    ).to(DEVICE_OCR).eval()

    ocr_processor = AutoProcessor.from_pretrained(
        ocr_model_path,
        trust_remote_code=True,
        use_fast=True
    )

    # image 설명 모델 로드
    DEVICE_DESC = f"cuda:{desc_device_id}" if torch.cuda.is_available() else "cpu"

    print(f"Loading Description model ({desc_model_name}) on device {DEVICE_DESC}...")
    desc_model = AutoModelForCausalLM.from_pretrained(
        desc_model_name,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16
    ).to(DEVICE_DESC).eval()

    desc_processor = AutoProcessor.from_pretrained(desc_model_name, trust_remote_code=True)
    desc_tokenizer = AutoTokenizer.from_pretrained(desc_model_name)
    
    loading_info = {
        "ocr_model": ocr_model,
        "ocr_processor": ocr_processor,
        "desc_model": desc_model,
        "desc_processor": desc_processor,
        "desc_tokenizer": desc_tokenizer,
        "device_ocr": DEVICE_OCR,
        "device_desc": DEVICE_DESC
    }
    #print(loading_info)
    return loading_info


# --------------------------------------------------------------
#  image 전체 분석 process
# --------------------------------------------------------------
def analyze_image(image_path: str, models: dict):
    """
    Run OCR (PaddleOCR-VL) and Image Description (HyperCLOVAX)
    using preloaded models.
    """

    image = Image.open(image_path).convert("RGB")
    PROMPTS = {
        "ocr": "OCR:",
        "table": "Table Recognition:",
        "formula": "Formula Recognition:",
        "chart": "Chart Recognition:",
    }

    ocr_model = models["ocr_model"]
    ocr_processor = models["ocr_processor"]
    desc_model = models["desc_model"]
    desc_processor = models["desc_processor"]
    DEVICE_OCR = models["device_ocr"]
    DEVICE_DESC = models["device_desc"]

    # --------------------- OCR ---------------------
    

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": PROMPTS["ocr"]},
            ],
        }
    ]

    ocr_inputs = ocr_processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    ).to(DEVICE_OCR)
    start_ocr = time.time()
    ocr_outputs = ocr_model.generate(**ocr_inputs, max_new_tokens=1024)
    extracted_text = ocr_processor.batch_decode(ocr_outputs, skip_special_tokens=True)[0]
    
    # preprocess 
    extracted_text = extracted_text.replace("User: OCR:\nAssistant:", "").replace("\n", " ").replace("<lcel>", "").replace("<fcel>", " ").replace("<nl>","").replace("<ucel>","").replace("\\",'')
    if len(extracted_text.strip()) == 1:
        extracted_text = extracted_text.replace("1", "None").replace("0", "None")
    ocr_elapsed = time.time() - start_ocr
    
    # Extracted text language portion extraction
    if len(extracted_text) < 1:
        #extracted_text_portion = 'English'
        extracted_text_portion = 'korean'
    else:    
        extracted_text_portion = extracted_text[:10]

    # ------------------ Image Description ------------------
    
    start_desc = time.time()
    system_prompt = "You are a helpful image analyzer"
    user_prompt = f"""
Find images in the file and provide the following:
Describe the visual content in the same language as the extracted text below.
A portion of extracted text: {extracted_text_portion}
If there is no image (only text), output: NO

**Important notes**:
**Keep it concise(maximum 300 words) and do not repeat**
"""

    vlm_chat = [
        {"role": "system", "content": [{"text": system_prompt, "type": "text"}]},
        {"role": "user", "content": [{"text": user_prompt, "type": "text"}]},
        {
            "role": "user",
            "content": [
                {
                    "filename": os.path.basename(image_path),
                    "image": image,
                    "type": "image",
                }
            ],
        },
    ]

    desc_inputs = desc_processor.apply_chat_template(
        vlm_chat,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        add_generation_prompt=True,
    ).to(DEVICE_DESC)

    output_ids = desc_model.generate(
        **desc_inputs,
        max_new_tokens=4096,
        do_sample=True,
        top_p=0.8,
        temperature=0.7,
        # 반복 안하게 하는 부분
        repetition_penalty=1.0,
    )

    image_description = desc_processor.batch_decode(output_ids, skip_special_tokens=True)[0]
    image_description = (
        image_description.replace("<|im_end|><|endofturn|>", "")
        .replace("**이미지 설명**:", "")
        .replace("\n", " ").replace("\\",'')
    )
    desc_elapsed = time.time() - start_desc
    # print TIME
    total_elapsed = ocr_elapsed + desc_elapsed
    time_total = {"elapsed_time": {
            "ocr": round(ocr_elapsed, 2),
            "description": round(desc_elapsed, 2),
            "total": round(total_elapsed, 2),
        }}
    
    print('Time:',time_total)
    # Result
    result = {
        "extracted_text": extracted_text.strip() or "None",
        "image_description": image_description.strip() or "None",
        
    }

    return result


# --------------------------------------------------------------
#  MAIN
# --------------------------------------------------------------
if __name__ == "__main__":
    #os.environ["CUDA_VISIBLE_DEVICES"] = "3"
    # 모델 로드 
    models = load_models(ocr_device_id=3, desc_device_id=3)

    # 이미지 목록
    file_name = "뉴스"
    image_list = [
       f"/home/ljm/ocr/picture/{file_name}.png",
        #"/home/ljm/ocr/picture/dog.png",
        # "/home/ljm/ocr/picture/example.png",
    ]

    # 결과 저장 경로
    output_dir = "/home/ljm/ocr/result"
    os.makedirs(output_dir, exist_ok=True)

    # 이미지 분석
    for image_path in image_list:
        start_time = time.time()
        result = analyze_image(image_path, models)
        total_time = time.time() - start_time

        print(f"\n Processed: {image_path}")
        print(json.dumps(result, indent=2, ensure_ascii=False))
        print(f"Total time: {total_time:.2f}s")

        # JSON 저장
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        output_path = os.path.join(output_dir, f"{base_name}_result.json")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"Saved to: {output_path}")
