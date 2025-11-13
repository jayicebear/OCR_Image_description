from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer
from PIL import Image
import torch
import time
import os
import json


def analyze_image(
    image_path: str,
    device_ocr: str = "cuda",
    device_desc: str = "cuda",
    ocr_device_id: str = "0",
    desc_device_id: str = "0",
    ocr_model_path: str = "PaddlePaddle/PaddleOCR-VL", # 0.9B
    desc_model_name: str = "naver-hyperclovax/HyperCLOVAX-SEED-Vision-Instruct-3B"
):
    """
    Run OCR (PaddleOCR-VL) and Image Description (HyperCLOVAX-SEED-Vision-Instruct)
    on the same image and return a unified JSON-style result.

    Args:
        image_path (str): Path to input image.
        device_ocr (str): Device for PaddleOCR-VL ('cuda' or 'cpu').
        device_desc (str): Device for HyperCLOVAX ('cuda' or 'cpu').
        ocr_device_id (str): GPU device ID for OCR model.
        desc_device_id (str): GPU device ID for description model.
        ocr_model_path (str): HuggingFace model repo or local path for PaddleOCR-VL.
        desc_model_name (str): HuggingFace model name or path for HyperCLOVAX.

    Returns:
        dict: {
            "extracted_text": "Extracted text or 'NO'",
            "image_description": "Visual description in {language} or 'NO'",
            "elapsed_time": {
                "ocr": seconds,
                "description": seconds,
                "total": seconds
            }
        }
    """

    # -------------------------------------------------------------------------
    # 1️. OCR
    # -------------------------------------------------------------------------
    os.environ["CUDA_VISIBLE_DEVICES"] = ocr_device_id
    DEVICE_OCR = "cuda" if torch.cuda.is_available() else "cpu"

    PROMPTS = {
    "ocr": "OCR:",
    "table": "Table Recognition:",
    "formula": "Formula Recognition:",
    "chart": "Chart Recognition:",
    }

    image = Image.open(image_path).convert("RGB")

    start_ocr = time.time()

    ocr_model = AutoModelForCausalLM.from_pretrained(
        ocr_model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    ).to(DEVICE_OCR).eval()

    ocr_processor = AutoProcessor.from_pretrained(ocr_model_path, trust_remote_code=True, use_fast=True)

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

    ocr_outputs = ocr_model.generate(**ocr_inputs, max_new_tokens=1024)
    extracted_text = ocr_processor.batch_decode(ocr_outputs, skip_special_tokens=True)[0]
    #print('-----------------------')
    #print(extracted_text)
    #print('-----------------------')
    
    # Preprocess
    extracted_text = extracted_text.replace('User: OCR:\nAssistant: ','').replace('\n',' ')
    if len(extracted_text) == 1:
        extracted_text = extracted_text.replace('1','').replace('0','')
        
    ocr_elapsed = time.time() - start_ocr
        # Extracted text language portion extraction
    if len(extracted_text) < 1:
        extracted_text_portion = 'English'
    else:    
        extracted_text_portion = extracted_text[:10]
    # -------------------------------------------------------------------------
    # 2️. Image Description
    # -------------------------------------------------------------------------
    os.environ["CUDA_VISIBLE_DEVICES"] = desc_device_id
    DEVICE_DESC = "cuda" if torch.cuda.is_available() else "cpu"


    desc_model = AutoModelForCausalLM.from_pretrained(desc_model_name, trust_remote_code=True,use_fast=True).to(DEVICE_DESC)
    desc_processor = AutoProcessor.from_pretrained(desc_model_name, trust_remote_code=True)
    desc_tokenizer = AutoTokenizer.from_pretrained(desc_model_name)

    system_prompt = "You are a helpful image analyzer"
    user_prompt = f"""
Find images in the file and provide the following:
Describe the visual content in the same language as the extracted text below.
A portion of extracted text: {extracted_text_portion}
If there is no image (only text), output: NO

**Important notes**:
**Keep it concise and do not repeat**
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
        max_new_tokens=1500,
        do_sample=True,
        top_p=0.6,
        temperature=0.5,
        repetition_penalty=1.0,
    )

    image_description = desc_processor.batch_decode(output_ids, skip_special_tokens=True)[0]
    image_description = (image_description
    .replace("<|im_end|><|endofturn|>", "")
    .replace("**이미지 설명**:", "")
    .replace('\n', ' '))

    # -------------------------------------------------------------------------
    # 3️.  Final JSON result
    # -------------------------------------------------------------------------
    result = {
        "extracted_text": extracted_text.strip() if extracted_text and extracted_text.strip() else "NO",
        "image_description": image_description.strip() if image_description and image_description.strip() else "NO",
    }

    return result


# Main
if __name__ == "__main__":
    
    os.environ["CUDA_VISIBLE_DEVICES"] = "3"
    file_name = "테이블"
    image_path = f"/home/ljm/ocr/picture/{file_name}.png"
    output_path = f"/home/ljm/ocr/result/{file_name}_result.json"
    start_time = time.time()
    result = analyze_image(
        image_path=image_path,
        ocr_device_id="0",
        desc_device_id="0")
    total_time = time.time() - start_time
    print('time:',total_time)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
