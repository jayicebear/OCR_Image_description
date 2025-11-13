from paddleocr import PaddleOCR
import numpy as np 
from PIL import Image
from main import MyPaddleOCR

ocr = MyPaddleOCR()

file_name = 'safe'
image_path = f"/home/ljm/ocr/picture/{file_name}.png"

ocr.run_ocr(image_path, debug=True)