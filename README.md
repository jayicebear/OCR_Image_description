## OCR + Image Description 파이프라인 (이미지 기반 PDF 전용)

PDF가 **스캔본/이미지 통짜**인 경우를 위해 **고속·고정확도 OCR + 언어 감지 기반 이미지 설명** 파이프라인을 추가.  
최종 결과는 MongoDB에 바로 저장되어 RAG에서 텍스트+시각 정보까지 활용 가능.

### 방식 비교
| 방식                              | 속도 (A4 1장) | OCR 정확도 | Description 품질 | 메모리 | 비고 |
|----------------------------------|---------------|------------|------------------|--------|------|
| Qwen2-VL-8B / Llama-3.2-11B-Vision | 4.2~6.8초     | 중     | 중상             | 22GB+  | 느리고 비쌈 |
| **PaddleOCR-VL + HyperCLOVAX-SEED 3B** | **0.84초**    | **상**  | **상**           | **6.2GB** | **5배 빠르고 3.5배 가볍** |

### 파이프라인 흐름도

```mermaid
graph TD
    A[PDF 이미지 페이지] --> B[PaddleOCR-VL]
    B --> C{추출 텍스트<br/>앞 10글자}
    C --> D[HyperCLOVAX-SEED-Vision-Instruct-3B<br/>언어 자동 감지 → 맞춤 설명]
    D --> E[JSON 출력]
    E --> F[MongoDB Collection<br/>pages_vision]
