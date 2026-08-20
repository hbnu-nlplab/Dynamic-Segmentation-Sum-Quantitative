FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime

# 한국어 폰트 및 기본 패키지
RUN apt-get update && apt-get install -y \
    fonts-nanum \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Python 의존성 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 소스 코드 복사
COPY segmenter.py topic_matcher.py seg_summarizer.py seg_topic_generator.py sub_summarizer.py total_summarizer.py speaker_summarizer.py kormo_backend.py pipeline.py main.py ./

# Qwen 모델 복사 (로컬 캐시 구조 그대로)
COPY models--Qwen--Qwen3-Embedding-8B/ ./models--Qwen--Qwen3-Embedding-8B/

# Kormo 모델 복사 (multitask LoRA가 base 모델에 병합된 standalone 모델, 약 21GB)
COPY kormo_multitask_merged/ ./kormo_multitask_merged/

ENV MODEL_NAME=/app/models--Qwen--Qwen3-Embedding-8B/snapshots/1d8ad4ca9b3dd8059ad90a75d4983776a23d44af
ENV KORMO_MODEL_DIR=/app/kormo_multitask_merged
ENV OPENAI_MODEL=gpt-4o-mini
# /summarize_dir이 접근을 허용하는 루트. 실제 데이터셋 디렉토리는
# `docker run -v <호스트 경로>:/app/data`로 이 경로에 마운트해서 사용한다.
ENV DATA_DIR=/app/data
RUN mkdir -p /app/data

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
