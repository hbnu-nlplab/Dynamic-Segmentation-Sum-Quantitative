# Meeting Summary API
- **wo_st (topic 정보 없이)**: 세그먼트 단위로 요약/토픽/화자요약을 만들고, 전체 요약을 생성
- **w_st (sub topic 정보 포함)**: 입력으로 받은 sub_topic 목록에 세그먼트 매칭 후, 토픽 단위로 요약/화자요약을 만들고, total_topic 요약 생성
- segmentation은 항상 **Qwen3-Embedding-8B** 사용, 생성 모델은 GPT/Kormo 선택 사용


## 1. GPU 참고 사항
GPU 여러장에 나눠 배치 권장 (`QWEN_DEVICE_MAP`/`KORMO_DEVICE_MAP)
> 실측 GPU 메모리 사용량 (Qwen을 `cuda:1`, Kormo를 `cuda:2`에 배치한 경우):
> - GPU 1 (Qwen): 약 21GB
> - GPU 2 (Kormo): 약 10GB


## 2. 디렉토리 구조

```
.
├── Dockerfile
├── requirements.txt
├── .env                          # OPENAI_API_KEY
├── .dockerignore
│
├── dataset_new_schema/           # 입력 JSON 스키마 예시
├── main.py                       # FastAPI 앱, 라우팅, 입력 검증, 전역 예외 처리
├── pipeline.py                   # 전체 파이프라인 오케스트레이션
├── segmenter.py                  # 세그먼트 분할
├── topic_matcher.py              # 세그먼트 ↔ sub_topic 유사도 매칭
├── seg_summarizer.py             # 세그먼트 요약
├── seg_topic_generator.py        # 세그먼트 토픽 생성
├── sub_summarizer.py             # 서브토픽 요약
├── total_summarizer.py           # 전체 요약
├── speaker_summarizer.py         # 화자별 요약
├── kormo_backend.py              # Kormo 로컬 추론 백엔드
│
├── models--Qwen--Qwen3-Embedding-8B/   # Qwen 모델 (models--Qwen--Qwen3-Embedding-8B.tar.gz으로 압축되어 있음)
└── kormo_multitask_merged/             # Kormo 모델 (kormo_multitask_merged.tar.gz으로 압축되어 있음)
```


## 3. Openai API 설정

`.env` 파일(또는 `docker run --env-file`)로 주입
```dotenv
#.env file 예시
OPENAI_API_KEY=sk-proj-dxxxxxxxxxxxREA
```



## 4. Docker 빌드

### 4-1. 이미지 빌드
```bash
cd Dynamic-Segmentation-Sum_docker
docker build -t meeting-summary-api:v1 .
```

빌드 완료 후 확인:
  ```bash
  docker images | grep meeting-summary-api
  ```

---

### 4-2. docker 띄우기

```bash
docker run -d \
  --name meeting-summary-api \
  --gpus all \
  -p 8000:8000 \
  --env-file .env \
  -e QWEN_DEVICE_MAP=cuda:0 \
  -e KORMO_DEVICE_MAP=cuda:1 \
  meeting-summary-api:v1
```
- GPU 인덱스는 컨테이너 안에서 `--gpus all`로 다 보이는 순서 기준
- 특정 GPU만 컨테이너에 노출하고 싶으면 `--gpus '"device=0,1"'` 형태로 제한 가능
- QWEN_DEVICE_MAP과 KORMO_DEVICE_MAP이 cuda:0, 1로 지정되어있는건 예시이며 **비어있는 GPU 지정 필요**
- 디렉토리 일괄 처리(`/summarize_dir`)를 쓰려면 `-v /host/dataset/path:/app/data`로 데이터 디렉토리도 마운트 (5장 참고)
---
### 4-3. 동작 확인
```bash
docker logs -f meeting-summary-api
```
- Qwen과 Kormo 둘 다 **서버 시작 시점에 즉시 로딩** (지연 로딩 아님). 아래 로그가 순서대로 뜨고 나면 준비 완료 상태:
```
Qwen 임베딩 모델 로딩 중...
✅ separator token: <sep> (id=...)
Qwen 임베딩 모델 로딩 완료
Kormo 모델 로딩 중...
Kormo 로딩 중... (model_dir=/app/kormo_multitask_merged)
Kormo 로딩 완료
Kormo 모델 로딩 완료
INFO:     Uvicorn running on http://0.0.0.0:8000
```

```bash
curl http://localhost:8000/health
# {"status":"ok","model_loaded":true}
```

### 컨테이너 중지/삭제
```bash
docker stop meeting-summary-api && docker rm meeting-summary-api
```

---

## 5. API 레퍼런스
| 필드 |  설명 |
|---|---|
| `file` |  회의록 JSON 파일 |
| `type` |  `"wo_st"` 또는 `"w_st"` |
| `model` |   `"gpt"` 또는 `"kormo"`. `"kormo"`를 명시하면 처음부터 GPT를 거치지 않고 Kormo만 사용 |

#### 입력 JSON 스키마 — `type=wo_st`
```json
{
  "dialogue": [
    {"utterance_id": 0, "speaker": "A", "utterance": "발화 내용..."},
    {"utterance_id": 1, "speaker": "B", "utterance": "발화 내용..."}
  ]
}
```

#### 입력 JSON 스키마 — `type=w_st`
```json
{
  "dialogue": [
    {"utterance_id": 0, "speaker": "A", "utterance": "발화 내용..."},
    {"utterance_id": 1, "speaker": "B", "utterance": "발화 내용..."}
  ],
  "sub_topic": ["세부 주제 1", "세부 주제 2", "세부 주제 3"],
  "total_topic": "회의 전체 주제"
}
```
### **입력 JSON 예시는 dataset_new_schema 디렉토리 내에 있습니다.**


#### 호출 예시
```bash
curl -X POST http://localhost:8000/summarize \
  -F "file=@meeting.json" \
  -F "type=wo_st" \
  -o output.json
```
```bash
curl -X POST http://localhost:8000/summarize \
  -F "file=@meeting.json" \
  -F "type=w_st" \
  -F "model=kormo" \
  -o output.json
```

#### 디렉토리 일괄 처리 — `POST /summarize_dir`
파일 하나씩이 아니라, **서버 쪽 디렉토리 안의 모든 `.json` 파일을 한 번에** 처리하고 싶을 때 사용합니다.

| 필드 | 설명 |
|---|---|
| `dir_path` | `DATA_DIR` 기준 **상대경로**. 하위 디렉토리는 재귀적으로 훑지 않고, 지정한 디렉토리 바로 아래의 `.json`만 대상으로 함 |
| `type` / `model` | `/summarize`와 동일 |

`dir_path`는 서버 파일시스템 경로라서, 컨테이너로 띄울 때는 실제 데이터가 있는 호스트 디렉토리를 `DATA_DIR`(기본 `/app/data`)에 볼륨으로 마운트해야 합니다:
```bash
docker run -d \
  --name meeting-summary-api \
  --gpus all \
  -p 8000:8000 \
  --env-file .env \
  -v /host/dataset/path:/app/data \
  meeting-summary-api:v1
```

```bash
curl -X POST http://localhost:8000/summarize_dir \
  -F "dir_path=wo_st" \
  -F "type=wo_st" \
  -o batch_output.json
```
- `dir_path`로 `DATA_DIR` 바깥 경로(`../../etc`, 절대경로 등)를 주면 400으로 거부됩니다 (임의 파일 접근 방지).
- 파일 하나가 깨져있거나 처리 중 실패해도 전체 요청이 죽지 않고, 그 파일의 결과 자리에 `{"file": ..., "error": ...}`만 남긴 채 나머지는 계속 처리됩니다.
- 파일 수가 많으면 응답이 올 때까지 오래 걸릴 수 있습니다 (파일마다 순차 처리). curl에 타임아웃을 걸어뒀다면 넉넉하게 늘려야 합니다 (예: `--max-time 0`).

응답 형식:
```json
{
  "dir": "wo_st",
  "type": "wo_st",
  "model": "gpt",
  "total": 3,
  "succeeded": 2,
  "results": [
    { "file": "a.json", "...": "정상 처리된 /summarize와 동일한 응답" },
    { "file": "b.json", "error": "유효한 JSON이 아닙니다." }
  ]
}
```

#### 응답 예시 — `wo_st` (실제 응답 발췌)
```json
{
  "file": "meeting.json",
  "type": "wo_st",
  "model": "gpt",
  "total_topic": "AI와 ChatGPT가 업무와 사회에 미치는 영향",
  "total_summary": "이번 회의에서는 인공지능(AI)과 ChatGPT의 영향, 윤리적 문제, 그리고 직업의 변화에 대한 심도 깊은 논의가 이루어졌다...",
  "speaker_summaries": {
    "사회자1": "회의 전반에 걸쳐 참석자들의 발언을 이끌어내고 논의를 정리했다. (3~5문장)",
    "게스트1": "AI 연구소에서의 경험을 바탕으로 여러 구간에서 의견을 제시했다. (3~5문장)"
  },
  "topics": [
    {
      "topic": "AI와 ChatGPT의 업무 영향",
      "summary": "이번 회의에서는 팀벨의 여러 연구원들이 참석하여 각자의 역할과 현재 진행 중인 프로젝트에 대해 소개했다...",
      "speaker_summaries": {
        "사회자1": "회의를 시작하며 참석자들에게 자기 소개를 요청하고... (3문장)",
        "게스트1": "현재 팀벨 AI 연구소에서 음성 인식 연구를 하고 있다고 소개했습니다. (3문장)"
      }
    }
  ]
}
```
- `total_topic`은 모든 세그먼트의 `summary`와 `topic`을 입력으로 삼아 생성됩니다 (w_st처럼 사용자가 입력으로 주는 값이 아니라, wo_st는 자체적으로 생성).
- `topics[].speaker_summaries`는 세그먼트(토픽) 단위 화자 요약이며, **3문장**으로 생성됩니다.
- 최상위 `speaker_summaries`는 그 화자의 모든 세그먼트별 요약을 입력으로 삼아 생성한 **회의 전체 기준 화자 요약**이며, **3~5문장**으로 생성됩니다.

#### 응답 예시 — `w_st` (실제 응답 발췌)
```json
{
  "file": "meeting.json",
  "type": "w_st",
  "model": "gpt",
  "total_topic": "현대인의 불규칙한 식습관과 배달 문제",
  "total_summary": "최근 회의에서는 현대인의 불규칙한 식습관과 배달 문화에 대한 심도 있는 논의가 이루어졌다...",
  "topics": [
    {
      "topic": "현대인 식습관의 문제",
      "summary": "현대인의 식습관 문제에 대한 논의가 진행됐다...",
      "speaker_summaries": {
        "사회자1": "..."
      }
    }
  ]
}
```

#### 오류 응답
| 상황 | 상태코드 | 응답 예시 |
|---|---|---|
| `type`/`model` 값이 잘못됨, `.json`이 아닌 파일, JSON 파싱 실패, `dialogue`/`sub_topic` 누락 또는 필드 누락 | 400 | `{"detail": "dialogue[3]에 필수 필드가 없습니다: ['utterance']"}` |
| 예상치 못한 서버 내부 오류 (Kormo까지 실패하는 등, 더 이상 대체할 수단이 없는 경우) | 500 | `{"error": "internal_server_error", "detail": "서버 내부 오류가 발생했습니다. 서버 로그를 확인해주세요."}` |

500이 나면 클라이언트에는 상세 원인이 노출되지 않으니, `docker logs meeting-summary-api`로 서버 로그의 스택트레이스 확인 필요

---

## 6. 모델 폴백 동작 방식

- `model=gpt`(기본값)로 요청하면, **파이프라인 전체를 GPT로 먼저 시도**
- 처리 도중 **어디서든 예외가 발생하면** (OpenAI 인증/과금 문제, 네트워크 오류 등 종류 불문), 그때까지 GPT로 처리된 부분까지 전부 폐기하고 **처음부터 전체를 Kormo로 다시 실행** (부분적으로 GPT+Kormo 결과가 섞이지 않습니다.)
- `model=kormo`로 명시하면 처음부터 GPT를 거치지 않고 Kormo만 사용
- 응답의 `"model"` 필드로 실제 어떤 모델이 사용됐는지 확인 가능


---

## 7. 동시 요청 처리 방식

- 여러 요청이 동시에 들어와도 서버(`/health` 포함)는 항상 즉시 응답
- **GPT 호출 구간(네트워크 I/O)**: 여러 요청이 실제로 동시에 진행
- **GPU를 쓰는 구간**(Qwen 세그멘테이션, Kormo 생성): 한 번에 하나씩만 처리되도록 내부적으로 순서가 매겨짐 (GPU 자원 하나를 여러 요청이 동시에 건드리면 안 되기 때문) 여러 요청이 몰리면 이 구간에서 대기 시간이 늘어날 수 있음
- 동시 요청 개수 자체에 대한 명시적 상한은 없음 (요청이 아주 많이 몰리면 대기열이 길어질 수 있음)

---


## 8. 트러블슈팅

| 증상 | 확인할 것 |
|---|---|
| `docker build` 중 디스크가 꽉 참 | `df -h /`로 여유공간 확인. 100GB 이상 확보 후 `docker builder prune -af`로 캐시 정리 후 재시도 |
| 컨테이너가 뜨자마자 죽음 / `docker ps`에 안 보임 | `docker logs meeting-summary-api`로 원인 확인. GPU 인식 문제(`--gpus all` 누락, NVIDIA Container Toolkit 설치 여부 확인) |
| `curl: (7) Failed to connect` | 컨테이너가 실제로 떠 있는지(`docker ps`), 포트 매핑(`-p 8000:8000`)이 됐는지 확인 |
| 모델 로딩이 오래 걸리거나 멈춘 것처럼 보임 | Qwen+Kormo 둘 다 기동 시 로딩하므로 정상적으로 수 분 걸릴 수 있음. `docker logs -f`로 진행상황 확인 |
| CUDA out of memory | `nvidia-smi`로 다른 프로세스가 GPU를 점유하고 있는지 확인. Qwen+Kormo 합쳐 최소 32GB 필요 (2장 참고). GPU 여러 장이면 `QWEN_DEVICE_MAP`/`KORMO_DEVICE_MAP`으로 분리 배치 |
| 한글 파일명 포함된 파일을 curl로 업로드했는데 `curl: (7)` 또는 `(26)` 에러 | 셸에 직접 타이핑하지 말고, `ls`나 glob(`file=@dataset/*.json`)으로 실제 파일명을 그대로 가져와서 사용 (유니코드 정규화 방식 차이로 육안상 같아 보여도 바이트가 다를 수 있음) |
