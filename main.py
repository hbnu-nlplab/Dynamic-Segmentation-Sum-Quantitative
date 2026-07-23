import os
import json
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool

from pipeline import SummaryPipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("meeting_summary_api")

_script_dir = os.path.dirname(os.path.abspath(__file__))
MODEL_NAME = os.getenv(
    "MODEL_NAME",
    os.path.join(_script_dir, "models--Qwen--Qwen3-Embedding-8B/snapshots/1d8ad4ca9b3dd8059ad90a75d4983776a23d44af"),
)
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
KORMO_MODEL_DIR = os.getenv(
    "KORMO_MODEL_DIR",
    os.path.join(_script_dir, "kormo_multitask_merged"),
)

state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    state["pipeline"] = SummaryPipeline(MODEL_NAME, OPENAI_MODEL, kormo_model_dir=KORMO_MODEL_DIR)
    yield
    state.clear()


app = FastAPI(title="Meeting Summary API", lifespan=lifespan)


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    logger.error("처리되지 않은 예외 발생: %s %s", request.method, request.url.path, exc_info=exc)
    return JSONResponse(
        status_code=500,
        content={
            "error": "internal_server_error",
            "detail": "서버 내부 오류가 발생했습니다. 서버 로그를 확인해주세요.",
        },
    )


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": "pipeline" in state}


def _validate_dialogue(dialogue):
    for i, ut in enumerate(dialogue):
        missing = [k for k in ("utterance", "speaker") if k not in ut]
        if missing:
            raise HTTPException(
                status_code=400,
                detail=f"dialogue[{i}]에 필수 필드가 없습니다: {missing}",
            )


@app.post("/summarize")
async def summarize(
    file: UploadFile = File(...),
    type: str = Form(...),
    model: str = Form("gpt"),
):
    if type not in ("w_st", "wo_st"):
        raise HTTPException(status_code=400, detail="type은 'w_st' 또는 'wo_st' 여야 합니다.")

    if model not in ("gpt", "kormo"):
        raise HTTPException(status_code=400, detail="model은 'gpt' 또는 'kormo' 여야 합니다.")

    if not file.filename.endswith(".json"):
        raise HTTPException(status_code=400, detail="json 파일만 업로드 가능합니다.")

    raw = await file.read()
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="유효한 JSON이 아닙니다.")

    dialogue = data.get("dialogue")
    if not dialogue:
        raise HTTPException(status_code=400, detail="'dialogue' 필드가 없습니다.")
    _validate_dialogue(dialogue)

    if type == "w_st":
        sub_topics = [t for t in (data.get("sub_topic") or []) if t]
        if not sub_topics:
            raise HTTPException(
                status_code=400,
                detail="type=w_st 이지만 'sub_topic' 필드가 없거나 비어 있습니다.",
            )

        total_topic = data.get("total_topic", "")

        logger.info(
            "요청 시작: file=%s type=w_st model=%s utterances=%d topics=%d",
            file.filename, model, len(dialogue), len(sub_topics),
        )
        result = await run_in_threadpool(
            state["pipeline"].run,
            file.filename, dialogue, "w_st",
            sub_topics=sub_topics, total_topic=total_topic, model=model,
        )
    else:
        logger.info(
            "요청 시작: file=%s type=wo_st model=%s utterances=%d",
            file.filename, model, len(dialogue),
        )
        result = await run_in_threadpool(
            state["pipeline"].run, file.filename, dialogue, "wo_st", model=model,
        )

    logger.info("요청 완료: file=%s 최종 사용 모델=%s", file.filename, result.get("model"))
    return result
