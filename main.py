import os
import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path

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
# 디렉토리 일괄 처리(/summarize_dir)가 접근할 수 있는 루트. 임의의 서버 경로를 그대로 읽지 못하도록
# dir_path는 항상 이 디렉토리 기준 상대경로로만 받는다 (컨테이너에서는 -v로 이 경로에 볼륨을 마운트해서 사용).
DATA_DIR = Path(os.getenv("DATA_DIR", os.path.join(_script_dir, "data"))).resolve()

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


def _validate_type_model(type_, model):
    if type_ not in ("w_st", "wo_st"):
        raise HTTPException(status_code=400, detail="type은 'w_st' 또는 'wo_st' 여야 합니다.")
    if model not in ("gpt", "kormo"):
        raise HTTPException(status_code=400, detail="model은 'gpt' 또는 'kormo' 여야 합니다.")


async def _summarize_one(filename, data, type_, model):
    """파싱된 JSON(data) 하나를 요약 파이프라인에 태워 결과를 반환한다.
    /summarize(단일 업로드)와 /summarize_dir(디렉토리 일괄)가 공통으로 사용한다."""
    dialogue = data.get("dialogue")
    if not dialogue:
        raise HTTPException(status_code=400, detail="'dialogue' 필드가 없습니다.")
    _validate_dialogue(dialogue)

    if type_ == "w_st":
        sub_topics = [t for t in (data.get("sub_topic") or []) if t]
        if not sub_topics:
            raise HTTPException(
                status_code=400,
                detail="type=w_st 이지만 'sub_topic' 필드가 없거나 비어 있습니다.",
            )
        total_topic = data.get("total_topic", "")

        logger.info(
            "요청 시작: file=%s type=w_st model=%s utterances=%d topics=%d",
            filename, model, len(dialogue), len(sub_topics),
        )
        result = await run_in_threadpool(
            state["pipeline"].run,
            filename, dialogue, "w_st",
            sub_topics=sub_topics, total_topic=total_topic, model=model,
        )
    else:
        logger.info(
            "요청 시작: file=%s type=wo_st model=%s utterances=%d",
            filename, model, len(dialogue),
        )
        result = await run_in_threadpool(
            state["pipeline"].run, filename, dialogue, "wo_st", model=model,
        )

    logger.info("요청 완료: file=%s 최종 사용 모델=%s", filename, result.get("model"))
    return result


@app.post("/summarize")
async def summarize(
    file: UploadFile = File(...),
    type: str = Form(...),
    model: str = Form("gpt"),
):
    _validate_type_model(type, model)

    if not file.filename.endswith(".json"):
        raise HTTPException(status_code=400, detail="json 파일만 업로드 가능합니다.")

    raw = await file.read()
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="유효한 JSON이 아닙니다.")

    return await _summarize_one(file.filename, data, type, model)


@app.post("/summarize_dir")
async def summarize_dir(
    dir_path: str = Form(...),
    type: str = Form(...),
    model: str = Form("gpt"),
):
    """dir_path(DATA_DIR 기준 상대경로) 안의 모든 .json 파일을 순서대로 요약한다.
    파일 하나가 실패해도 나머지는 계속 처리하고, 그 파일 결과에만 'error'를 담는다."""
    _validate_type_model(type, model)

    target_dir = (DATA_DIR / dir_path).resolve()
    if DATA_DIR not in target_dir.parents and target_dir != DATA_DIR:
        raise HTTPException(status_code=400, detail="dir_path가 허용된 디렉토리를 벗어났습니다.")
    if not target_dir.is_dir():
        raise HTTPException(status_code=400, detail=f"디렉토리를 찾을 수 없습니다: {dir_path}")

    json_files = sorted(target_dir.glob("*.json"))
    if not json_files:
        raise HTTPException(status_code=400, detail=f"'{dir_path}'에 .json 파일이 없습니다.")

    logger.info(
        "배치 요청 시작: dir=%s type=%s model=%s 파일수=%d",
        dir_path, type, model, len(json_files),
    )

    results = []
    for json_path in json_files:
        filename = json_path.name
        try:
            data = json.loads(json_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            results.append({"file": filename, "error": "유효한 JSON이 아닙니다."})
            continue

        try:
            result = await _summarize_one(filename, data, type, model)
            results.append(result)
        except HTTPException as e:
            results.append({"file": filename, "error": e.detail})
        except Exception as e:
            logger.error("배치 처리 중 오류: file=%s | %s", filename, e, exc_info=True)
            results.append({"file": filename, "error": "서버 내부 오류가 발생했습니다. 서버 로그를 확인해주세요."})

    succeeded = sum(1 for r in results if "error" not in r)
    logger.info(
        "배치 요청 완료: dir=%s 총 %d개 중 성공 %d개",
        dir_path, len(json_files), succeeded,
    )

    return {
        "dir": dir_path,
        "type": type,
        "model": model,
        "total": len(json_files),
        "succeeded": succeeded,
        "results": results,
    }
