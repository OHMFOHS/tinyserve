"""OpenAI-compatible HTTP server with SSE streaming.

The engine runs in a dedicated thread (GPU work never blocks the event loop). Requests enter through a thread-safe
queue; per-request token updates go back to asyncio queues via loop.call_soon_threadsafe.

    python -m tinyserve.server --model Qwen/Qwen3-0.6B --port 8000
"""

import argparse
import asyncio
import json
import logging
import queue
import threading
import time
import uuid
from dataclasses import dataclass

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse
from pydantic import BaseModel

from tinyserve.config import EngineConfig
from tinyserve.engine import LLMEngine
from tinyserve.sampling_params import SamplingParams

log = logging.getLogger("tinyserve")


@dataclass
class _Update:
    token_ids: list[int]
    finished: bool
    finish_reason: str | None
    num_prompt_tokens: int
    num_cached_tokens: int


class AsyncEngine:
    def __init__(self, engine: LLMEngine):
        self.engine = engine
        self.inbox: queue.Queue = queue.Queue()
        self.streams: dict[str, tuple[asyncio.AbstractEventLoop, asyncio.Queue]] = {}
        self.lock = threading.Lock()
        self.started = time.time()
        self.thread = threading.Thread(target=self._loop, daemon=True, name="engine")
        self.thread.start()

    def _loop(self):
        while True:
            # Block only when idle; otherwise drain new arrivals between steps (continuous batching).
            if not self.engine.has_unfinished():
                self._handle(self.inbox.get())
            while True:
                try:
                    self._handle(self.inbox.get_nowait())
                except queue.Empty:
                    break
            if not self.engine.has_unfinished():
                continue
            try:
                updated = self.engine.step()
            except Exception:
                log.exception("engine step failed")
                raise
            for seq in updated:
                self._emit(seq.request_id, _Update(seq.completion_token_ids, seq.is_finished, seq.finish_reason,
                                                   seq.num_prompt_tokens, seq.num_cached_tokens))

    def _handle(self, msg):
        kind, rid, payload = msg
        if kind == "add":
            prompt, params = payload
            try:
                self.engine.add_request(prompt, params, request_id=rid)
            except ValueError as e:
                self._emit(rid, e)
        elif kind == "abort":
            self.engine.abort(rid)

    def _emit(self, rid, item):
        with self.lock:
            target = self.streams.get(rid)
        if target:
            loop, q = target
            loop.call_soon_threadsafe(q.put_nowait, item)

    async def generate(self, prompt, params: SamplingParams):
        rid = uuid.uuid4().hex
        q: asyncio.Queue = asyncio.Queue()
        with self.lock:
            self.streams[rid] = (asyncio.get_running_loop(), q)
        self.inbox.put(("add", rid, (prompt, params)))
        try:
            while True:
                item = await q.get()
                if isinstance(item, Exception):
                    raise item
                yield item
                if item.finished:
                    return
        finally:
            with self.lock:
                self.streams.pop(rid, None)
            self.inbox.put(("abort", rid, None))  # no-op if already finished; frees KV if the client went away


class CompletionRequest(BaseModel):
    model: str | None = None
    prompt: str | list[int]
    max_tokens: int = 256
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = -1
    stream: bool = False
    ignore_eos: bool = False


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model: str | None = None
    messages: list[ChatMessage]
    max_tokens: int = 256
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = -1
    stream: bool = False
    ignore_eos: bool = False


class IncrementalDetokenizer:
    def __init__(self, tokenizer):
        self.tok = tokenizer
        self.sent = ""

    def delta(self, token_ids: list[int]) -> str:
        text = self.tok.decode(token_ids, skip_special_tokens=True)
        if text.endswith("�"):  # incomplete multi-byte character; wait for the next token
            return ""
        out, self.sent = text[len(self.sent):], text
        return out


def build_app(aengine: AsyncEngine, model_name: str) -> FastAPI:
    app = FastAPI(title="tinyserve")
    tokenizer = aengine.engine.tokenizer

    def params_of(r) -> SamplingParams:
        try:
            return SamplingParams(temperature=r.temperature, top_p=r.top_p, top_k=r.top_k, max_tokens=r.max_tokens,
                                  ignore_eos=r.ignore_eos)
        except ValueError as e:
            raise HTTPException(400, str(e))

    async def run(prompt, params, stream: bool, chat: bool):
        rid = ("chatcmpl-" if chat else "cmpl-") + uuid.uuid4().hex[:24]
        created = int(time.time())
        obj = "chat.completion" if chat else "text_completion"

        def chunk(text, finish_reason):
            if chat:
                choice = {"index": 0, "delta": {"content": text} if text else {}, "finish_reason": finish_reason}
                return {"id": rid, "object": "chat.completion.chunk", "created": created, "model": model_name,
                        "choices": [choice]}
            return {"id": rid, "object": obj, "created": created, "model": model_name,
                    "choices": [{"index": 0, "text": text, "finish_reason": finish_reason}]}

        if stream:
            async def sse():
                detok = IncrementalDetokenizer(tokenizer)
                try:
                    async for u in aengine.generate(prompt, params):
                        text = detok.delta(u.token_ids)
                        if text or u.finished:
                            yield f"data: {json.dumps(chunk(text, u.finish_reason if u.finished else None))}\n\n"
                except ValueError as e:
                    yield f"data: {json.dumps({'error': str(e)})}\n\n"
                yield "data: [DONE]\n\n"

            return StreamingResponse(sse(), media_type="text/event-stream")

        last = None
        try:
            async for u in aengine.generate(prompt, params):
                last = u
        except ValueError as e:
            raise HTTPException(400, str(e))
        text = tokenizer.decode(last.token_ids, skip_special_tokens=True)
        usage = {"prompt_tokens": last.num_prompt_tokens, "completion_tokens": len(last.token_ids),
                 "total_tokens": last.num_prompt_tokens + len(last.token_ids),
                 "prompt_tokens_details": {"cached_tokens": last.num_cached_tokens}}
        if chat:
            choice = {"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": last.finish_reason}
        else:
            choice = {"index": 0, "text": text, "finish_reason": last.finish_reason}
        return JSONResponse({"id": rid, "object": obj, "created": created, "model": model_name, "choices": [choice],
                             "usage": usage})

    @app.post("/v1/completions")
    async def completions(req: CompletionRequest):
        return await run(req.prompt, params_of(req), req.stream, chat=False)

    @app.post("/v1/chat/completions")
    async def chat_completions(req: ChatRequest):
        msgs = [m.model_dump() for m in req.messages]
        prompt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        return await run(prompt, params_of(req), req.stream, chat=True)

    @app.get("/v1/models")
    async def models():
        return {"object": "list", "data": [{"id": model_name, "object": "model", "owned_by": "tinyserve"}]}

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.get("/metrics")
    async def metrics(_: Request):
        e = aengine.engine
        s, bm = e.stats, e.block_manager
        lines = {
            "tinyserve_requests_running": len(e.scheduler.running),
            "tinyserve_requests_waiting": len(e.scheduler.waiting),
            "tinyserve_kv_cache_usage_ratio": round(bm.usage, 4),
            "tinyserve_kv_cache_blocks_total": bm.num_blocks,
            "tinyserve_prefix_cache_hit_rate": round(bm.prefix_hit_rate, 4),
            "tinyserve_preemptions_total": e.scheduler.num_preemptions,
            "tinyserve_prompt_tokens_total": s.prompt_tokens,
            "tinyserve_generation_tokens_total": s.generated_tokens,
            "tinyserve_engine_steps_total": s.steps,
        }
        return PlainTextResponse("".join(f"{k} {v}\n" for k, v in lines.items()))

    return app


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--max-num-seqs", type=int, default=128)
    ap.add_argument("--max-num-batched-tokens", type=int, default=8192)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    ap.add_argument("--no-prefix-caching", action="store_true")
    ap.add_argument("--enforce-eager", action="store_true")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    engine = LLMEngine(EngineConfig(
        model=a.model, max_model_len=a.max_model_len, max_num_seqs=a.max_num_seqs,
        max_num_batched_tokens=a.max_num_batched_tokens, gpu_memory_utilization=a.gpu_memory_utilization,
        enable_prefix_caching=not a.no_prefix_caching, enforce_eager=a.enforce_eager,
    ))
    app = build_app(AsyncEngine(engine), a.model)
    uvicorn.run(app, host=a.host, port=a.port, log_level="warning")


if __name__ == "__main__":
    main()
