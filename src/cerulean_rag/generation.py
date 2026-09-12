"""One LLM call per question: Ollama's json_schema mode, falling back to
format="json" with a strict parse and one corrective retry. If everything
fails the caller still gets a well-formed result, so the CLI never crashes on
model output.
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from functools import lru_cache

from langchain_core.messages import BaseMessage, HumanMessage
from langchain_ollama import ChatOllama
from pydantic import ValidationError

from cerulean_rag.config import Settings, get_settings
from cerulean_rag.models import AnswerSchema

log = logging.getLogger(__name__)

KEEP_ALIVE = "30m"
FAILURE_ANSWER = "The assistant could not produce a well-formed answer. Please retry or rephrase."


@dataclass
class GenerationResult:
    parsed: AnswerSchema
    raw: str = ""
    method: str = ""            # "json_schema", "json_mode", "json_mode_retry", "failed"
    attempts: int = 0
    elapsed_s: float = 0.0
    error: str | None = None
    warnings: list[str] = field(default_factory=list)
    token_stats: dict = field(default_factory=dict)   # eval_count, eval_duration_ns, prompt_eval_count, ...


def token_stats_from(msg) -> dict:
    """Ollama's token and timing counters; durations are nanoseconds."""
    md = getattr(msg, "response_metadata", None) or {}
    out: dict = {}
    for k in ("eval_count", "eval_duration", "prompt_eval_count", "prompt_eval_duration",
              "total_duration", "load_duration"):
        if md.get(k) is not None:
            out[k + ("_ns" if k.endswith("duration") else "")] = md[k]
    return out


def _llm_kwargs(s: Settings) -> dict:
    return dict(
        model=s.GEN_MODEL,
        base_url=s.OLLAMA_BASE_URL,
        temperature=s.TEMPERATURE,
        seed=s.SEED,
        num_ctx=s.NUM_CTX,
        num_predict=s.MAX_ANSWER_TOKENS,
        keep_alive=KEEP_ALIVE,
    )


@lru_cache(maxsize=4)
def _get_llm(model: str, base_url: str, temperature: float, seed: int, num_ctx: int,
             num_predict: int, json_mode: bool) -> ChatOllama:
    kwargs = dict(model=model, base_url=base_url, temperature=temperature, seed=seed,
                  num_ctx=num_ctx, num_predict=num_predict, keep_alive=KEEP_ALIVE)
    if json_mode:
        kwargs["format"] = "json"
    return ChatOllama(**kwargs)


def get_llm(settings: Settings | None = None, json_mode: bool = False) -> ChatOllama:
    s = settings or get_settings()
    return _get_llm(s.GEN_MODEL, s.OLLAMA_BASE_URL, s.TEMPERATURE, s.SEED, s.NUM_CTX,
                    s.MAX_ANSWER_TOKENS, json_mode)


def _extract_json_object(text: str) -> str:
    """First top-level {...}; models sometimes wrap it in prose or fences."""
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.M)
    start = text.find("{")
    if start < 0:
        raise ValueError("no JSON object in model output")
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start: i + 1]
    raise ValueError("unterminated JSON object in model output")


def parse_answer(raw: str) -> AnswerSchema:
    return AnswerSchema.model_validate_json(_extract_json_object(raw))


_LAST_STATS: dict = {}   # token counters of the most recent model call (read by generate())


def _structured_call(messages: list[BaseMessage], s: Settings) -> tuple[AnswerSchema | None, str, str | None]:
    """Returns (parsed or None, raw text, error)."""
    llm = get_llm(s).with_structured_output(AnswerSchema, method="json_schema", include_raw=True)
    out = llm.invoke(messages)
    raw_msg = out.get("raw")
    _LAST_STATS.clear()
    _LAST_STATS.update(token_stats_from(raw_msg))
    raw = raw_msg.content if raw_msg is not None and isinstance(raw_msg.content, str) else str(raw_msg)
    parsed = out.get("parsed")
    err = out.get("parsing_error")
    if isinstance(parsed, AnswerSchema):
        return parsed, raw, None
    # with_structured_output may hand back a dict when validation was skipped
    if isinstance(parsed, dict):
        try:
            return AnswerSchema.model_validate(parsed), raw, None
        except ValidationError as exc:
            return None, raw, str(exc)
    return None, raw, str(err) if err else "no parsed output"


def _json_mode_call(messages: list[BaseMessage], s: Settings) -> tuple[AnswerSchema | None, str, str | None]:
    llm = get_llm(s, json_mode=True)
    msg = llm.invoke(messages)
    _LAST_STATS.clear()
    _LAST_STATS.update(token_stats_from(msg))
    raw = msg.content if isinstance(msg.content, str) else json.dumps(msg.content)
    try:
        return parse_answer(raw), raw, None
    except (ValidationError, ValueError) as exc:
        return None, raw, str(exc)


def generate(messages: list[BaseMessage], settings: Settings | None = None) -> GenerationResult:
    result = _generate(messages, settings)
    result.token_stats = dict(_LAST_STATS)
    return result


def _generate(messages: list[BaseMessage], settings: Settings | None = None) -> GenerationResult:
    s = settings or get_settings()
    t0 = time.perf_counter()
    attempts = 0
    warnings: list[str] = []
    raw = ""
    error: str | None = None

    attempts += 1
    try:
        parsed, raw, error = _structured_call(messages, s)
        if parsed is not None:
            log.info("generation ok via json_schema in %.1fs", time.perf_counter() - t0)
            return GenerationResult(parsed, raw, "json_schema", attempts, time.perf_counter() - t0)
        log.warning("json_schema output failed to parse: %s", error)
    except Exception as exc:  # transport or schema rejection -> fall back
        error = f"{type(exc).__name__}: {exc}"
        log.warning("structured output path failed (%s); falling back to json mode", error)
    warnings.append(f"structured output failed: {str(error)[:160]}")

    attempts += 1
    try:
        parsed, raw, error = _json_mode_call(messages, s)
        if parsed is not None:
            log.info("generation ok via json_mode in %.1fs", time.perf_counter() - t0)
            return GenerationResult(parsed, raw, "json_mode", attempts, time.perf_counter() - t0, warnings=warnings)
        log.warning("json_mode output failed to parse: %s", error)
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        log.error("json mode call failed: %s", error)
        return GenerationResult(_failure(), raw, "failed", attempts, time.perf_counter() - t0, error,
                                warnings + [f"generation failed: {error[:160]}"])

    # retry once with the validation error appended, so the model can correct itself
    attempts += 1
    retry_messages = list(messages) + [
        HumanMessage(content=(
            "Your previous response was not a valid answer object. Validation error:\n"
            f"{str(error)[:800]}\n\nReturn ONLY the JSON object matching the schema, with no other text."
        ))
    ]
    try:
        parsed, raw, error = _json_mode_call(retry_messages, s)
        if parsed is not None:
            log.info("generation ok via json_mode retry in %.1fs", time.perf_counter() - t0)
            return GenerationResult(parsed, raw, "json_mode_retry", attempts, time.perf_counter() - t0,
                                    warnings=warnings)
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    log.error("generation failed after %d attempts: %s", attempts, error)
    return GenerationResult(_failure(), raw, "failed", attempts, time.perf_counter() - t0, error,
                            warnings + [f"generation failed after {attempts} attempts: {str(error)[:160]}"])


def _failure() -> AnswerSchema:
    return AnswerSchema(decision="insufficient_evidence", answer=FAILURE_ANSWER)
