"""本地演示上游（固定上游的模拟实现）。

能力：
- 公告内容按 Accept-Language 返回中/英双语；
- 每个公告可切换 public / personal 响应：
  * public  -> Cache-Control: public, max-age=N（可缓存）
  * personal -> Set-Cookie + Cache-Control: private, no-store（不可缓存）；
    同时下发 X-Session-Preview 回显会话值（Set-Cookie 对浏览器 JS 不可见，
    该头仅用于演示页展示"每个访客拿到各自的 SESSION"）；
- 失败模式：none（正常）/ error500（全部 500）/ intermittent（间歇失败，
  每 fail_every 个请求失败 1 次，用于演示拆分后独立取数的部分失败）；
- 可配置人工延迟与 public 的 max-age；
- 管理端可查看请求计数与最近请求记录，便于验证缓存与合并行为。
"""

from __future__ import annotations

import asyncio
import time
import uuid
from collections import deque
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

app = FastAPI(title="Mock Upstream", version="1.0.0")

NOTICES: dict[str, dict[str, dict[str, str]]] = {
    "1": {
        "zh": {
            "title": "关于年度系统维护的通知",
            "body": "本周六 02:00-04:00 门户系统例行维护，期间暂停服务。",
        },
        "en": {
            "title": "Annual System Maintenance Notice",
            "body": "Routine maintenance this Saturday 02:00-04:00; service paused.",
        },
    },
    "2": {
        "zh": {
            "title": "新版内网门户上线公告",
            "body": "新版门户已上线，支持统一搜索与个性化工作台。",
        },
        "en": {
            "title": "New Intranet Portal Launch",
            "body": "The new portal is live with unified search and a personal dashboard.",
        },
    },
    "3": {
        "zh": {
            "title": "信息安全培训报名通知",
            "body": "请各部门员工于月底前完成信息安全培训报名。",
        },
        "en": {
            "title": "Security Training Enrollment",
            "body": "All staff should enroll in the security training before month-end.",
        },
    },
}

DEFAULT_CONFIG = {
    "profile": "public",        # public | personal
    "latency_ms": 200,
    "failure": "none",          # none | error500 | intermittent
    "fail_every": 2,            # intermittent：每 N 个请求失败 1 次
    "max_age": 30,
    "cache_control": None,      # 非空时作为原始 Cache-Control 直接下发
}

# 全局默认 + 按公告覆盖
state: dict[str, Any] = {
    "defaults": dict(DEFAULT_CONFIG),
    "overrides": {},
    "content_version": 1,
    "started_at": time.time(),
    # intermittent 模式的独立计数器：配置/重置时清零，
    # 保证"配置后第 N 个请求失败"的演示与验证行为可复现
    "intermittent_count": 0,
}
counters = {"total_requests": 0, "by_notice": {}, "by_profile": {}}
request_log: deque[dict[str, Any]] = deque(maxlen=50)
_log_lock = asyncio.Lock()


def _cfg_for(notice_id: str) -> dict[str, Any]:
    cfg = dict(state["defaults"])
    cfg.update(state["overrides"].get(notice_id, {}))
    return cfg


def _pick_language(accept_language: str) -> str:
    if not accept_language:
        return "zh"
    for part in accept_language.split(","):
        tag = part.split(";", 1)[0].strip().lower()
        if tag.startswith("zh"):
            return "zh"
        if tag.startswith("en"):
            return "en"
    return "zh"


def _bump_version() -> None:
    state["content_version"] += 1


@app.get("/healthz")
async def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/notices/{notice_id}")
async def get_notice(notice_id: str, request: Request) -> JSONResponse:
    cfg = _cfg_for(notice_id)
    request_id = uuid.uuid4().hex[:12]
    started = time.perf_counter()

    if cfg["latency_ms"]:
        await asyncio.sleep(cfg["latency_ms"] / 1000.0)

    cookie = request.headers.get("cookie", "")
    auth = request.headers.get("authorization", "")
    lang = _pick_language(request.headers.get("accept-language", ""))

    log_entry = {
        "request_id": request_id,
        "notice_id": notice_id,
        "language": lang,
        "accept_language_raw": request.headers.get("accept-language", ""),
        "has_cookie": bool(cookie.strip()),
        "has_authorization": bool(auth.strip()),
        "profile": cfg["profile"],
        "failure": cfg["failure"],
        "latency_ms": cfg["latency_ms"],
        "elapsed_ms": round((time.perf_counter() - started) * 1000, 1),
        "ts": round(time.time(), 3),
    }
    counters["total_requests"] += 1
    counters["by_notice"][notice_id] = counters["by_notice"].get(notice_id, 0) + 1
    counters["by_profile"][cfg["profile"]] = (
        counters["by_profile"].get(cfg["profile"], 0) + 1
    )
    async with _log_lock:
        request_log.appendleft(log_entry)

    base_headers = {
        "X-Upstream-Request-Id": request_id,
        "X-Notice-Id": notice_id,
        "X-Notice-Kind": cfg["profile"],
    }

    if cfg["failure"] == "error500":
        return JSONResponse(
            status_code=500,
            content={"error": "simulated upstream failure", "request_id": request_id},
            headers=base_headers,
        )

    if cfg["failure"] == "intermittent":
        # 每 fail_every 个请求失败 1 次（独立计数器，配置时清零）
        state["intermittent_count"] += 1
        every = max(1, int(cfg.get("fail_every") or 1))
        if state["intermittent_count"] % every == 0:
            return JSONResponse(
                status_code=500,
                content={
                    "error": "simulated intermittent failure",
                    "request_id": request_id,
                    "intermittent_seq": state["intermittent_count"],
                },
                headers=base_headers,
            )

    if notice_id not in NOTICES:
        return JSONResponse(
            status_code=404,
            content={"error": f"notice {notice_id} not found", "request_id": request_id},
            headers=base_headers,
        )

    content = NOTICES[notice_id][lang]
    payload = {
        "notice_id": notice_id,
        "language": lang,
        "title": content["title"],
        "body": content["body"],
        "profile": cfg["profile"],
        "content_version": state["content_version"],
        "served_by": "mock-upstream",
        "request_id": request_id,
        "identity_seen": {
            "cookie": cookie or None,
            "authorization": auth or None,
        },
    }

    headers = dict(base_headers)
    headers["Content-Language"] = lang
    if cfg["cache_control"] is not None:
        headers["Cache-Control"] = str(cfg["cache_control"])
    elif cfg["profile"] == "public":
        headers["Cache-Control"] = f"public, max-age={int(cfg['max_age'])}"
    else:
        # 个人页面：显式私有、禁止缓存，并下发 Set-Cookie
        headers["Cache-Control"] = "private, no-store"
        session_value = f"SESSION=user-{request_id}"
        headers["Set-Cookie"] = f"{session_value}; Path=/; HttpOnly; SameSite=Lax"
        # Set-Cookie 对浏览器 JS 不可见；该回显头仅供演示页展示
        # "每个访客拿到各自的 SESSION"，代理白名单透传，不进入缓存。
        headers["X-Session-Preview"] = session_value

    return JSONResponse(status_code=200, content=payload, headers=headers)


# ---------------- 管理接口 ----------------


@app.get("/admin/state")
async def admin_state() -> dict[str, Any]:
    return {
        "defaults": state["defaults"],
        "overrides": state["overrides"],
        "content_version": state["content_version"],
        "counters": counters,
        "intermittent_count": state["intermittent_count"],
        "uptime_seconds": round(time.time() - state["started_at"], 1),
    }


@app.get("/admin/requests")
async def admin_requests() -> dict[str, Any]:
    async with _log_lock:
        return {"requests": list(request_log)}


@app.post("/admin/config")
async def admin_config(payload: dict[str, Any]) -> dict[str, Any]:
    notice_id = payload.get("notice_id")
    fields = {
        k: payload[k]
        for k in (
            "profile",
            "latency_ms",
            "failure",
            "fail_every",
            "max_age",
            "cache_control",
        )
        if k in payload
    }
    if "profile" in fields and fields["profile"] not in ("public", "personal"):
        return JSONResponse(
            status_code=400, content={"error": "profile must be public|personal"}
        )
    if "failure" in fields and fields["failure"] not in (
        "none",
        "error500",
        "intermittent",
    ):
        return JSONResponse(
            status_code=400,
            content={"error": "failure must be none|error500|intermittent"},
        )
    if "fail_every" in fields:
        try:
            fields["fail_every"] = max(1, int(fields["fail_every"]))
        except (TypeError, ValueError):
            return JSONResponse(
                status_code=400, content={"error": "fail_every must be an int >= 1"}
            )
    if notice_id:
        target = state["overrides"].setdefault(str(notice_id), {})
        target.update(fields)
    else:
        state["defaults"].update(fields)
    # 失败模式相关配置变更后，间歇计数清零，保证演示行为可复现
    state["intermittent_count"] = 0
    _bump_version()
    return {"ok": True, "state": await admin_state()}


@app.post("/admin/reset")
async def admin_reset() -> dict[str, Any]:
    state["defaults"] = dict(DEFAULT_CONFIG)
    state["overrides"] = {}
    state["intermittent_count"] = 0
    _bump_version()
    counters["total_requests"] = 0
    counters["by_notice"] = {}
    counters["by_profile"] = {}
    async with _log_lock:
        request_log.clear()
    return {"ok": True, "state": await admin_state()}
