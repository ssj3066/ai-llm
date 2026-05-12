#!/usr/bin/env python3
"""Small LLM operations console for the 118 GPU server.

This service intentionally avoids third-party Python dependencies. It proxies
Ollama operations, exposes GPU/runtime health, and provides a simple internal
web UI for operators and future ERP/NMS integrations.
"""

from __future__ import annotations

import csv
import io
import json
import os
import socket
import sqlite3
import subprocess
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any


APP_DIR = Path(__file__).resolve().parent
ENV_FILE = APP_DIR / "llm-ops.env"


def load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


load_env_file(ENV_FILE)


HOST = os.getenv("LLM_OPS_HOST", "0.0.0.0")
PORT = int(os.getenv("LLM_OPS_PORT", "8090"))
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")
API_TOKEN = os.getenv("LLM_OPS_TOKEN", "")
DEFAULT_MODEL = os.getenv("LLM_OPS_DEFAULT_MODEL", "metro-report:latest")
FAST_MODEL = os.getenv("LLM_OPS_FAST_MODEL", "metro-fast:latest")
REQUEST_TIMEOUT = int(os.getenv("LLM_OPS_REQUEST_TIMEOUT_SECONDS", "600"))
MAX_BODY_BYTES = int(os.getenv("LLM_OPS_MAX_BODY_BYTES", str(4 * 1024 * 1024)))
KEEP_ALIVE = os.getenv("LLM_OPS_KEEP_ALIVE", "30m")
ALLOWED_ORIGIN = os.getenv("LLM_OPS_ALLOWED_ORIGIN", "*")
NMS_CONTEXT_BASE_URL = os.getenv("NMS_CONTEXT_BASE_URL", "http://192.168.1.33:7443").rstrip("/")
NMS_CONTEXT_TOKEN = os.getenv("NMS_CONTEXT_TOKEN", "")
NMS_CONTEXT_TIMEOUT = int(os.getenv("NMS_CONTEXT_TIMEOUT_SECONDS", "12"))
NMS_MONITOR_ENABLED = os.getenv("NMS_MONITOR_ENABLED", "true").strip().lower() not in {"0", "false", "no", "off"}
NMS_MONITOR_INTERVAL = int(os.getenv("NMS_MONITOR_INTERVAL_SECONDS", "60"))
NMS_MONITOR_WINDOW_HOURS = int(os.getenv("NMS_MONITOR_WINDOW_HOURS", "24"))
NMS_MONITOR_LIMIT = int(os.getenv("NMS_MONITOR_LIMIT", "500"))
NMS_DEEP_ANALYSIS_WINDOW_HOURS = int(os.getenv("NMS_DEEP_ANALYSIS_WINDOW_HOURS", "48"))
NMS_DEEP_ANALYSIS_LIMIT = int(os.getenv("NMS_DEEP_ANALYSIS_LIMIT", "80"))
NMS_ANALYSIS_CONTEXT_CHAR_LIMIT = int(os.getenv("NMS_ANALYSIS_CONTEXT_CHAR_LIMIT", "90000"))
NMS_ANALYSIS_TIMEOUT = int(os.getenv("NMS_ANALYSIS_TIMEOUT_SECONDS", "900"))
NMS_ANALYSIS_NUM_CTX = int(os.getenv("NMS_ANALYSIS_NUM_CTX", "16384"))
NMS_ANALYSIS_NUM_PREDICT = int(os.getenv("NMS_ANALYSIS_NUM_PREDICT", "3072"))
NMS_ANALYSIS_REPEAT_PENALTY = float(os.getenv("NMS_ANALYSIS_REPEAT_PENALTY", "1.08"))
NMS_AUTOPILOT_ENABLED = os.getenv("NMS_AUTOPILOT_ENABLED", "true").strip().lower() not in {"0", "false", "no", "off"}
NMS_AUTOPILOT_INTERVAL = int(os.getenv("NMS_AUTOPILOT_INTERVAL_SECONDS", "600"))
NMS_AUTOPILOT_TARGET_COOLDOWN = int(os.getenv("NMS_AUTOPILOT_TARGET_COOLDOWN_SECONDS", "1800"))
NMS_AUTOPILOT_MAX_TARGETS = int(os.getenv("NMS_AUTOPILOT_MAX_TARGETS", "1"))
NMS_AUTOPILOT_WINDOW_HOURS = int(os.getenv("NMS_AUTOPILOT_WINDOW_HOURS", "48"))
NMS_AUTOPILOT_CONTEXT_LIMIT = int(os.getenv("NMS_AUTOPILOT_CONTEXT_LIMIT", "40"))
NMS_AUTOPILOT_MODEL = os.getenv("NMS_AUTOPILOT_MODEL", DEFAULT_MODEL)
NMS_AUTOPILOT_CONVERSATION_PREFIX = os.getenv("NMS_AUTOPILOT_CONVERSATION_PREFIX", "nms-autopilot")
NMS_AUTOPILOT_MIN_EVENT_COUNT = int(os.getenv("NMS_AUTOPILOT_MIN_EVENT_COUNT", "1"))
NMS_AUTOPILOT_START_DELAY_SECONDS = int(os.getenv("NMS_AUTOPILOT_START_DELAY_SECONDS", "30"))
NMS_AUTOPILOT_TIMEOUT = int(os.getenv("NMS_AUTOPILOT_TIMEOUT_SECONDS", "300"))
NMS_AUTOPILOT_NUM_CTX = int(os.getenv("NMS_AUTOPILOT_NUM_CTX", "8192"))
NMS_AUTOPILOT_NUM_PREDICT = int(os.getenv("NMS_AUTOPILOT_NUM_PREDICT", "1200"))
CONVERSATION_DB_PATH = Path(os.getenv("LLM_OPS_CONVERSATION_DB", str(APP_DIR / "data" / "conversations.sqlite3")))
CONVERSATION_HISTORY_LIMIT = int(os.getenv("LLM_OPS_CONVERSATION_HISTORY_LIMIT", "18"))
CONVERSATION_MESSAGE_CHAR_LIMIT = int(os.getenv("LLM_OPS_CONVERSATION_MESSAGE_CHAR_LIMIT", "16000"))
CONVERSATION_TITLE_CHAR_LIMIT = int(os.getenv("LLM_OPS_CONVERSATION_TITLE_CHAR_LIMIT", "80"))
OPENAI_COMPAT_OBJECT = "chat.completion"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


CONVERSATION_LOCK = threading.Lock()


def clip_text(value: Any, limit: int = CONVERSATION_MESSAGE_CHAR_LIMIT) -> str:
    text = str(value or "")
    if len(text) <= limit:
        return text
    return f"{text[:limit].rstrip()}\n\n...[truncated {len(text) - limit} chars]"


def init_conversation_store() -> None:
    CONVERSATION_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with CONVERSATION_LOCK, sqlite3.connect(CONVERSATION_DB_PATH) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS conversations (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                meta_json TEXT NOT NULL DEFAULT '{}'
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS conversation_messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TEXT NOT NULL,
                meta_json TEXT NOT NULL DEFAULT '{}',
                FOREIGN KEY(conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
            )
            """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_conversation_messages_conversation_id
            ON conversation_messages(conversation_id, id)
            """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_conversations_updated_at
            ON conversations(updated_at)
            """
        )
        conn.commit()


def normalize_conversation_id(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    allowed = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"
    normalized = "".join(ch for ch in text if ch in allowed)
    return normalized[:80]


def serialize_conversation_row(row: sqlite3.Row) -> dict[str, Any]:
    try:
        meta = json.loads(row["meta_json"] or "{}")
    except Exception:
        meta = {}
    return {
        "id": row["id"],
        "title": row["title"] or "",
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
        "meta": meta,
    }


def create_conversation(title: str = "", meta: dict[str, Any] | None = None) -> dict[str, Any]:
    init_conversation_store()
    conversation_id = uuid.uuid4().hex
    now = utc_now()
    normalized_title = clip_text(title or "새 대화", CONVERSATION_TITLE_CHAR_LIMIT)
    with CONVERSATION_LOCK, sqlite3.connect(CONVERSATION_DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        conn.execute(
            """
            INSERT INTO conversations (id, title, created_at, updated_at, meta_json)
            VALUES (?, ?, ?, ?, ?)
            """,
            (conversation_id, normalized_title, now, now, json.dumps(meta or {}, ensure_ascii=False)),
        )
        conn.commit()
        row = conn.execute("SELECT * FROM conversations WHERE id = ?", (conversation_id,)).fetchone()
        return serialize_conversation_row(row)


def create_conversation_with_id(conversation_id: str, title: str = "", meta: dict[str, Any] | None = None) -> dict[str, Any]:
    init_conversation_store()
    normalized_id = normalize_conversation_id(conversation_id)
    if not normalized_id:
        return create_conversation(title=title, meta=meta)
    now = utc_now()
    normalized_title = clip_text(title or "새 대화", CONVERSATION_TITLE_CHAR_LIMIT)
    with CONVERSATION_LOCK, sqlite3.connect(CONVERSATION_DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        conn.execute(
            """
            INSERT INTO conversations (id, title, created_at, updated_at, meta_json)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                title = CASE WHEN conversations.title = '' THEN excluded.title ELSE conversations.title END,
                updated_at = excluded.updated_at,
                meta_json = excluded.meta_json
            """,
            (normalized_id, normalized_title, now, now, json.dumps(meta or {}, ensure_ascii=False)),
        )
        conn.commit()
        row = conn.execute("SELECT * FROM conversations WHERE id = ?", (normalized_id,)).fetchone()
        return serialize_conversation_row(row)


def ensure_conversation(conversation_id: str, title: str = "", meta: dict[str, Any] | None = None) -> dict[str, Any]:
    init_conversation_store()
    normalized_id = normalize_conversation_id(conversation_id)
    if not normalized_id:
        return create_conversation(title=title, meta=meta)
    with CONVERSATION_LOCK, sqlite3.connect(CONVERSATION_DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT * FROM conversations WHERE id = ?", (normalized_id,)).fetchone()
        if row:
            return serialize_conversation_row(row)
    return create_conversation_with_id(normalized_id, title=title, meta=meta)


def list_conversations(limit: int = 50) -> list[dict[str, Any]]:
    init_conversation_store()
    safe_limit = max(1, min(int(limit or 50), 200))
    with CONVERSATION_LOCK, sqlite3.connect(CONVERSATION_DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """
            SELECT c.*,
                   (SELECT COUNT(*) FROM conversation_messages m WHERE m.conversation_id = c.id) AS message_count
            FROM conversations c
            ORDER BY c.updated_at DESC
            LIMIT ?
            """,
            (safe_limit,),
        ).fetchall()
        items = []
        for row in rows:
            item = serialize_conversation_row(row)
            item["message_count"] = int(row["message_count"] or 0)
            items.append(item)
        return items


def load_conversation(conversation_id: str, limit: int = 100) -> dict[str, Any] | None:
    init_conversation_store()
    normalized_id = normalize_conversation_id(conversation_id)
    if not normalized_id:
        return None
    safe_limit = max(1, min(int(limit or 100), 500))
    with CONVERSATION_LOCK, sqlite3.connect(CONVERSATION_DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT * FROM conversations WHERE id = ?", (normalized_id,)).fetchone()
        if not row:
            return None
        message_rows = conn.execute(
            """
            SELECT id, role, content, created_at, meta_json
            FROM conversation_messages
            WHERE conversation_id = ?
            ORDER BY id DESC
            LIMIT ?
            """,
            (normalized_id, safe_limit),
        ).fetchall()
        messages = []
        for item in reversed(message_rows):
            try:
                meta = json.loads(item["meta_json"] or "{}")
            except Exception:
                meta = {}
            messages.append({
                "id": item["id"],
                "role": item["role"],
                "content": item["content"],
                "created_at": item["created_at"],
                "meta": meta,
            })
        conversation = serialize_conversation_row(row)
        conversation["messages"] = messages
        return conversation


def append_conversation_messages(
    conversation_id: str,
    messages: list[dict[str, Any]],
    meta: dict[str, Any] | None = None,
) -> None:
    init_conversation_store()
    normalized_id = normalize_conversation_id(conversation_id)
    if not normalized_id or not messages:
        return
    now = utc_now()
    rows = []
    for message in messages:
        role = str(message.get("role") or "").strip()
        content = clip_text(message.get("content") or "")
        if role not in {"user", "assistant", "tool"} or not content:
            continue
        rows.append((normalized_id, role, content, now, json.dumps(meta or {}, ensure_ascii=False)))
    if not rows:
        return
    with CONVERSATION_LOCK, sqlite3.connect(CONVERSATION_DB_PATH) as conn:
        conn.executemany(
            """
            INSERT INTO conversation_messages (conversation_id, role, content, created_at, meta_json)
            VALUES (?, ?, ?, ?, ?)
            """,
            rows,
        )
        title = ""
        for message in messages:
            if str(message.get("role") or "") == "user":
                title = clip_text(message.get("content") or "", CONVERSATION_TITLE_CHAR_LIMIT)
                break
        if title:
            conn.execute(
                """
                UPDATE conversations
                SET updated_at = ?,
                    title = CASE
                        WHEN title = '' OR title = '새 대화' THEN ?
                        ELSE title
                    END
                WHERE id = ?
                """,
                (now, title, normalized_id),
            )
        else:
            conn.execute("UPDATE conversations SET updated_at = ? WHERE id = ?", (now, normalized_id))
        conn.commit()


def delete_conversation(conversation_id: str) -> bool:
    init_conversation_store()
    normalized_id = normalize_conversation_id(conversation_id)
    if not normalized_id:
        return False
    with CONVERSATION_LOCK, sqlite3.connect(CONVERSATION_DB_PATH) as conn:
        conn.execute("DELETE FROM conversation_messages WHERE conversation_id = ?", (normalized_id,))
        result = conn.execute("DELETE FROM conversations WHERE id = ?", (normalized_id,))
        conn.commit()
        return result.rowcount > 0


def split_system_messages(messages: list[dict[str, Any]]) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    system_messages = []
    rest = []
    for message in messages:
        role = str(message.get("role") or "").strip()
        content = str(message.get("content") or "")
        if not content:
            continue
        normalized = {"role": role, "content": content}
        if role == "system":
            system_messages.append(normalized)
        elif role in {"user", "assistant", "tool"}:
            rest.append(normalized)
    return system_messages, rest


def conversation_history_for_model(conversation_id: str, limit: int = CONVERSATION_HISTORY_LIMIT) -> list[dict[str, str]]:
    conversation = load_conversation(conversation_id, limit=limit)
    if not conversation:
        return []
    history = []
    for message in conversation.get("messages") or []:
        role = str(message.get("role") or "")
        content = str(message.get("content") or "")
        if role in {"user", "assistant", "tool"} and content:
            history.append({"role": role, "content": content})
    return history


def run_command(args: list[str], timeout: int = 8) -> dict[str, Any]:
    try:
        proc = subprocess.run(
            args,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return {
            "ok": proc.returncode == 0,
            "returncode": proc.returncode,
            "stdout": proc.stdout.strip(),
            "stderr": proc.stderr.strip(),
        }
    except FileNotFoundError:
        return {"ok": False, "returncode": None, "stdout": "", "stderr": f"{args[0]} not found"}
    except subprocess.TimeoutExpired:
        return {"ok": False, "returncode": None, "stdout": "", "stderr": "command timeout"}


def parse_gpu_csv(output: str) -> list[dict[str, Any]]:
    if not output:
        return []
    reader = csv.reader(io.StringIO(output))
    gpus: list[dict[str, Any]] = []
    for row in reader:
        if len(row) < 5:
            continue
        name, total, used, util, temp = [item.strip() for item in row[:5]]
        gpus.append(
            {
                "name": name,
                "memory_total_mib": int(total.split()[0]) if total else None,
                "memory_used_mib": int(used.split()[0]) if used else None,
                "utilization_gpu_percent": int(util.split()[0]) if util else None,
                "temperature_c": int(temp.split()[0]) if temp else None,
            }
        )
    return gpus


def gpu_status() -> dict[str, Any]:
    cmd = run_command(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,memory.used,utilization.gpu,temperature.gpu",
            "--format=csv,noheader",
        ],
        timeout=8,
    )
    return {
        "available": cmd["ok"],
        "gpus": parse_gpu_csv(cmd["stdout"]) if cmd["ok"] else [],
        "error": None if cmd["ok"] else cmd["stderr"],
    }


def ollama_request(path: str, payload: dict[str, Any] | None = None, timeout: int | None = None) -> dict[str, Any]:
    url = f"{OLLAMA_BASE_URL}{path}"
    data = None
    headers = {"Accept": "application/json"}
    method = "GET"
    if payload is not None:
        method = "POST"
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers["Content-Type"] = "application/json; charset=utf-8"
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout or REQUEST_TIMEOUT) as res:
            body = res.read().decode("utf-8", errors="replace")
            return {"ok": True, "status": res.status, "data": json.loads(body) if body else {}}
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        try:
            parsed = json.loads(body)
        except json.JSONDecodeError:
            parsed = {"error": body}
        return {"ok": False, "status": exc.code, "data": parsed}
    except Exception as exc:  # noqa: BLE001 - response should include exact runtime failure.
        return {"ok": False, "status": 0, "data": {"error": str(exc)}}


def http_json_request(
    url: str,
    payload: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
    timeout: int | None = None,
) -> dict[str, Any]:
    data = None
    method = "GET"
    request_headers = {"Accept": "application/json"}
    request_headers.update(headers or {})
    if payload is not None:
        method = "POST"
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request_headers["Content-Type"] = "application/json; charset=utf-8"
    req = urllib.request.Request(url, data=data, headers=request_headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout or REQUEST_TIMEOUT) as res:
            body = res.read().decode("utf-8", errors="replace")
            return {"ok": True, "status": res.status, "data": json.loads(body) if body else {}}
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        try:
            parsed = json.loads(body)
        except json.JSONDecodeError:
            parsed = {"error": body}
        return {"ok": False, "status": exc.code, "data": parsed}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "status": 0, "data": {"error": str(exc)}}


def nms_headers() -> dict[str, str]:
    return {"X-NMS-ERP-Context-Token": NMS_CONTEXT_TOKEN} if NMS_CONTEXT_TOKEN else {}


def nms_get(path: str, params: dict[str, Any] | None = None, timeout: int | None = None) -> dict[str, Any]:
    if not NMS_CONTEXT_BASE_URL or not NMS_CONTEXT_TOKEN:
        return {"ok": False, "status": 503, "data": {"error": "NMS context integration is not configured"}}
    query = urllib.parse.urlencode({k: v for k, v in (params or {}).items() if v is not None and v != ""}, doseq=True)
    url = f"{NMS_CONTEXT_BASE_URL}{path}"
    if query:
        url = f"{url}?{query}"
    return http_json_request(url, headers=nms_headers(), timeout=timeout or NMS_CONTEXT_TIMEOUT)


def ollama_status() -> dict[str, Any]:
    tags = ollama_request("/api/tags", timeout=8)
    ps = ollama_request("/api/ps", timeout=8)
    service = run_command(["systemctl", "is-active", "ollama"], timeout=4)
    return {
        "base_url": OLLAMA_BASE_URL,
        "reachable": tags["ok"],
        "service_active": service["stdout"] if service["stdout"] else None,
        "models": tags["data"].get("models", []) if tags["ok"] else [],
        "running": ps["data"].get("models", []) if ps["ok"] else [],
        "error": None if tags["ok"] else tags["data"],
    }


def build_analysis_prompt(body: dict[str, Any]) -> list[dict[str, str]]:
    template = body.get("template") or "freeform"
    customer = body.get("customer") or "미지정 고객"
    question = body.get("question") or ""
    data = body.get("data") or body.get("logs") or ""
    if isinstance(data, (dict, list)):
        data_text = json.dumps(data, ensure_ascii=False, indent=2)
    else:
        data_text = str(data)

    templates = {
        "nms_events": (
            "너는 네트워크 관제 분석 담당자다. NMS 이벤트/시스로그/트래픽 관측값을 바탕으로 "
            "장애 가능성, 영향 범위, 우선 조치, 추가 확인 명령을 실무적으로 정리해라."
        ),
        "maintenance_report": (
            "너는 유지보수 보고서 작성자다. 작업일지, 현장상황, 방문일정, NMS 이벤트를 종합해 "
            "고객에게 제출 가능한 보고서 초안을 작성해라. 과장하지 말고 근거와 조치 중심으로 작성해라."
        ),
        "customer_history": (
            "너는 고객사 이력 요약 담당자다. 제공된 이력과 로그를 시간순으로 정리하고 반복 장애, "
            "장비 위험, 다음 점검 항목을 요약해라."
        ),
        "freeform": (
            "너는 한국어 실무형 IT 운영 보조자다. 제공된 자료를 근거로 짧고 명확하게 판단하고 "
            "다음 조치를 제안해라."
        ),
    }
    system_prompt = templates.get(template, templates["freeform"])
    user_prompt = (
        f"고객사: {customer}\n"
        f"요청: {question or '제공된 데이터를 분석해줘.'}\n\n"
        f"데이터:\n{data_text}\n\n"
        "출력 형식:\n"
        "1. 현재 판단\n"
        "2. 근거\n"
        "3. 위험도\n"
        "4. 즉시 조치\n"
        "5. 추가 확인 항목"
    )
    return [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]


def build_nms_evidence_digest(context: dict[str, Any]) -> str:
    matched = context.get("matched") or {}
    logs = context.get("logs") or {}
    traffic = context.get("traffic") or {}
    temporal = context.get("temporal") or {}
    off_hours = temporal.get("off_hours") or {}
    nas_file_activity = temporal.get("nas_file_activity") or {}
    nas_file_risk = temporal.get("nas_file_risk") or {}
    top_hours = temporal.get("top_event_hours") or []
    lines = [
        f"- 매칭 고객/현장: customers={matched.get('customer_count', 0)}, sites={matched.get('site_count', 0)}, site_codes={matched.get('site_codes') or []}",
        f"- 로그 건수: syslog={logs.get('syslog_count', 0)}, trap={logs.get('trap_count', 0)}",
        f"- 트래픽 인터페이스 자료: interface_count={traffic.get('interface_count', 0)}",
        f"- 새벽 이벤트: total={off_hours.get('total', 0)}, syslog={off_hours.get('syslog', 0)}, trap={off_hours.get('trap', 0)}",
        (
            "- 전체 NAS 파일작업: "
            f"risk={nas_file_activity.get('risk', 'unknown')}, "
            f"file_operation_count={nas_file_activity.get('file_operation_count', 0)}, "
            f"delete_rename_move_count={nas_file_activity.get('delete_rename_move_count', 0)}, "
            f"suspicious_keyword_count={nas_file_activity.get('suspicious_keyword_count', 0)}"
        ),
        (
            "- 새벽 NAS 파일위험: "
            f"risk={nas_file_risk.get('risk', 'unknown')}, "
            f"file_operation_count={nas_file_risk.get('file_operation_count', 0)}, "
            f"delete_rename_move_count={nas_file_risk.get('delete_rename_move_count', 0)}, "
            f"suspicious_keyword_count={nas_file_risk.get('suspicious_keyword_count', 0)}"
        ),
        f"- 이벤트 집중 시간대: {top_hours[:8]}",
    ]
    return "\n".join(lines)


def build_nms_deterministic_brief(context: dict[str, Any]) -> str:
    requested = context.get("requested") or {}
    matched = context.get("matched") or {}
    logs = context.get("logs") or {}
    traffic = context.get("traffic") or {}
    temporal = context.get("temporal") or {}
    off_hours = temporal.get("off_hours") or {}
    nas_file_activity = temporal.get("nas_file_activity") or {}
    nas_file_risk = temporal.get("nas_file_risk") or {}
    recent_off_hours = off_hours.get("recent_events") or []
    top_actors = nas_file_activity.get("top_actors") or []
    top_paths = nas_file_activity.get("top_paths") or []
    top_hours = temporal.get("top_event_hours") or []

    lines = [
        "검증된 NMS 근거 브리핑",
        f"- 대상: {', '.join(matched.get('customer_names') or []) or requested.get('customer_name') or '전체'} / site_codes={matched.get('site_codes') or []}",
        f"- 분석 범위: 최근 {requested.get('hours', '-')}시간, from={requested.get('from', '-')}",
        f"- 로그 원천값: syslog={logs.get('syslog_count', 0)}건, trap={logs.get('trap_count', 0)}건",
        f"- 이벤트 집중 시간대: {top_hours[:5]}",
        f"- 새벽 이벤트: total={off_hours.get('total', 0)}건, syslog={off_hours.get('syslog', 0)}건, trap={off_hours.get('trap', 0)}건",
        (
            "- 전체 NAS 파일작업: "
            f"{nas_file_activity.get('file_operation_count', 0)}건, "
            f"delete/rename/move={nas_file_activity.get('delete_rename_move_count', 0)}건, "
            f"랜섬웨어 키워드={nas_file_activity.get('suspicious_keyword_count', 0)}건"
        ),
        (
            "- 새벽 NAS 파일작업: "
            f"{nas_file_risk.get('file_operation_count', 0)}건, "
            f"delete/rename/move={nas_file_risk.get('delete_rename_move_count', 0)}건, "
            f"랜섬웨어 키워드={nas_file_risk.get('suspicious_keyword_count', 0)}건"
        ),
    ]

    if int(traffic.get("interface_count") or 0) == 0:
        lines.append("- 트래픽 해석: 이번 컨텍스트에는 traffic.interfaces 자료가 없어 이벤트 시간분포를 트래픽량으로 해석하면 안 됨")
    else:
        lines.append(f"- 트래픽 인터페이스 자료: {traffic.get('interface_count')}건")

    if recent_off_hours:
        lines.append("- 새벽 이벤트 샘플:")
        for item in recent_off_hours[:5]:
            lines.append(
                f"  · {item.get('received_at')} / {item.get('device_name')} / "
                f"{item.get('app_name') or item.get('trap_oid') or '-'} / {item.get('message_text') or '-'}"
            )

    if top_actors:
        lines.append("- 전체 NAS 파일작업 상위 사용자/IP:")
        for item in top_actors[:5]:
            lines.append(f"  · {item.get('user')}@{item.get('ip')}: {item.get('count')}건, last={item.get('last_seen_at')}")

    if top_paths:
        lines.append("- 전체 NAS 파일작업 상위 경로:")
        for item in top_paths[:5]:
            lines.append(f"  · {item.get('path')}: {item.get('count')}건")

    if int(nas_file_risk.get("file_operation_count") or 0) == 0 and int(nas_file_risk.get("suspicious_keyword_count") or 0) == 0:
        lines.append("- 1차 판정: 새벽 시간대 랜섬웨어성 파일작업 직접 징후는 현재 관측되지 않음")
    if int(nas_file_activity.get("file_operation_count") or 0) >= 100:
        lines.append("- 1차 판정: 전체 시간대 파일명 변경이 많으므로 정상 업무/프로그램 임시파일 패턴인지 사용자와 경로 기준으로 확인 필요")

    return "\n".join(lines)


def build_nms_analysis_prompt(context: dict[str, Any], question: str = "", depth: str = "deep") -> list[dict[str, str]]:
    requested = context.get("requested") or {}
    matched = context.get("matched") or {}
    customer_names = ", ".join(matched.get("customer_names") or []) or requested.get("customer_name") or "전체"
    evidence_digest = build_nms_evidence_digest(context)
    data_text = json.dumps(context, ensure_ascii=False, indent=2)[:NMS_ANALYSIS_CONTEXT_CHAR_LIMIT]
    system_prompt = (
        "너는 33번 NMS 상시 관제와 ERP 업무 이력을 함께 보는 실무 분석 담당자다. "
        "NMS 시스로그, SNMP Trap, 장비 상태, 트래픽/프로브/수집기 값, Grafana가 참조하는 "
        "NMS 원천 데이터를 근거로 장애 징후를 판단한다. 답변은 빠른 추측이 아니라 근거 기반 "
        "심층 분석이어야 한다. JSON에 없는 내용은 단정하지 말고 '자료 없음' 또는 '추정'으로 "
        "표시해라. 같은 말을 반복하지 말고, 시간/장비/수치/로그 문구를 근거로 인용해라. "
        "숫자가 0인 항목은 반드시 '관측되지 않음'으로 해석하고, 없는 데이터를 만들어내지 마라."
    )
    user_prompt = (
        f"분석 대상: {customer_names}\n"
        f"분석 모드: {depth}\n"
        f"요청: {question or '최근 관제 데이터 기준으로 이상 징후와 조치 우선순위를 판단해줘.'}\n\n"
        f"우선 적용할 근거 요약:\n{evidence_digest}\n\n"
        f"NMS 컨텍스트 JSON:\n{data_text}\n\n"
        "분석 지침:\n"
        "- 먼저 전체 JSON을 훑고, matched/sites/devices/traffic/logs/temporal 순서로 근거를 정리한다.\n"
        "- temporal.off_hours와 temporal.nas_file_risk가 있으면 새벽 이벤트, 랜섬웨어성 파일 작업, 반복 시간대를 별도 판단한다.\n"
        "- temporal.nas_file_activity는 전체 시간대 NAS 파일작업이고, temporal.nas_file_risk는 새벽 시간대 파일작업이다. 둘을 반드시 분리해서 설명해라.\n"
        "- Grafana 화면은 NMS DB를 시각화한 것이므로, Grafana 값이라고 표현할 때도 JSON의 traffic/logs/temporal 원천값을 근거로 삼는다.\n"
        "- temporal.top_event_hours는 이벤트 발생 시간 분포다. traffic.interfaces가 없으면 트래픽량으로 해석하지 마라.\n"
        "- nas_file_risk.file_operation_count가 0이면 새벽 파일 삭제/이동/이름변경은 관측되지 않았다고 써라. 전체 시간대 파일작업 여부는 nas_file_activity로 따로 판단해라.\n"
        "- 이벤트가 많아도 정상 주기 작업일 가능성과 장애/보안 이벤트 가능성을 분리한다.\n"
        "- 근거가 부족하면 어떤 데이터가 더 필요한지 명확히 적는다.\n\n"
        "출력 형식:\n"
        "1. 종합 판정: 정상/주의/위험 중 하나와 신뢰도\n"
        "2. 핵심 근거: 시간, 장비명, 수치, 로그 문구를 포함해 5개 이상\n"
        "3. 새벽 시간대 이벤트 분석: 집중 시간, 반복성, NAS 파일작업/랜섬웨어 의심 여부\n"
        "4. NMS/Grafana 관측값 해석: 장비 상태, 트래픽, 프로브, Trap/Syslog를 분리\n"
        "5. 의심 원인 우선순위: 가능성 높은 순서와 반박 근거\n"
        "6. 즉시 원격 확인 작업: 명령 또는 화면 경로 중심\n"
        "7. 현장 확인 필요 작업: 케이블/스위치/공유기/NAS/단말 기준\n"
        "8. 추가로 수집해야 할 데이터"
    )
    return [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]


def run_ollama_chat_messages(
    messages: list[dict[str, str]],
    model: str = DEFAULT_MODEL,
    timeout: int = REQUEST_TIMEOUT,
    options: dict[str, Any] | None = None,
    keep_alive: str = KEEP_ALIVE,
) -> dict[str, Any]:
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "keep_alive": keep_alive,
    }
    if options:
        payload["options"] = options
    started = time.monotonic()
    result = ollama_request("/api/chat", payload, timeout=timeout)
    elapsed_ms = int((time.monotonic() - started) * 1000)
    response_text = (result.get("data") or {}).get("message", {}).get("content")
    return {
        "ok": result["ok"],
        "status": result["status"],
        "model": model,
        "elapsed_ms": elapsed_ms,
        "response": response_text or "",
        "raw": result.get("data") or {},
    }


def model_summary(model: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": model.get("name") or model.get("model"),
        "size": model.get("size"),
        "modified_at": model.get("modified_at"),
        "details": model.get("details"),
        "expires_at": model.get("expires_at"),
        "size_vram": model.get("size_vram"),
    }


class NmsMonitor:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.state: dict[str, Any] = {
            "ok": False,
            "enabled": NMS_MONITOR_ENABLED,
            "configured": bool(NMS_CONTEXT_BASE_URL and NMS_CONTEXT_TOKEN),
            "last_checked_at": None,
            "next_check_seconds": NMS_MONITOR_INTERVAL,
            "summary": {},
            "customers": [],
            "error": None,
        }
        self.thread: threading.Thread | None = None

    def start(self) -> None:
        if not NMS_MONITOR_ENABLED:
            return
        self.thread = threading.Thread(target=self.run, name="nms-monitor", daemon=True)
        self.thread.start()

    def run(self) -> None:
        while True:
            self.refresh()
            time.sleep(max(NMS_MONITOR_INTERVAL, 15))

    def refresh(self) -> dict[str, Any]:
        result = nms_get(
            "/api/integrations/erp/customer-sites",
            {"hours": NMS_MONITOR_WINDOW_HOURS, "limit": NMS_MONITOR_LIMIT},
            timeout=NMS_CONTEXT_TIMEOUT,
        )
        if not result["ok"]:
            new_state = {
                "ok": False,
                "enabled": NMS_MONITOR_ENABLED,
                "configured": bool(NMS_CONTEXT_BASE_URL and NMS_CONTEXT_TOKEN),
                "last_checked_at": utc_now(),
                "next_check_seconds": NMS_MONITOR_INTERVAL,
                "summary": {},
                "customers": [],
                "error": result["data"],
            }
        else:
            data = result["data"]
            customers = data.get("customers") or []
            site_count = sum(int(customer.get("site_count") or 0) for customer in customers)
            event_count = sum(int(customer.get("recent_event_count") or 0) for customer in customers)
            down_devices = sum(int(customer.get("down_device_count") or 0) for customer in customers)
            degraded_devices = sum(int(customer.get("degraded_device_count") or 0) for customer in customers)
            active_customers = [c for c in customers if int(c.get("recent_event_count") or 0) > 0]
            risk = "normal"
            if down_devices > 0:
                risk = "critical"
            elif degraded_devices > 0 or event_count > 0:
                risk = "watch"
            new_state = {
                "ok": True,
                "enabled": NMS_MONITOR_ENABLED,
                "configured": True,
                "last_checked_at": utc_now(),
                "next_check_seconds": NMS_MONITOR_INTERVAL,
                "summary": {
                    "risk": risk,
                    "customer_count": len(customers),
                    "site_count": site_count,
                    "active_customer_count": len(active_customers),
                    "recent_event_count": event_count,
                    "down_device_count": down_devices,
                    "degraded_device_count": degraded_devices,
                    "window_hours": NMS_MONITOR_WINDOW_HOURS,
                },
                "customers": customers,
                "error": None,
            }
        with self.lock:
            self.state = new_state
        return new_state

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            return json.loads(json.dumps(self.state, default=str, ensure_ascii=False))


NMS_MONITOR = NmsMonitor()


def safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value or default)
    except (TypeError, ValueError):
        return default


def build_autopilot_target(customer: dict[str, Any], site: dict[str, Any] | None = None) -> dict[str, Any]:
    source = site or customer
    recent_events = safe_int(source.get("recent_event_count"))
    down_devices = safe_int(source.get("down_device_count"))
    degraded_devices = safe_int(source.get("degraded_device_count"))
    priority = safe_int(source.get("priority"), 3)
    score = (down_devices * 10000) + (degraded_devices * 2500) + min(recent_events, 5000) + max(0, 5 - priority) * 100
    site_id = source.get("site_id") if site else None
    site_code = str(source.get("site_code") or source.get("customer_site_code") or "").strip()
    customer_id = customer.get("customer_id")
    key = f"site:{site_id or site_code}" if site else f"customer:{customer_id or customer.get('customer_name')}"
    return {
        "key": normalize_conversation_id(key),
        "score": score,
        "customer_id": customer_id,
        "customer_name": customer.get("customer_name") or "",
        "customer_code": customer.get("customer_code") or "",
        "site_id": site_id,
        "site_name": source.get("site_name") or "",
        "site_code": site_code,
        "recent_event_count": recent_events,
        "down_device_count": down_devices,
        "degraded_device_count": degraded_devices,
    }


def select_autopilot_targets(monitor_state: dict[str, Any]) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for customer in monitor_state.get("customers") or []:
        sites = customer.get("sites") or []
        if sites:
            for site in sites:
                target = build_autopilot_target(customer, site)
                if target["score"] > 0 and target["recent_event_count"] >= NMS_AUTOPILOT_MIN_EVENT_COUNT:
                    candidates.append(target)
        else:
            target = build_autopilot_target(customer)
            if target["score"] > 0 and target["recent_event_count"] >= NMS_AUTOPILOT_MIN_EVENT_COUNT:
                candidates.append(target)
    candidates.sort(key=lambda item: (item["score"], item["recent_event_count"]), reverse=True)
    return candidates[:max(1, NMS_AUTOPILOT_MAX_TARGETS)]


class NmsAutopilot:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.thread: threading.Thread | None = None
        self.last_target_run: dict[str, float] = {}
        self.state: dict[str, Any] = {
            "ok": False,
            "enabled": NMS_AUTOPILOT_ENABLED,
            "configured": bool(NMS_CONTEXT_BASE_URL and NMS_CONTEXT_TOKEN),
            "running": False,
            "last_run_at": None,
            "next_check_seconds": NMS_AUTOPILOT_INTERVAL,
            "last_error": None,
            "last_targets": [],
            "last_results": [],
            "summary": {},
        }

    def start(self) -> None:
        if not NMS_AUTOPILOT_ENABLED:
            return
        self.thread = threading.Thread(target=self.run, name="nms-autopilot", daemon=True)
        self.thread.start()

    def run(self) -> None:
        time.sleep(max(0, NMS_AUTOPILOT_START_DELAY_SECONDS))
        while True:
            started = time.monotonic()
            try:
                self.run_once(force=False)
            except Exception as exc:  # noqa: BLE001
                self._set_error(str(exc))
                print(f"{utc_now()} nms-autopilot failed: {exc}", flush=True)
            elapsed = time.monotonic() - started
            time.sleep(max(60, NMS_AUTOPILOT_INTERVAL - elapsed))

    def run_once(self, force: bool = False) -> dict[str, Any]:
        with self.lock:
            self.state["running"] = True
            self.state["last_error"] = None
        monitor_state = NMS_MONITOR.refresh()
        if not monitor_state.get("ok"):
            self._set_error(f"NMS monitor not ok: {monitor_state.get('error')}")
            return self.snapshot()

        selected = []
        now = time.time()
        for target in select_autopilot_targets(monitor_state):
            last_run = self.last_target_run.get(target["key"], 0)
            if not force and now - last_run < NMS_AUTOPILOT_TARGET_COOLDOWN:
                continue
            selected.append(target)

        results = []
        for target in selected:
            result = self.analyze_target(target)
            results.append(result)
            self.last_target_run[target["key"]] = time.time()

        summary = {
            "candidate_count": len(select_autopilot_targets(monitor_state)),
            "selected_count": len(selected),
            "succeeded_count": sum(1 for item in results if item.get("ok")),
            "failed_count": sum(1 for item in results if not item.get("ok")),
            "model": NMS_AUTOPILOT_MODEL,
            "window_hours": NMS_AUTOPILOT_WINDOW_HOURS,
        }
        with self.lock:
            self.state.update({
                "ok": True,
                "enabled": NMS_AUTOPILOT_ENABLED,
                "configured": bool(NMS_CONTEXT_BASE_URL and NMS_CONTEXT_TOKEN),
                "running": False,
                "last_run_at": utc_now(),
                "next_check_seconds": NMS_AUTOPILOT_INTERVAL,
                "last_error": None,
                "last_targets": selected,
                "last_results": results,
                "summary": summary,
            })
        return self.snapshot()

    def analyze_target(self, target: dict[str, Any]) -> dict[str, Any]:
        site_codes = target.get("site_code") or ""
        customer_name = target.get("customer_name") or ""
        context_result = nms_get(
            "/api/integrations/erp/nms-context",
            {
                "customer_name": customer_name,
                "site_codes": site_codes,
                "hours": NMS_AUTOPILOT_WINDOW_HOURS,
                "limit": NMS_AUTOPILOT_CONTEXT_LIMIT,
            },
            timeout=max(NMS_CONTEXT_TIMEOUT, 30),
        )
        conversation_id = normalize_conversation_id(
            f"{NMS_AUTOPILOT_CONVERSATION_PREFIX}-"
            f"{'site-' + str(target.get('site_id')) if target.get('site_id') else 'customer-' + str(target.get('customer_id') or 'unknown')}"
        )
        title = f"NMS 자동분석 {target.get('customer_name') or '-'} {target.get('site_name') or target.get('site_code') or ''}".strip()
        conversation = ensure_conversation(conversation_id, title=title, meta={"source": "nms-autopilot", "target": target})
        if not context_result["ok"]:
            return {
                "ok": False,
                "target": target,
                "conversation_id": conversation["id"],
                "error": context_result["data"],
                "time": utc_now(),
            }

        context = context_result["data"]
        question = (
            "상시 모니터링 자동 점검이다. 최근 NMS/Grafana 원천값을 근거로 "
            "장애/보안/랜섬웨어/장비 불량 가능성과 다음 조치를 실무적으로 판단해라."
        )
        prompt_messages = build_nms_analysis_prompt(context, question, "deep")
        system_messages, current_messages = split_system_messages(prompt_messages)
        history = conversation_history_for_model(conversation["id"], limit=min(CONVERSATION_HISTORY_LIMIT, 8))
        messages = [*system_messages, *history, *current_messages]
        deterministic_brief = build_nms_deterministic_brief(context)
        chat = run_ollama_chat_messages(
            messages,
            model=NMS_AUTOPILOT_MODEL,
            timeout=NMS_AUTOPILOT_TIMEOUT,
            options={
                "temperature": 0.08,
                "top_p": 0.82,
                "repeat_penalty": NMS_ANALYSIS_REPEAT_PENALTY,
                "num_ctx": NMS_AUTOPILOT_NUM_CTX,
                "num_predict": NMS_AUTOPILOT_NUM_PREDICT,
            },
        )
        response_text = chat.get("response") or ""
        if response_text:
            response_text = f"{deterministic_brief}\n\n---\n\nLLM 심층 분석\n{response_text}"
        append_conversation_messages(
            conversation["id"],
            [
                {
                    "role": "user",
                    "content": (
                        f"NMS 자동분석: customer={customer_name or '-'}, site_codes={site_codes or '-'}, "
                        f"score={target.get('score')}, events={target.get('recent_event_count')}, "
                        f"degraded={target.get('degraded_device_count')}, down={target.get('down_device_count')}"
                    ),
                },
                {
                    "role": "assistant",
                    "content": response_text or json.dumps(chat.get("raw") or {}, ensure_ascii=False),
                },
            ],
            meta={"model": NMS_AUTOPILOT_MODEL, "source": "nms-autopilot", "elapsed_ms": chat.get("elapsed_ms")},
        )
        return {
            "ok": bool(chat.get("ok")),
            "target": target,
            "conversation_id": conversation["id"],
            "elapsed_ms": chat.get("elapsed_ms"),
            "response_chars": len(response_text),
            "error": None if chat.get("ok") else chat.get("raw"),
            "time": utc_now(),
        }

    def _set_error(self, message: str) -> None:
        with self.lock:
            self.state.update({
                "ok": False,
                "running": False,
                "last_run_at": utc_now(),
                "last_error": message,
            })

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            return json.loads(json.dumps(self.state, default=str, ensure_ascii=False))


NMS_AUTOPILOT = NmsAutopilot()


class Handler(BaseHTTPRequestHandler):
    server_version = "MetroLLMOps/1.0"

    def log_message(self, fmt: str, *args: Any) -> None:
        print(f"{utc_now()} {self.address_string()} {fmt % args}", flush=True)

    def end_headers(self) -> None:
        self.send_header("Access-Control-Allow-Origin", ALLOWED_ORIGIN)
        self.send_header("Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization, X-LLM-Ops-Token")
        self.send_header("X-Content-Type-Options", "nosniff")
        super().end_headers()

    def do_OPTIONS(self) -> None:  # noqa: N802
        self.send_response(HTTPStatus.NO_CONTENT)
        self.end_headers()

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/" or self.path.startswith("/?"):
            return self.html_response(INDEX_HTML)
        route = self.path.split("?", 1)[0]
        if route == "/v1/models":
            if not self.authorized():
                return self.openai_error_response("unauthorized", "unauthorized", HTTPStatus.UNAUTHORIZED)
            return self.handle_openai_models()
        if route == "/api/config":
            return self.json_response(
                {
                    "auth_required": bool(API_TOKEN),
                    "default_model": DEFAULT_MODEL,
                    "fast_model": FAST_MODEL,
                    "ollama_base_url": OLLAMA_BASE_URL,
                    "nms_context_configured": bool(NMS_CONTEXT_BASE_URL and NMS_CONTEXT_TOKEN),
                    "nms_context_base_url": NMS_CONTEXT_BASE_URL,
                    "keep_alive": KEEP_ALIVE,
                    "time": utc_now(),
                }
            )
        if route == "/api/health":
            return self.json_response(self.health_payload())
        if route == "/api/models":
            status = ollama_status()
            return self.json_response(
                {
                    "ok": status["reachable"],
                    "models": [model_summary(m) for m in status["models"]],
                    "running": [model_summary(m) for m in status["running"]],
                    "default_model": DEFAULT_MODEL,
                    "fast_model": FAST_MODEL,
                    "error": status["error"],
                },
                HTTPStatus.OK if status["reachable"] else HTTPStatus.BAD_GATEWAY,
            )
        if route == "/api/conversations" or route.startswith("/api/conversations/"):
            if not self.authorized():
                return self.json_response({"error": "unauthorized"}, HTTPStatus.UNAUTHORIZED)
            return self.handle_conversations_get(route, requestUrl=urllib.parse.urlparse(self.path))
        if route.startswith("/api/nms/"):
            if not self.authorized():
                return self.json_response({"error": "unauthorized"}, HTTPStatus.UNAUTHORIZED)
            return self.handle_nms_get(route, requestUrl=urllib.parse.urlparse(self.path))
        return self.json_response({"error": "not found"}, HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:  # noqa: N802
        route = self.path.split("?", 1)[0]
        if not self.authorized():
            return self.json_response({"error": "unauthorized"}, HTTPStatus.UNAUTHORIZED)
        body = self.read_json_body()
        if body is None:
            return
        if route == "/v1/chat/completions":
            return self.handle_openai_chat_completions(body)
        if route == "/api/conversations":
            return self.handle_create_conversation(body)
        if route.startswith("/api/conversations/"):
            return self.handle_conversations_post(route, body)
        if route == "/api/chat":
            return self.handle_chat(body)
        if route == "/api/analyze":
            body["messages"] = build_analysis_prompt(body)
            return self.handle_chat(body)
        if route == "/api/nms/analyze":
            return self.handle_nms_analyze(body)
        if route == "/api/nms/autopilot/run":
            return self.json_response(NMS_AUTOPILOT.run_once(force=bool(body.get("force", True))))
        if route == "/api/preload":
            return self.handle_model_lifecycle(body, keep_alive=body.get("keep_alive") or KEEP_ALIVE)
        if route == "/api/unload":
            return self.handle_model_lifecycle(body, keep_alive=0)
        if route == "/api/benchmark":
            body.setdefault("prompt", "현재 GPU 서버의 LLM 응답 상태를 한 문장으로 점검해줘.")
            body.setdefault("model", FAST_MODEL)
            return self.handle_chat(body)
        return self.json_response({"error": "not found"}, HTTPStatus.NOT_FOUND)

    def do_DELETE(self) -> None:  # noqa: N802
        route = self.path.split("?", 1)[0]
        if not self.authorized():
            return self.json_response({"error": "unauthorized"}, HTTPStatus.UNAUTHORIZED)
        if route.startswith("/api/conversations/"):
            conversation_id = route.rsplit("/", 1)[-1]
            deleted = delete_conversation(conversation_id)
            return self.json_response({"success": deleted, "deleted": deleted}, HTTPStatus.OK if deleted else HTTPStatus.NOT_FOUND)
        return self.json_response({"error": "not found"}, HTTPStatus.NOT_FOUND)

    def handle_conversations_get(self, route: str, requestUrl: urllib.parse.ParseResult) -> None:
        params = urllib.parse.parse_qs(requestUrl.query)
        first = lambda key, default="": (params.get(key) or [default])[0]
        if route == "/api/conversations":
            limit = int(first("limit", "50") or "50")
            return self.json_response({"success": True, "items": list_conversations(limit=limit)})
        conversation_id = route.rsplit("/", 1)[-1]
        conversation = load_conversation(conversation_id, limit=int(first("limit", "100") or "100"))
        if not conversation:
            return self.json_response({"error": "conversation not found"}, HTTPStatus.NOT_FOUND)
        return self.json_response({"success": True, "conversation": conversation})

    def handle_create_conversation(self, body: dict[str, Any]) -> None:
        conversation = create_conversation(
            title=str(body.get("title") or "새 대화").strip(),
            meta=body.get("meta") if isinstance(body.get("meta"), dict) else {},
        )
        return self.json_response({"success": True, "conversation": conversation}, HTTPStatus.CREATED)

    def handle_conversations_post(self, route: str, body: dict[str, Any]) -> None:
        if route.endswith("/clear"):
            conversation_id = route.split("/")[-2]
            conversation = load_conversation(conversation_id, limit=1)
            if not conversation:
                return self.json_response({"error": "conversation not found"}, HTTPStatus.NOT_FOUND)
            with CONVERSATION_LOCK, sqlite3.connect(CONVERSATION_DB_PATH) as conn:
                conn.execute("DELETE FROM conversation_messages WHERE conversation_id = ?", (conversation["id"],))
                conn.execute("UPDATE conversations SET updated_at = ? WHERE id = ?", (utc_now(), conversation["id"]))
                conn.commit()
            return self.json_response({"success": True, "conversation_id": conversation["id"]})
        return self.json_response({"error": "not found"}, HTTPStatus.NOT_FOUND)

    def handle_openai_models(self) -> None:
        status = ollama_status()
        data = []
        for model in status.get("models") or []:
            model_name = model.get("name") or model.get("model")
            if not model_name:
                continue
            data.append({
                "id": model_name,
                "object": "model",
                "created": 0,
                "owned_by": "metro-llm-ops",
            })
        return self.json_response({"object": "list", "data": data}, HTTPStatus.OK if status["reachable"] else HTTPStatus.BAD_GATEWAY)

    def handle_nms_get(self, route: str, requestUrl: urllib.parse.ParseResult) -> None:
        params = urllib.parse.parse_qs(requestUrl.query)
        first = lambda key, default="": (params.get(key) or [default])[0]
        if route == "/api/nms/customers":
            result = nms_get(
                "/api/integrations/erp/customer-sites",
                {
                    "q": first("q"),
                    "hours": first("hours", str(NMS_MONITOR_WINDOW_HOURS)),
                    "limit": first("limit", str(NMS_MONITOR_LIMIT)),
                },
                timeout=NMS_CONTEXT_TIMEOUT,
            )
            return self.json_response(result["data"], HTTPStatus.OK if result["ok"] else HTTPStatus.BAD_GATEWAY)
        if route == "/api/nms/context":
            result = nms_get(
                "/api/integrations/erp/nms-context",
                {
                    "customer_name": first("customer_name"),
                    "site_codes": first("site_codes"),
                    "hours": first("hours", str(NMS_DEEP_ANALYSIS_WINDOW_HOURS)),
                    "limit": first("limit", str(NMS_DEEP_ANALYSIS_LIMIT)),
                },
                timeout=max(NMS_CONTEXT_TIMEOUT, 30),
            )
            return self.json_response(result["data"], HTTPStatus.OK if result["ok"] else HTTPStatus.BAD_GATEWAY)
        if route == "/api/nms/monitor/status":
            return self.json_response(NMS_MONITOR.snapshot())
        if route == "/api/nms/monitor/refresh":
            return self.json_response(NMS_MONITOR.refresh())
        if route == "/api/nms/autopilot/status":
            return self.json_response(NMS_AUTOPILOT.snapshot())
        return self.json_response({"error": "not found"}, HTTPStatus.NOT_FOUND)

    def authorized(self) -> bool:
        if not API_TOKEN:
            return True
        auth = self.headers.get("Authorization", "")
        token = self.headers.get("X-LLM-Ops-Token", "")
        if auth.lower().startswith("bearer "):
            token = auth.split(" ", 1)[1].strip()
        return token == API_TOKEN

    def read_json_body(self) -> dict[str, Any] | None:
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            return self.json_response({"error": "invalid content-length"}, HTTPStatus.BAD_REQUEST)
        if length > MAX_BODY_BYTES:
            return self.json_response({"error": "body too large"}, HTTPStatus.REQUEST_ENTITY_TOO_LARGE)
        raw = self.rfile.read(length) if length else b"{}"
        try:
            parsed = json.loads(raw.decode("utf-8"))
            if not isinstance(parsed, dict):
                raise ValueError("body must be an object")
            return parsed
        except Exception as exc:  # noqa: BLE001
            self.json_response({"error": f"invalid json: {exc}"}, HTTPStatus.BAD_REQUEST)
            return None

    def run_chat_completion(self, body: dict[str, Any]) -> tuple[dict[str, Any], HTTPStatus]:
        model = str(body.get("model") or DEFAULT_MODEL)
        messages = body.get("messages")
        if not messages:
            prompt = str(body.get("prompt") or "")
            system = str(body.get("system") or "")
            if not prompt:
                return {"error": "prompt or messages is required"}, HTTPStatus.BAD_REQUEST
            messages = []
            if system:
                messages.append({"role": "system", "content": system})
            messages.append({"role": "user", "content": prompt})
        if not isinstance(messages, list):
            return {"error": "messages must be a list"}, HTTPStatus.BAD_REQUEST

        current_messages = [
            {"role": str(message.get("role") or ""), "content": str(message.get("content") or "")}
            for message in messages
            if isinstance(message, dict) and str(message.get("content") or "")
        ]
        conversation_id = normalize_conversation_id(body.get("conversation_id") or body.get("thread_id") or "")
        remember = bool(conversation_id or body.get("remember") or body.get("persist"))
        conversation = None
        if remember:
            title = str(body.get("conversation_title") or "").strip()
            if not title:
                first_user = next((item.get("content") for item in current_messages if item.get("role") == "user"), "")
                title = clip_text(first_user, CONVERSATION_TITLE_CHAR_LIMIT) if first_user else "새 대화"
            conversation = ensure_conversation(conversation_id, title=title, meta={"source": body.get("source") or "chat"})
            conversation_id = conversation["id"]
            history = conversation_history_for_model(conversation_id, limit=CONVERSATION_HISTORY_LIMIT)
            system_messages, non_system_current = split_system_messages(current_messages)
            messages = [*system_messages, *history, *non_system_current]

        options = body.get("options") if isinstance(body.get("options"), dict) else {}
        for key in ("temperature", "num_ctx", "num_predict", "top_p", "top_k", "repeat_penalty"):
            if key in body:
                options[key] = body[key]
        payload = {
            "model": model,
            "messages": messages,
            "stream": False,
            "keep_alive": body.get("keep_alive", KEEP_ALIVE),
        }
        if options:
            payload["options"] = options
        started = time.monotonic()
        result = ollama_request("/api/chat", payload, timeout=int(body.get("timeout") or REQUEST_TIMEOUT))
        elapsed_ms = int((time.monotonic() - started) * 1000)
        response_text = (result["data"].get("message") or {}).get("content")
        prepend_report = str(body.get("prepend_report") or "").strip()
        if prepend_report and response_text:
            response_text = f"{prepend_report}\n\n---\n\nLLM 심층 분석\n{response_text}"
            if isinstance(result["data"].get("message"), dict):
                result["data"]["message"]["content"] = response_text
        if remember and conversation_id and response_text:
            store_messages = body.get("conversation_store_messages")
            if not isinstance(store_messages, list):
                store_messages = [
                    message for message in current_messages
                    if message.get("role") in {"user", "assistant", "tool"}
                ]
            append_conversation_messages(
                conversation_id,
                [
                    *[
                        {"role": str(message.get("role") or ""), "content": str(message.get("content") or "")}
                        for message in store_messages
                        if isinstance(message, dict)
                    ],
                    {"role": "assistant", "content": response_text},
                ],
                meta={"model": model, "elapsed_ms": elapsed_ms},
            )
        response_body = {
            "ok": result["ok"],
            "model": model,
            "elapsed_ms": elapsed_ms,
            "conversation_id": conversation_id or None,
            "message": result["data"].get("message"),
            "response": response_text,
            "raw": result["data"],
            "gpu": gpu_status(),
            "time": utc_now(),
        }
        status = HTTPStatus.OK if result["ok"] else HTTPStatus.BAD_GATEWAY
        return response_body, status

    def handle_chat(self, body: dict[str, Any]) -> None:
        payload, status = self.run_chat_completion(body)
        return self.json_response(payload, status)

    def handle_openai_chat_completions(self, body: dict[str, Any]) -> None:
        chat_body = dict(body)
        if "max_tokens" in body and "num_predict" not in chat_body:
            chat_body["num_predict"] = body.get("max_tokens")
        if "max_completion_tokens" in body and "num_predict" not in chat_body:
            chat_body["num_predict"] = body.get("max_completion_tokens")
        payload, status = self.run_chat_completion(chat_body)
        if status != HTTPStatus.OK or not payload.get("ok"):
            message = payload.get("error") or (payload.get("raw") or {}).get("error") or "chat completion failed"
            return self.openai_error_response(str(message), "server_error", status)

        content = payload.get("response") or ""
        created = int(time.time())
        completion = {
            "id": f"chatcmpl-{uuid.uuid4().hex}",
            "object": OPENAI_COMPAT_OBJECT,
            "created": created,
            "model": payload.get("model") or str(body.get("model") or DEFAULT_MODEL),
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            },
            "conversation_id": payload.get("conversation_id"),
        }
        return self.json_response(completion, HTTPStatus.OK)

    def handle_nms_analyze(self, body: dict[str, Any]) -> None:
        customer_name = str(body.get("customer_name") or body.get("customer") or "").strip()
        site_codes = body.get("site_codes") or body.get("site_code") or ""
        if isinstance(site_codes, list):
            site_codes = ",".join(str(value) for value in site_codes if value)
        site_codes = str(site_codes).strip()
        if not customer_name and not site_codes:
            return self.json_response({"error": "customer_name or site_codes is required"}, HTTPStatus.BAD_REQUEST)

        depth = str(body.get("depth") or body.get("analysis_depth") or "deep").strip().lower()
        default_hours = NMS_DEEP_ANALYSIS_WINDOW_HOURS if depth in {"deep", "심층", "detailed"} else NMS_MONITOR_WINDOW_HOURS
        default_limit = NMS_DEEP_ANALYSIS_LIMIT if depth in {"deep", "심층", "detailed"} else 20

        result = nms_get(
            "/api/integrations/erp/nms-context",
            {
                "customer_name": customer_name,
                "site_codes": site_codes,
                "hours": body.get("hours") or default_hours,
                "limit": body.get("limit") or default_limit,
            },
            timeout=max(NMS_CONTEXT_TIMEOUT, 30),
        )
        if not result["ok"]:
            return self.json_response(result["data"], HTTPStatus.BAD_GATEWAY)

        body["messages"] = build_nms_analysis_prompt(result["data"], str(body.get("question") or ""), depth)
        body["prepend_report"] = build_nms_deterministic_brief(result["data"])
        body["conversation_store_messages"] = [
            {
                "role": "user",
                "content": (
                    f"NMS 심층 분석 요청: customer={customer_name or '-'}, "
                    f"site_codes={site_codes or '-'}, hours={body.get('hours') or default_hours}, "
                    f"question={str(body.get('question') or '').strip() or '-'}"
                ),
            }
        ]
        body.setdefault("model", DEFAULT_MODEL)
        body.setdefault("temperature", 0.08)
        body.setdefault("top_p", 0.82)
        body.setdefault("repeat_penalty", NMS_ANALYSIS_REPEAT_PENALTY)
        body.setdefault("num_ctx", NMS_ANALYSIS_NUM_CTX)
        body.setdefault("num_predict", NMS_ANALYSIS_NUM_PREDICT)
        body.setdefault("timeout", NMS_ANALYSIS_TIMEOUT)
        return self.handle_chat(body)

    def handle_model_lifecycle(self, body: dict[str, Any], keep_alive: str | int) -> None:
        model = str(body.get("model") or DEFAULT_MODEL)
        result = ollama_request(
            "/api/generate",
            {"model": model, "prompt": "", "stream": False, "keep_alive": keep_alive},
            timeout=int(body.get("timeout") or 120),
        )
        return self.json_response(
            {
                "ok": result["ok"],
                "model": model,
                "keep_alive": keep_alive,
                "ollama": result["data"],
                "running": ollama_status().get("running"),
                "gpu": gpu_status(),
                "time": utc_now(),
            },
            HTTPStatus.OK if result["ok"] else HTTPStatus.BAD_GATEWAY,
        )

    def health_payload(self) -> dict[str, Any]:
        return {
            "ok": True,
            "time": utc_now(),
            "host": socket.gethostname(),
            "app": {
                "bind": f"{HOST}:{PORT}",
                "auth_required": bool(API_TOKEN),
                "default_model": DEFAULT_MODEL,
                "fast_model": FAST_MODEL,
            },
            "nms_monitor": NMS_MONITOR.snapshot(),
            "nms_autopilot": NMS_AUTOPILOT.snapshot(),
            "ollama": ollama_status(),
            "gpu": gpu_status(),
        }

    def json_response(self, payload: dict[str, Any], status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def openai_error_response(self, message: str, error_type: str, status: HTTPStatus = HTTPStatus.BAD_REQUEST) -> None:
        return self.json_response(
            {
                "error": {
                    "message": message,
                    "type": error_type,
                    "param": None,
                    "code": None,
                }
            },
            status,
        )

    def html_response(self, html: str) -> None:
        body = html.encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


INDEX_HTML = r"""<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Metro LLM Ops</title>
  <style>
    :root {
      color-scheme: dark;
      --bg: #0b1117;
      --panel: rgba(18, 29, 38, 0.92);
      --panel-2: rgba(28, 45, 56, 0.86);
      --text: #e9f0f2;
      --muted: #91a4ae;
      --line: rgba(255, 255, 255, 0.10);
      --accent: #59d6b5;
      --warn: #f7c46c;
      --bad: #ff7b7b;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      min-height: 100vh;
      font-family: "Pretendard", "Noto Sans KR", sans-serif;
      color: var(--text);
      background:
        radial-gradient(circle at top left, rgba(89, 214, 181, 0.22), transparent 32rem),
        radial-gradient(circle at 80% 10%, rgba(247, 196, 108, 0.13), transparent 28rem),
        linear-gradient(135deg, #071016, #111820 52%, #0a1016);
    }
    header {
      padding: 26px clamp(18px, 4vw, 48px) 12px;
      display: flex;
      gap: 18px;
      justify-content: space-between;
      align-items: end;
    }
    h1 { margin: 0; font-size: clamp(26px, 4vw, 46px); letter-spacing: -0.05em; }
    .subtitle { margin-top: 8px; color: var(--muted); }
    main {
      padding: 16px clamp(18px, 4vw, 48px) 42px;
      display: grid;
      grid-template-columns: minmax(280px, 420px) 1fr;
      gap: 18px;
    }
    .card {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 22px;
      padding: 18px;
      box-shadow: 0 24px 80px rgba(0, 0, 0, 0.24);
      backdrop-filter: blur(18px);
    }
    .stack { display: grid; gap: 14px; }
    .row { display: flex; gap: 10px; flex-wrap: wrap; align-items: center; }
    label { color: var(--muted); font-size: 13px; display: block; margin-bottom: 6px; }
    input, select, textarea, button {
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 14px;
      background: rgba(0, 0, 0, 0.28);
      color: var(--text);
      padding: 12px 13px;
      font: inherit;
      outline: none;
    }
    textarea { min-height: 220px; resize: vertical; line-height: 1.55; }
    button {
      width: auto;
      min-width: 110px;
      background: linear-gradient(135deg, var(--accent), #8ee2a4);
      color: #071016;
      border: none;
      font-weight: 800;
      cursor: pointer;
    }
    button.secondary { background: var(--panel-2); color: var(--text); border: 1px solid var(--line); }
    .metric {
      display: grid;
      grid-template-columns: 1fr auto;
      gap: 8px;
      padding: 12px;
      border-radius: 16px;
      background: rgba(255, 255, 255, 0.05);
      border: 1px solid var(--line);
    }
    .metric b { font-size: 22px; letter-spacing: -0.04em; }
    .pill {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      padding: 6px 10px;
      border-radius: 999px;
      background: rgba(89, 214, 181, 0.13);
      color: var(--accent);
      font-size: 12px;
    }
    .pill.warn { background: rgba(247, 196, 108, 0.13); color: var(--warn); }
    .pill.bad { background: rgba(255, 123, 123, 0.13); color: var(--bad); }
    pre {
      margin: 0;
      white-space: pre-wrap;
      word-break: break-word;
      line-height: 1.58;
      color: #f5faf9;
      font-size: 14px;
    }
    .result { min-height: 360px; background: rgba(0, 0, 0, 0.25); }
    .grid2 { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }
    @media (max-width: 920px) {
      header { align-items: start; flex-direction: column; }
      main { grid-template-columns: 1fr; }
      .grid2 { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <header>
    <div>
      <h1>Metro LLM Ops</h1>
      <div class="subtitle">118 GPU 서버 로컬 LLM 운영 콘솔</div>
    </div>
    <div id="statusPill" class="pill warn">상태 확인 중</div>
  </header>
  <main>
    <section class="stack">
      <div class="card stack">
        <div>
          <label>API 토큰</label>
          <input id="token" type="password" placeholder="llm-ops.env의 LLM_OPS_TOKEN" />
        </div>
        <div>
          <label>대화 세션</label>
          <select id="conversationSelect" onchange="selectConversation()">
            <option value="">새 대화</option>
          </select>
        </div>
        <div class="grid2">
          <div>
            <label>모델</label>
            <select id="model"></select>
          </div>
          <div>
            <label>분석 템플릿</label>
            <select id="template">
              <option value="freeform">일반 분석</option>
              <option value="nms_events">NMS/장애 로그</option>
              <option value="maintenance_report">유지보수 보고서</option>
              <option value="customer_history">고객사 이력 요약</option>
            </select>
          </div>
        </div>
        <div class="row">
          <button onclick="saveToken()">토큰 저장</button>
          <button class="secondary" onclick="newConversation()">새 대화</button>
          <button class="secondary" onclick="loadConversations()">대화 새로고침</button>
          <button class="secondary" onclick="preload()">프리로드</button>
          <button class="secondary" onclick="unload()">언로드</button>
          <button class="secondary" onclick="benchmark()">벤치마크</button>
        </div>
      </div>
      <div class="card stack" id="metrics"></div>
      <div class="card stack">
        <div class="row" style="justify-content:space-between">
          <strong>33 NMS 상시 모니터링</strong>
          <span id="nmsMonitorPill" class="pill warn">대기</span>
        </div>
        <div class="grid2">
          <div>
            <label>업체</label>
            <select id="nmsCustomer" onchange="selectNmsCustomer()">
              <option value="">업체 목록 불러오기 필요</option>
            </select>
          </div>
          <div>
            <label>현장</label>
            <select id="nmsSite" onchange="selectNmsSite()">
              <option value="">전체 현장</option>
            </select>
          </div>
        </div>
        <div class="row">
          <button class="secondary" onclick="loadNmsCustomers()">업체 새로고침</button>
          <button class="secondary" onclick="loadNmsContext()">NMS 불러오기</button>
          <button class="secondary" onclick="analyzeNms()">NMS 분석</button>
          <button class="secondary" onclick="toggleNmsMonitor()">상시 시작/중지</button>
        </div>
        <div id="nmsMonitorSummary" class="metric">
          <span>모니터링 상태</span><b>확인 전</b>
        </div>
      </div>
    </section>
    <section class="stack">
      <div class="card stack">
        <div class="grid2">
          <div>
            <label>고객사</label>
            <input id="customer" placeholder="예: (주)농업회사법인돈우" />
          </div>
          <div>
            <label>요청</label>
            <input id="question" placeholder="예: 최근 24시간 장애 가능성을 판단해줘" />
          </div>
        </div>
        <div>
          <label>프롬프트 / 로그 / JSON 데이터</label>
          <textarea id="prompt" placeholder="여기에 NMS 이벤트, 작업일지, 로그, 질문을 넣습니다."></textarea>
        </div>
        <div class="row">
          <button onclick="analyze()">분석 실행</button>
          <button class="secondary" onclick="chat()">일반 채팅</button>
          <button class="secondary" onclick="refresh()">상태 새로고침</button>
        </div>
      </div>
      <div class="card result"><pre id="result">대기 중입니다.</pre></div>
    </section>
  </main>
  <script>
    const $ = (id) => document.getElementById(id);
    const tokenKey = "metro_llm_ops_token";
    const conversationKey = "metro_llm_ops_conversation_id";
    let nmsCustomers = [];
    let nmsMonitorTimer = null;
    let healthTimer = null;
    let conversations = [];
    let currentConversationId = localStorage.getItem(conversationKey) || "";
    $("token").value = localStorage.getItem(tokenKey) || "";

    function saveToken() {
      localStorage.setItem(tokenKey, $("token").value.trim());
      setResult("토큰을 브라우저에 저장했습니다.");
      loadConversations();
      loadNmsCustomers();
      refreshNmsMonitor();
    }

    function headers() {
      const token = $("token").value.trim();
      return {
        "Content-Type": "application/json",
        ...(token ? {"X-LLM-Ops-Token": token} : {}),
      };
    }

    function setResult(value) {
      $("result").textContent = typeof value === "string" ? value : JSON.stringify(value, null, 2);
    }

    async function api(path, options = {}) {
      const res = await fetch(path, options);
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || JSON.stringify(data));
      return data;
    }

    function renderConversations() {
      const select = $("conversationSelect");
      select.innerHTML = '<option value="">새 대화</option>';
      for (const item of conversations) {
        const opt = document.createElement("option");
        opt.value = item.id;
        opt.textContent = `${item.title || "대화"} (${item.message_count || 0})`;
        if (item.id === currentConversationId) opt.selected = true;
        select.appendChild(opt);
      }
    }

    async function loadConversations() {
      try {
        const data = await api("/api/conversations?limit=80", {headers: headers()});
        conversations = data.items || [];
        if (currentConversationId && !conversations.some(c => c.id === currentConversationId)) {
          currentConversationId = "";
          localStorage.removeItem(conversationKey);
        }
        renderConversations();
      } catch (err) {
        console.warn("conversation load failed", err);
      }
    }

    function selectConversation() {
      currentConversationId = $("conversationSelect").value;
      if (currentConversationId) localStorage.setItem(conversationKey, currentConversationId);
      else localStorage.removeItem(conversationKey);
    }

    async function newConversation() {
      const data = await api("/api/conversations", {
        method: "POST",
        headers: headers(),
        body: JSON.stringify({title: $("customer").value || "운영 분석 대화"}),
      });
      currentConversationId = data.conversation.id;
      localStorage.setItem(conversationKey, currentConversationId);
      await loadConversations();
      setResult(`새 대화를 시작했습니다.\nconversation_id=${currentConversationId}`);
    }

    function conversationPayload(extra = {}) {
      return {
        ...extra,
        remember: true,
        ...(currentConversationId ? {conversation_id: currentConversationId} : {}),
      };
    }

    function applyConversationFromResponse(data) {
      if (data.conversation_id) {
        currentConversationId = data.conversation_id;
        localStorage.setItem(conversationKey, currentConversationId);
        loadConversations();
      }
    }

    function renderHealthMetrics(health, runningModels = null) {
      $("statusPill").className = health.ollama.reachable ? "pill" : "pill bad";
      $("statusPill").textContent = health.ollama.reachable ? "Ollama 연결 정상" : "Ollama 연결 실패";
      const gpu = (health.gpu.gpus || [])[0] || {};
      const runningSource = Array.isArray(runningModels) ? runningModels : (health.ollama.running || []);
      const running = runningSource.map(m => m.name).join(", ") || "없음";
      $("metrics").innerHTML = `
        <div class="metric"><span>GPU</span><b>${gpu.name || "미확인"}</b></div>
        <div class="metric"><span>VRAM</span><b>${gpu.memory_used_mib || 0} / ${gpu.memory_total_mib || 0} MiB</b></div>
        <div class="metric"><span>GPU 사용률</span><b>${gpu.utilization_gpu_percent ?? 0}%</b></div>
        <div class="metric"><span>온도</span><b>${gpu.temperature_c ?? "-"}℃</b></div>
        <div class="metric"><span>실행 모델</span><b style="font-size:14px">${running}</b></div>
      `;
    }

    async function refreshHealthOnly() {
      const health = await api("/api/health");
      renderHealthMetrics(health);
    }

    function startHealthAutoRefresh() {
      if (healthTimer) clearInterval(healthTimer);
      healthTimer = setInterval(() => {
        refreshHealthOnly().catch((err) => {
          console.warn("health refresh failed", err);
        });
      }, 10000);
    }

    async function refresh() {
      try {
        const health = await api("/api/health");
        const models = await api("/api/models");
        $("model").innerHTML = "";
        for (const m of models.models || []) {
          const opt = document.createElement("option");
          opt.value = m.name;
          opt.textContent = m.name;
          if (m.name === models.default_model) opt.selected = true;
          $("model").appendChild(opt);
        }
        renderHealthMetrics(health, models.running || []);
      } catch (err) {
        $("statusPill").className = "pill bad";
        $("statusPill").textContent = "상태 확인 실패";
        setResult(String(err));
      }
    }

    function renderNmsMonitor(state) {
      const summary = state.summary || {};
      const risk = summary.risk || "unknown";
      $("nmsMonitorPill").className = risk === "critical" ? "pill bad" : (risk === "watch" ? "pill warn" : "pill");
      $("nmsMonitorPill").textContent = state.ok ? `NMS ${risk}` : "NMS 연결 확인 필요";
      $("nmsMonitorSummary").innerHTML = `
        <span>최근 ${summary.window_hours || "-"}h / 최종 ${state.last_checked_at || "-"}</span>
        <b style="font-size:14px">
          고객 ${summary.customer_count || 0} · 현장 ${summary.site_count || 0} · 이벤트 ${summary.recent_event_count || 0}
          · Down ${summary.down_device_count || 0} · Degraded ${summary.degraded_device_count || 0}
        </b>
      `;
    }

    async function refreshNmsMonitor() {
      try {
        const data = await api("/api/nms/monitor/status", {headers: headers()});
        renderNmsMonitor(data);
        if ((!nmsCustomers.length) && data.customers && data.customers.length) {
          nmsCustomers = data.customers;
          renderNmsCustomers();
        }
      } catch (err) {
        $("nmsMonitorPill").className = "pill bad";
        $("nmsMonitorPill").textContent = "NMS 인증/연결 실패";
        $("nmsMonitorSummary").innerHTML = `<span>오류</span><b style="font-size:14px">${String(err)}</b>`;
      }
    }

    function renderNmsCustomers() {
      const select = $("nmsCustomer");
      select.innerHTML = '<option value="">업체 선택</option>';
      for (const customer of nmsCustomers) {
        const opt = document.createElement("option");
        opt.value = customer.customer_name;
        opt.textContent = `${customer.customer_name} (${customer.site_count || 0}현장 / ${customer.recent_event_count || 0}이벤트)`;
        opt.dataset.customerCode = customer.customer_code || "";
        select.appendChild(opt);
      }
      selectNmsCustomer();
    }

    async function loadNmsCustomers() {
      try {
        const data = await api("/api/nms/customers?hours=24&limit=500", {headers: headers()});
        nmsCustomers = data.customers || [];
        renderNmsCustomers();
        renderNmsMonitor({
          ok: true,
          last_checked_at: data.generated_at,
          summary: {
            risk: "watch",
            window_hours: data.requested?.hours,
            customer_count: data.count,
            site_count: data.site_count,
            recent_event_count: nmsCustomers.reduce((sum, c) => sum + (c.recent_event_count || 0), 0),
            down_device_count: nmsCustomers.reduce((sum, c) => sum + (c.down_device_count || 0), 0),
            degraded_device_count: nmsCustomers.reduce((sum, c) => sum + (c.degraded_device_count || 0), 0),
          },
        });
      } catch (err) {
        setResult(`업체 목록을 불러오지 못했습니다.\n${String(err)}`);
      }
    }

    function selectedCustomer() {
      const name = $("nmsCustomer").value;
      return nmsCustomers.find(c => c.customer_name === name) || null;
    }

    function selectNmsCustomer() {
      const customer = selectedCustomer();
      const siteSelect = $("nmsSite");
      siteSelect.innerHTML = '<option value="">전체 현장</option>';
      if (!customer) return;
      $("customer").value = customer.customer_name;
      for (const site of customer.sites || []) {
        const opt = document.createElement("option");
        opt.value = site.site_code;
        opt.textContent = `${site.site_name} / ${site.site_code} (${site.recent_event_count || 0})`;
        siteSelect.appendChild(opt);
      }
    }

    function selectNmsSite() {
      const customer = selectedCustomer();
      if (customer) $("customer").value = customer.customer_name;
    }

    async function loadNmsContext() {
      const customer = selectedCustomer();
      const siteCode = $("nmsSite").value;
      if (!customer && !siteCode) {
        setResult("업체 또는 현장을 먼저 선택하세요.");
        return;
      }
      $("template").value = "nms_events";
      $("question").value ||= "최근 48시간 기준으로 새벽 이벤트, 장애 징후, 보안 위험, 조치 우선순위를 근거 중심으로 판단해줘.";
      const query = new URLSearchParams({
        customer_name: customer?.customer_name || "",
        site_codes: siteCode || (customer?.sites || []).map(s => s.site_code).join(","),
        hours: "48",
        limit: "80",
      });
      const data = await api(`/api/nms/context?${query.toString()}`, {headers: headers()});
      $("prompt").value = JSON.stringify(data, null, 2);
      setResult({
        matched: data.matched,
        event_count: (data.logs?.syslog_count || 0) + (data.logs?.trap_count || 0),
        traffic_count: data.traffic?.interface_count || 0,
        off_hours_events: data.temporal?.off_hours?.total || 0,
        nas_file_risk: data.temporal?.nas_file_risk?.risk || "unknown",
      });
    }

    async function analyzeNms() {
      const customer = selectedCustomer();
      const siteCode = $("nmsSite").value;
      if (!customer && !siteCode) {
        setResult("업체 또는 현장을 먼저 선택하세요.");
        return;
      }
      setResult("33 NMS 데이터 분석 중...");
      const data = await api("/api/nms/analyze", {
        method: "POST",
        headers: headers(),
        body: JSON.stringify(conversationPayload({
          model: $("model").value,
          customer_name: customer?.customer_name || "",
          site_codes: siteCode || (customer?.sites || []).map(s => s.site_code),
          question: $("question").value,
          hours: 48,
          limit: 80,
          depth: "deep",
          source: "nms-analysis",
        })),
      });
      applyConversationFromResponse(data);
      setResult(data.response || data);
      refresh();
    }

    function toggleNmsMonitor() {
      if (nmsMonitorTimer) {
        clearInterval(nmsMonitorTimer);
        nmsMonitorTimer = null;
        $("nmsMonitorPill").textContent = "상시 중지";
        return;
      }
      refreshNmsMonitor();
      nmsMonitorTimer = setInterval(refreshNmsMonitor, 60000);
      $("nmsMonitorPill").textContent = "상시 실행";
    }

    async function chat() {
      setResult("생성 중...");
      const data = await api("/api/chat", {
        method: "POST",
        headers: headers(),
        body: JSON.stringify(conversationPayload({
          model: $("model").value,
          prompt: $("prompt").value,
          temperature: 0.2,
          source: "free-chat",
        })),
      });
      applyConversationFromResponse(data);
      setResult(data.response || data);
      refresh();
    }

    async function analyze() {
      setResult("분석 중...");
      const data = await api("/api/analyze", {
        method: "POST",
        headers: headers(),
        body: JSON.stringify(conversationPayload({
          model: $("model").value,
          template: $("template").value,
          customer: $("customer").value,
          question: $("question").value,
          data: $("prompt").value,
          temperature: 0.15,
          source: "template-analysis",
        })),
      });
      applyConversationFromResponse(data);
      setResult(data.response || data);
      refresh();
    }

    async function preload() {
      setResult("모델 프리로드 중...");
      const data = await api("/api/preload", {
        method: "POST",
        headers: headers(),
        body: JSON.stringify({model: $("model").value}),
      });
      setResult(data);
      refresh();
    }

    async function unload() {
      setResult("모델 언로드 중...");
      const data = await api("/api/unload", {
        method: "POST",
        headers: headers(),
        body: JSON.stringify({model: $("model").value}),
      });
      setResult(data);
      refresh();
    }

    async function benchmark() {
      setResult("벤치마크 중...");
      const data = await api("/api/benchmark", {
        method: "POST",
        headers: headers(),
        body: JSON.stringify({model: $("model").value}),
      });
      setResult(`응답시간: ${data.elapsed_ms}ms\n\n${data.response || ""}`);
      refresh();
    }

    refresh();
    startHealthAutoRefresh();
    loadConversations();
    refreshNmsMonitor();
    loadNmsCustomers();
    toggleNmsMonitor();
  </script>
</body>
</html>
"""


def main() -> None:
    init_conversation_store()
    NMS_MONITOR.start()
    NMS_AUTOPILOT.start()
    httpd = ThreadingHTTPServer((HOST, PORT), Handler)
    print(f"{utc_now()} Metro LLM Ops listening on http://{HOST}:{PORT}", flush=True)
    httpd.serve_forever()


if __name__ == "__main__":
    main()
