#!/usr/bin/env python3
"""Small LLM operations console for the 118 GPU server.

This service intentionally avoids third-party Python dependencies. It proxies
Ollama operations, exposes GPU/runtime health, and provides a simple internal
web UI for operators and future ERP/NMS integrations.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import base64
import html
import re
import shutil
import socket
import sqlite3
import subprocess
import tempfile
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
import xml.etree.ElementTree as ET
import zipfile
from collections import Counter, OrderedDict
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
OLLAMA_ENABLED = os.getenv("OLLAMA_ENABLED", "true").strip().lower() not in {"0", "false", "no", "off"}
OLLAMA_CPU_ONLY_MODELS_RAW = os.getenv("LLM_OPS_OLLAMA_CPU_ONLY_MODELS", "qwen3:14b")
OLLAMA_KEEP_SGLANG_MODELS_RAW = os.getenv("LLM_OPS_OLLAMA_KEEP_SGLANG_MODELS", "qwen3:14b")
OLLAMA_MODEL_NUM_CTX_RAW = os.getenv("LLM_OPS_OLLAMA_MODEL_NUM_CTX", "qwen3:14b=8192")
OLLAMA_MODEL_NUM_GPU_RAW = os.getenv("LLM_OPS_OLLAMA_MODEL_NUM_GPU", "qwen3:14b=0")
OLLAMA_MODEL_NUM_THREAD_RAW = os.getenv("LLM_OPS_OLLAMA_MODEL_NUM_THREAD", "qwen3:14b=6")
OLLAMA_MODEL_NUM_PREDICT_RAW = os.getenv("LLM_OPS_OLLAMA_MODEL_NUM_PREDICT", "qwen3:14b=1400")
OLLAMA_DISABLE_THINK_MODELS_RAW = os.getenv("LLM_OPS_OLLAMA_DISABLE_THINK_MODELS", "qwen3:14b")
SGLANG_ENABLED = os.getenv("SGLANG_ENABLED", "true").strip().lower() not in {"0", "false", "no", "off"}
SGLANG_BASE_URL = os.getenv("SGLANG_BASE_URL", "http://127.0.0.1:30000").rstrip("/")
SGLANG_MODEL = os.getenv("SGLANG_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
SGLANG_DEFAULT_AVAILABLE_MODELS = f"{SGLANG_MODEL},Qwen/Qwen3-4B-Instruct-2507"
SGLANG_AVAILABLE_MODELS_RAW = os.getenv("SGLANG_AVAILABLE_MODELS", SGLANG_DEFAULT_AVAILABLE_MODELS)
SGLANG_MODEL_CONTEXT_LENGTHS_RAW = os.getenv(
    "SGLANG_MODEL_CONTEXT_LENGTHS",
    "Qwen/Qwen2.5-0.5B-Instruct=16384,Qwen/Qwen3-4B-Instruct-2507=32768",
)
SGLANG_MODEL_MEM_FRACTIONS_RAW = os.getenv(
    "SGLANG_MODEL_MEM_FRACTIONS",
    "Qwen/Qwen2.5-0.5B-Instruct=0.85,Qwen/Qwen3-4B-Instruct-2507=0.82",
)
SGLANG_AUTO_SWITCH_ENABLED = os.getenv("SGLANG_AUTO_SWITCH_ENABLED", "false").strip().lower() in {"1", "true", "yes", "on"}
SGLANG_REQUEST_TIMEOUT = int(os.getenv("SGLANG_REQUEST_TIMEOUT_SECONDS", "600"))
SGLANG_START_SCRIPT = os.getenv("SGLANG_START_SCRIPT", "/home/metroai/sglang-runtime/start-sglang.sh")
SGLANG_STOP_SCRIPT = os.getenv("SGLANG_STOP_SCRIPT", "/home/metroai/sglang-runtime/stop-sglang.sh")
SGLANG_START_WAIT_SECONDS = int(os.getenv("SGLANG_START_WAIT_SECONDS", "90"))
SGLANG_RUNTIME_ENV_FILE = Path(os.getenv("SGLANG_RUNTIME_ENV_FILE", "/home/metroai/sglang-runtime/sglang.env"))
API_TOKEN = os.getenv("LLM_OPS_TOKEN", "")
DEFAULT_MODEL = os.getenv("LLM_OPS_DEFAULT_MODEL", "metro-report:latest")
FAST_MODEL = os.getenv("LLM_OPS_FAST_MODEL", "metro-fast:latest")
MODEL_ALIASES_RAW = os.getenv(
    "LLM_OPS_MODEL_ALIASES",
    f"metro-report:latest={SGLANG_MODEL},metro-fast:latest={SGLANG_MODEL}",
)
PUBLIC_MODEL_IDS_RAW = os.getenv("LLM_OPS_PUBLIC_MODELS", "")
DEFAULT_CHAT_SYSTEM_PROMPT = os.getenv(
    "LLM_OPS_DEFAULT_CHAT_SYSTEM_PROMPT",
    "기본 응답 언어는 한국어다. 사용자가 다른 언어를 명시적으로 요청하지 않으면 자연스럽고 간결한 한국어로 답해라.",
)
KOREAN_RESPONSE_GUARD = (
    "중요: 사용자가 다른 언어를 명시적으로 요청하지 않으면 반드시 한국어로만 답한다. "
    "중국어, 영어, 일본어로 시작하지 말고 첫 문장부터 한국어로 작성한다."
)
REQUEST_TIMEOUT = int(os.getenv("LLM_OPS_REQUEST_TIMEOUT_SECONDS", "600"))
MAX_BODY_BYTES = int(os.getenv("LLM_OPS_MAX_BODY_BYTES", str(16 * 1024 * 1024)))
KEEP_ALIVE = os.getenv("LLM_OPS_KEEP_ALIVE", "30m")
LLM_OPS_DEFAULT_NUM_CTX = int(os.getenv("LLM_OPS_DEFAULT_NUM_CTX", "16384"))
ALLOWED_ORIGIN = os.getenv("LLM_OPS_ALLOWED_ORIGIN", "*")
MODEL_FALLBACK_ENABLED = os.getenv("LLM_OPS_MODEL_FALLBACK_ENABLED", "true").strip().lower() not in {"0", "false", "no", "off"}
MODEL_RUNNING_SWITCH_GUARD_ENABLED = os.getenv(
    "LLM_OPS_MODEL_RUNNING_SWITCH_GUARD_ENABLED",
    "false",
).strip().lower() not in {"0", "false", "no", "off"}
MODEL_AUTO_UNLOAD_BEFORE_SWITCH_ENABLED = os.getenv(
    "LLM_OPS_MODEL_AUTO_UNLOAD_BEFORE_SWITCH_ENABLED",
    "true",
).strip().lower() not in {"0", "false", "no", "off"}
MODEL_SWITCH_UNLOAD_TIMEOUT = int(os.getenv("LLM_OPS_MODEL_SWITCH_UNLOAD_TIMEOUT_SECONDS", "60"))
KOREAN_RETRY_ENABLED = os.getenv("LLM_OPS_KOREAN_RETRY_ENABLED", "true").strip().lower() not in {"0", "false", "no", "off"}
NMS_CONTEXT_BASE_URL = os.getenv("NMS_CONTEXT_BASE_URL", "http://192.168.1.33:7443").rstrip("/")
NMS_CONTEXT_TOKEN = os.getenv("NMS_CONTEXT_TOKEN", "")
NMS_CONTEXT_TIMEOUT = int(os.getenv("NMS_CONTEXT_TIMEOUT_SECONDS", "12"))
NMS_CONTEXT_CACHE_ENABLED = os.getenv("NMS_CONTEXT_CACHE_ENABLED", "true").strip().lower() not in {"0", "false", "no", "off"}
NMS_CONTEXT_CACHE_TTL_SECONDS = int(os.getenv("NMS_CONTEXT_CACHE_TTL_SECONDS", "30"))
NMS_CONTEXT_CACHE_MAX_ITEMS = int(os.getenv("NMS_CONTEXT_CACHE_MAX_ITEMS", "64"))
NMS_MONITOR_ENABLED = os.getenv("NMS_MONITOR_ENABLED", "true").strip().lower() not in {"0", "false", "no", "off"}
NMS_MONITOR_INTERVAL = int(os.getenv("NMS_MONITOR_INTERVAL_SECONDS", "60"))
NMS_MONITOR_WINDOW_HOURS = int(os.getenv("NMS_MONITOR_WINDOW_HOURS", "24"))
NMS_MONITOR_LIMIT = int(os.getenv("NMS_MONITOR_LIMIT", "500"))
NMS_DEEP_ANALYSIS_WINDOW_HOURS = int(os.getenv("NMS_DEEP_ANALYSIS_WINDOW_HOURS", "48"))
NMS_DEEP_ANALYSIS_LIMIT = int(os.getenv("NMS_DEEP_ANALYSIS_LIMIT", "80"))
NMS_ANALYSIS_CONTEXT_CHAR_LIMIT = int(os.getenv("NMS_ANALYSIS_CONTEXT_CHAR_LIMIT", "90000"))
NMS_LLM_EVIDENCE_CHAR_LIMIT = int(os.getenv("NMS_LLM_EVIDENCE_CHAR_LIMIT", "25000"))
LLM_OPS_MAX_PROMPT_CHARS = int(os.getenv("LLM_OPS_MAX_PROMPT_CHARS", "50000"))
LLM_OPS_MAX_SINGLE_MESSAGE_CHARS = int(os.getenv("LLM_OPS_MAX_SINGLE_MESSAGE_CHARS", "30000"))
NMS_ANALYSIS_TIMEOUT = int(os.getenv("NMS_ANALYSIS_TIMEOUT_SECONDS", "900"))
NMS_ANALYSIS_NUM_CTX = int(os.getenv("NMS_ANALYSIS_NUM_CTX", "16384"))
NMS_ANALYSIS_NUM_PREDICT = int(os.getenv("NMS_ANALYSIS_NUM_PREDICT", "3072"))
NMS_ANALYSIS_REPEAT_PENALTY = float(os.getenv("NMS_ANALYSIS_REPEAT_PENALTY", "1.08"))
NMS_FAST_ANALYSIS_MODEL = os.getenv("NMS_FAST_ANALYSIS_MODEL", FAST_MODEL)
NMS_DEEP_ANALYSIS_MODEL = os.getenv("NMS_DEEP_ANALYSIS_MODEL", DEFAULT_MODEL)
NMS_FAST_ANALYSIS_NUM_CTX = int(os.getenv("NMS_FAST_ANALYSIS_NUM_CTX", "8192"))
NMS_FAST_ANALYSIS_NUM_PREDICT = int(os.getenv("NMS_FAST_ANALYSIS_NUM_PREDICT", "900"))
NMS_FAST_ANALYSIS_TIMEOUT = int(os.getenv("NMS_FAST_ANALYSIS_TIMEOUT_SECONDS", "240"))
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
NMS_AUTOPILOT_NUM_CTX = int(os.getenv("NMS_AUTOPILOT_NUM_CTX", "16384"))
NMS_AUTOPILOT_NUM_PREDICT = int(os.getenv("NMS_AUTOPILOT_NUM_PREDICT", "1200"))
CONVERSATION_DB_PATH = Path(os.getenv("LLM_OPS_CONVERSATION_DB", str(APP_DIR / "data" / "conversations.sqlite3")))
CONVERSATION_HISTORY_LIMIT = int(os.getenv("LLM_OPS_CONVERSATION_HISTORY_LIMIT", "18"))
CONVERSATION_MESSAGE_CHAR_LIMIT = int(os.getenv("LLM_OPS_CONVERSATION_MESSAGE_CHAR_LIMIT", "16000"))
CONVERSATION_TITLE_CHAR_LIMIT = int(os.getenv("LLM_OPS_CONVERSATION_TITLE_CHAR_LIMIT", "80"))
SAVED_ANALYSIS_CONTEXT_LIMIT = int(os.getenv("LLM_OPS_SAVED_ANALYSIS_CONTEXT_LIMIT", "4"))
SAVED_ANALYSIS_CONTENT_CHAR_LIMIT = int(os.getenv("LLM_OPS_SAVED_ANALYSIS_CONTENT_CHAR_LIMIT", "24000"))
SAVED_ANALYSIS_SOURCE_CHAR_LIMIT = int(os.getenv("LLM_OPS_SAVED_ANALYSIS_SOURCE_CHAR_LIMIT", "10000"))
SAVED_ANALYSIS_PREVIEW_CHAR_LIMIT = int(os.getenv("LLM_OPS_SAVED_ANALYSIS_PREVIEW_CHAR_LIMIT", "180"))
NMS_EVIDENCE_STORE_ENABLED = os.getenv("NMS_EVIDENCE_STORE_ENABLED", "true").strip().lower() not in {"0", "false", "no", "off"}
NMS_EVIDENCE_STORE_MAX_FULL_CHARS = int(os.getenv("NMS_EVIDENCE_STORE_MAX_FULL_CHARS", "1200000"))
NMS_EVIDENCE_STORE_CONTEXT_LIMIT = int(os.getenv("NMS_EVIDENCE_STORE_CONTEXT_LIMIT", "12000"))
NMS_EVIDENCE_STORE_RECENT_LIMIT = int(os.getenv("NMS_EVIDENCE_STORE_RECENT_LIMIT", "5"))
NMS_EVIDENCE_STORE_MAX_ITEMS_PER_SCOPE = int(os.getenv("NMS_EVIDENCE_STORE_MAX_ITEMS_PER_SCOPE", "80"))
ATTACHMENT_MAX_COUNT = int(os.getenv("LLM_OPS_ATTACHMENT_MAX_COUNT", "5"))
ATTACHMENT_MAX_FILE_BYTES = int(os.getenv("LLM_OPS_ATTACHMENT_MAX_FILE_BYTES", str(5 * 1024 * 1024)))
ATTACHMENT_MAX_TOTAL_BYTES = int(os.getenv("LLM_OPS_ATTACHMENT_MAX_TOTAL_BYTES", str(12 * 1024 * 1024)))
ATTACHMENT_TEXT_CHAR_LIMIT = int(os.getenv("LLM_OPS_ATTACHMENT_TEXT_CHAR_LIMIT", "8000"))
ATTACHMENT_TOTAL_TEXT_CHAR_LIMIT = int(os.getenv("LLM_OPS_ATTACHMENT_TOTAL_TEXT_CHAR_LIMIT", "28000"))
ATTACHMENT_CSV_ROW_LIMIT = int(os.getenv("LLM_OPS_ATTACHMENT_CSV_ROW_LIMIT", "120"))
ATTACHMENT_TABLE_COLUMN_LIMIT = int(os.getenv("LLM_OPS_ATTACHMENT_TABLE_COLUMN_LIMIT", "20"))
LARGE_LOG_DIR = Path(os.getenv("LLM_OPS_LARGE_LOG_DIR", "/mnt/llm-data/llm-ops/data/large-logs"))
LARGE_LOG_MAX_FILE_BYTES = int(os.getenv("LLM_OPS_LARGE_LOG_MAX_FILE_BYTES", str(64 * 1024 * 1024)))
LARGE_LOG_MAX_TOTAL_BYTES = int(os.getenv("LLM_OPS_LARGE_LOG_MAX_TOTAL_BYTES", str(128 * 1024 * 1024)))
LARGE_LOG_CHUNK_CHARS = int(os.getenv("LLM_OPS_LARGE_LOG_CHUNK_CHARS", "12000"))
LARGE_LOG_MAX_CHUNKS = int(os.getenv("LLM_OPS_LARGE_LOG_MAX_CHUNKS", "32"))
LARGE_LOG_SUMMARY_CHAR_LIMIT = int(os.getenv("LLM_OPS_LARGE_LOG_SUMMARY_CHAR_LIMIT", "60000"))
LARGE_LOG_ANALYSIS_TIMEOUT = int(os.getenv("LLM_OPS_LARGE_LOG_ANALYSIS_TIMEOUT_SECONDS", "1200"))
LARGE_LOG_NUM_PREDICT = int(os.getenv("LLM_OPS_LARGE_LOG_NUM_PREDICT", "2200"))
OPENAI_COMPAT_OBJECT = "chat.completion"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


CONVERSATION_LOCK = threading.Lock()
SGLANG_SWITCH_LOCK = threading.Lock()
NMS_CONTEXT_CACHE_LOCK = threading.Lock()
NMS_CONTEXT_CACHE: OrderedDict[str, tuple[float, dict[str, Any]]] = OrderedDict()
NMS_CONTEXT_CACHE_STATS = {
    "hits": 0,
    "misses": 0,
    "stores": 0,
    "evictions": 0,
}


def clip_text(value: Any, limit: int = CONVERSATION_MESSAGE_CHAR_LIMIT) -> str:
    text = str(value or "")
    if len(text) <= limit:
        return text
    return f"{text[:limit].rstrip()}\n\n...[truncated {len(text) - limit} chars]"


def clip_middle_text(value: Any, limit: int = CONVERSATION_MESSAGE_CHAR_LIMIT) -> str:
    text = str(value or "")
    if len(text) <= limit:
        return text
    if limit <= 200:
        return clip_text(text, limit)
    head = max(80, int(limit * 0.62))
    tail = max(80, limit - head - 80)
    omitted = len(text) - head - tail
    return f"{text[:head].rstrip()}\n\n...[middle truncated {omitted} chars]...\n\n{text[-tail:].lstrip()}"


def parse_csv_values(value: str) -> list[str]:
    items: list[str] = []
    for raw_item in str(value or "").split(","):
        item = raw_item.strip()
        if item and item not in items:
            items.append(item)
    return items


def parse_model_int_map(value: str) -> dict[str, int]:
    parsed: dict[str, int] = {}
    for raw_item in str(value or "").split(","):
        if "=" not in raw_item:
            continue
        key, raw_value = raw_item.split("=", 1)
        try:
            parsed[key.strip()] = int(raw_value.strip())
        except Exception:
            continue
    return parsed


def parse_model_float_map(value: str) -> dict[str, float]:
    parsed: dict[str, float] = {}
    for raw_item in str(value or "").split(","):
        if "=" not in raw_item:
            continue
        key, raw_value = raw_item.split("=", 1)
        try:
            parsed[key.strip()] = float(raw_value.strip())
        except Exception:
            continue
    return parsed


def parse_model_alias_map(value: str) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for raw_item in str(value or "").split(","):
        if "=" not in raw_item:
            continue
        alias, target = raw_item.split("=", 1)
        alias = alias.strip()
        target = target.strip()
        if alias and target:
            parsed[alias] = target
    return parsed


SGLANG_AVAILABLE_MODELS = parse_csv_values(SGLANG_AVAILABLE_MODELS_RAW)
if SGLANG_MODEL and SGLANG_MODEL not in SGLANG_AVAILABLE_MODELS:
    SGLANG_AVAILABLE_MODELS.insert(0, SGLANG_MODEL)
SGLANG_MODEL_CONTEXT_LENGTHS = parse_model_int_map(SGLANG_MODEL_CONTEXT_LENGTHS_RAW)
SGLANG_MODEL_MEM_FRACTIONS = parse_model_float_map(SGLANG_MODEL_MEM_FRACTIONS_RAW)
MODEL_ALIASES = parse_model_alias_map(MODEL_ALIASES_RAW)
PUBLIC_MODEL_IDS = parse_csv_values(PUBLIC_MODEL_IDS_RAW)
OLLAMA_CPU_ONLY_MODELS = set(parse_csv_values(OLLAMA_CPU_ONLY_MODELS_RAW))
OLLAMA_KEEP_SGLANG_MODELS = set(parse_csv_values(OLLAMA_KEEP_SGLANG_MODELS_RAW))
OLLAMA_MODEL_NUM_CTX = parse_model_int_map(OLLAMA_MODEL_NUM_CTX_RAW)
OLLAMA_MODEL_NUM_GPU = parse_model_int_map(OLLAMA_MODEL_NUM_GPU_RAW)
OLLAMA_MODEL_NUM_THREAD = parse_model_int_map(OLLAMA_MODEL_NUM_THREAD_RAW)
OLLAMA_MODEL_NUM_PREDICT = parse_model_int_map(OLLAMA_MODEL_NUM_PREDICT_RAW)
OLLAMA_DISABLE_THINK_MODELS = set(parse_csv_values(OLLAMA_DISABLE_THINK_MODELS_RAW))


def sanitize_attachment_name(value: Any) -> str:
    name = Path(str(value or "attachment").strip() or "attachment").name
    return clip_text(name.replace("\x00", ""), 180)


def attachment_extension(name: str) -> str:
    return Path(name).suffix.lower()


def decode_text_bytes(raw: bytes) -> str:
    for encoding in ("utf-8-sig", "utf-8", "cp949", "euc-kr"):
        try:
            return raw.decode(encoding)
        except Exception:
            continue
    return raw.decode("utf-8", errors="replace")


def extract_csv_text(raw: bytes) -> str:
    text = decode_text_bytes(raw)
    reader = csv.reader(io.StringIO(text))
    rows: list[str] = []
    for index, row in enumerate(reader):
        if index >= ATTACHMENT_CSV_ROW_LIMIT:
            rows.append(f"...[rows truncated after {ATTACHMENT_CSV_ROW_LIMIT}]")
            break
        normalized = [cell.strip() for cell in row[:ATTACHMENT_TABLE_COLUMN_LIMIT]]
        rows.append(" | ".join(normalized))
    return "\n".join(rows) if rows else text


def xml_text(node: ET.Element) -> str:
    return "".join(part for part in node.itertext() if part)


def extract_docx_text(raw: bytes) -> str:
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        xml_bytes = archive.read("word/document.xml")
    root = ET.fromstring(xml_bytes)
    ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
    lines: list[str] = []
    for paragraph in root.findall(".//w:p", ns):
        paragraph_text = "".join(
            text for text in (node.text or "" for node in paragraph.findall(".//w:t", ns)) if text
        ).strip()
        if paragraph_text:
            lines.append(html.unescape(paragraph_text))
    return "\n".join(lines)


def extract_xlsx_text(raw: bytes) -> str:
    ns = {
        "main": "http://schemas.openxmlformats.org/spreadsheetml/2006/main",
        "rel": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
        "pkg": "http://schemas.openxmlformats.org/package/2006/relationships",
    }
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        shared_strings: list[str] = []
        if "xl/sharedStrings.xml" in archive.namelist():
            shared_root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
            for item in shared_root.findall(".//main:si", ns):
                shared_strings.append("".join(text for text in item.itertext() if text))

        workbook_root = ET.fromstring(archive.read("xl/workbook.xml"))
        rel_root = ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
        rel_map = {
            rel.attrib.get("Id"): rel.attrib.get("Target", "")
            for rel in rel_root.findall(".//pkg:Relationship", ns)
        }

        blocks: list[str] = []
        for sheet_index, sheet in enumerate(workbook_root.findall(".//main:sheets/main:sheet", ns), start=1):
            if sheet_index > 4:
                blocks.append("...[sheets truncated]")
                break
            rel_id = sheet.attrib.get("{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id")
            target = rel_map.get(rel_id, "")
            if not target:
                continue
            sheet_path = f"xl/{target}" if not target.startswith("xl/") else target
            sheet_root = ET.fromstring(archive.read(sheet_path))
            sheet_name = sheet.attrib.get("name") or f"Sheet{sheet_index}"
            rows: list[str] = [f"시트: {sheet_name}"]
            for row_index, row in enumerate(sheet_root.findall(".//main:sheetData/main:row", ns), start=1):
                if row_index > ATTACHMENT_CSV_ROW_LIMIT:
                    rows.append(f"...[rows truncated after {ATTACHMENT_CSV_ROW_LIMIT}]")
                    break
                values: list[str] = []
                for cell_index, cell in enumerate(row.findall("main:c", ns), start=1):
                    if cell_index > ATTACHMENT_TABLE_COLUMN_LIMIT:
                        values.append("...[cols truncated]")
                        break
                    cell_type = cell.attrib.get("t") or ""
                    value = ""
                    if cell_type == "inlineStr":
                        value = "".join(text for text in cell.itertext() if text)
                    else:
                        raw_value = cell.findtext("main:v", default="", namespaces=ns)
                        if cell_type == "s" and raw_value.isdigit():
                            index = int(raw_value)
                            value = shared_strings[index] if index < len(shared_strings) else raw_value
                        else:
                            value = raw_value
                    value = value.strip()
                    if value or values:
                        values.append(value)
                if any(item.strip() for item in values):
                    rows.append(" | ".join(values))
            blocks.append("\n".join(rows))
        return "\n\n".join(blocks)


def fallback_strings_text(raw: bytes, suffix: str) -> str:
    strings_cmd = shutil.which("strings")
    if not strings_cmd:
        raise RuntimeError("strings command not found")
    with tempfile.TemporaryDirectory(prefix="llmops-strings-") as tempdir:
        source_path = Path(tempdir) / f"attachment{suffix or '.bin'}"
        source_path.write_bytes(raw)
        proc = subprocess.run(
            [strings_cmd, "-n", "4", str(source_path)],
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
        if proc.returncode != 0:
            raise RuntimeError(proc.stderr.strip() or "strings extraction failed")
        return proc.stdout.strip()


def libreoffice_convert(raw: bytes, suffix: str, target_ext: str, filter_name: str = "") -> str:
    libreoffice_cmd = shutil.which("libreoffice") or shutil.which("soffice")
    if not libreoffice_cmd:
        raise RuntimeError("libreoffice not installed")
    with tempfile.TemporaryDirectory(prefix="llmops-lo-") as tempdir:
        temp_path = Path(tempdir)
        source_path = temp_path / f"attachment{suffix or '.bin'}"
        source_path.write_bytes(raw)
        convert_arg = f"{target_ext}:{filter_name}" if filter_name else target_ext
        proc = subprocess.run(
            [
                libreoffice_cmd,
                "--headless",
                "--convert-to",
                convert_arg,
                "--outdir",
                str(temp_path),
                str(source_path),
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=90,
        )
        output_path = source_path.with_suffix(f".{target_ext}")
        if not output_path.exists():
            raise RuntimeError(proc.stderr.strip() or proc.stdout.strip() or "libreoffice conversion failed")
        return output_path.read_text(encoding="utf-8", errors="replace")


def libreoffice_text(raw: bytes, suffix: str) -> str:
    return libreoffice_convert(raw, suffix, "txt", "Text")


def libreoffice_spreadsheet_text(raw: bytes, suffix: str) -> str:
    converted = libreoffice_convert(raw, suffix, "csv", "Text - txt - csv (StarCalc)")
    return extract_csv_text(converted.encode("utf-8"))


def extract_attachment_text(name: str, raw: bytes) -> tuple[str, str]:
    suffix = attachment_extension(name)
    if suffix in {".txt", ".log", ".md", ".json"}:
        return decode_text_bytes(raw), "text"
    if suffix in {".csv", ".tsv"}:
        return extract_csv_text(raw), "csv"
    if suffix == ".docx":
        return extract_docx_text(raw), "docx"
    if suffix == ".xlsx":
        return extract_xlsx_text(raw), "xlsx"
    if suffix in {".doc", ".xls"}:
        try:
            if suffix == ".xls":
                return libreoffice_spreadsheet_text(raw, suffix), "libreoffice-csv"
            return libreoffice_text(raw, suffix), "libreoffice"
        except Exception:
            return fallback_strings_text(raw, suffix), "strings"
    raise ValueError(f"unsupported attachment type: {suffix or 'unknown'}")


def extract_attachments_context(attachments: Any) -> dict[str, Any]:
    if not attachments:
        return {"text": "", "items": [], "errors": [], "total_bytes": 0}
    if not isinstance(attachments, list):
        raise ValueError("attachments must be a list")
    if len(attachments) > ATTACHMENT_MAX_COUNT:
        raise ValueError(f"too many attachments: max {ATTACHMENT_MAX_COUNT}")

    total_bytes = 0
    items: list[dict[str, Any]] = []
    errors: list[str] = []
    blocks: list[str] = []
    for index, attachment in enumerate(attachments, start=1):
        if not isinstance(attachment, dict):
            errors.append(f"attachment #{index}: invalid payload")
            continue
        name = sanitize_attachment_name(attachment.get("name") or f"attachment-{index}")
        encoded = str(attachment.get("content_base64") or "").strip()
        if not encoded:
            errors.append(f"{name}: empty content")
            continue
        try:
            raw = base64.b64decode(encoded, validate=True)
        except Exception:
            errors.append(f"{name}: invalid base64")
            continue
        size = len(raw)
        total_bytes += size
        if size > ATTACHMENT_MAX_FILE_BYTES:
            errors.append(f"{name}: file too large ({size} bytes)")
            continue
        if total_bytes > ATTACHMENT_MAX_TOTAL_BYTES:
            errors.append(f"{name}: total attachment size exceeded")
            continue
        suffix = attachment_extension(name)
        try:
            if suffix in {".txt", ".log", ".md", ".json", ".csv", ".tsv"}:
                text, method = decode_text_bytes(raw), "raw-text"
            else:
                text, method = extract_attachment_text(name, raw)
        except Exception as exc:
            errors.append(f"{name}: {exc}")
            continue
        clipped = clip_text(text.strip(), ATTACHMENT_TEXT_CHAR_LIMIT)
        if not clipped:
            errors.append(f"{name}: extracted text is empty")
            continue
        item = {
            "name": name,
            "size": size,
            "type": str(attachment.get("type") or ""),
            "method": method,
            "preview": clip_text(clipped, 240),
        }
        items.append(item)
        blocks.append(
            f"[첨부 {index}] {name} / size={size} bytes / method={method}\n{clipped}"
        )
    combined = ""
    if blocks:
        combined = "첨부자료 추출본\n\n" + "\n\n".join(blocks)
        combined = clip_text(combined, ATTACHMENT_TOTAL_TEXT_CHAR_LIMIT)
    return {"text": combined, "items": items, "errors": errors, "total_bytes": total_bytes}


IP_PATTERN = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
MAC_PATTERN = re.compile(r"\b[0-9A-Fa-f]{2}(?::|-){1}[0-9A-Fa-f]{2}(?::|-){1}[0-9A-Fa-f]{2}(?::|-){1}[0-9A-Fa-f]{2}(?::|-){1}[0-9A-Fa-f]{2}(?::|-){1}[0-9A-Fa-f]{2}\b")
HOUR_PATTERN = re.compile(r"\b(?:20\d{2}[-/]\d{1,2}[-/]\d{1,2}[ T])?([01]?\d|2[0-3]):[0-5]\d(?::[0-5]\d)?\b")
LOG_KEYWORDS = {
    "error": ("error", "err", "failed", "failure", "timeout", "unreachable", "down", "disconnect", "끊김", "실패", "오류"),
    "warning": ("warn", "warning", "주의", "경고", "retry", "retransmit"),
    "security": ("denied", "blocked", "attack", "malware", "virus", "ransom", "unauthorized", "login failed", "차단", "침입"),
    "network": ("link down", "link up", "flap", "storm", "broadcast", "arp", "mdns", "packet loss", "latency", "snmp", "trap"),
    "storage": ("backup", "snapshot", "hyper backup", "smb", "rename", "delete", "encrypt", "disk", "raid", "volume", "백업", "삭제", "이름 변경"),
}


def extract_large_log_context(attachments: Any) -> dict[str, Any]:
    if not attachments:
        raise ValueError("대용량 로그 분석에는 첨부파일이 필요합니다.")
    if not isinstance(attachments, list):
        raise ValueError("attachments must be a list")
    if len(attachments) > ATTACHMENT_MAX_COUNT:
        raise ValueError(f"too many attachments: max {ATTACHMENT_MAX_COUNT}")

    request_id = uuid.uuid4().hex
    request_dir = LARGE_LOG_DIR / datetime.now().strftime("%Y%m%d") / request_id
    total_bytes = 0
    combined_blocks: list[str] = []
    items: list[dict[str, Any]] = []
    errors: list[str] = []
    request_dir.mkdir(parents=True, exist_ok=True)
    for index, attachment in enumerate(attachments, start=1):
        if not isinstance(attachment, dict):
            errors.append(f"attachment #{index}: invalid payload")
            continue
        name = sanitize_attachment_name(attachment.get("name") or f"log-{index}.txt")
        encoded = str(attachment.get("content_base64") or "").strip()
        if not encoded:
            errors.append(f"{name}: empty content")
            continue
        try:
            raw = base64.b64decode(encoded, validate=True)
        except Exception:
            errors.append(f"{name}: invalid base64")
            continue
        size = len(raw)
        total_bytes += size
        if size > LARGE_LOG_MAX_FILE_BYTES:
            errors.append(f"{name}: file too large ({size} bytes)")
            continue
        if total_bytes > LARGE_LOG_MAX_TOTAL_BYTES:
            errors.append(f"{name}: total large-log attachment size exceeded")
            continue
        stored_path = request_dir / name
        stored_path.write_bytes(raw)
        try:
            text, method = extract_attachment_text(name, raw)
        except Exception as exc:
            errors.append(f"{name}: {exc}")
            continue
        text = text.strip()
        if not text:
            errors.append(f"{name}: extracted text is empty")
            continue
        combined_blocks.append(f"[파일 {index}] {name} / size={size} bytes / method={method}\n{text}")
        items.append(
            {
                "name": name,
                "size": size,
                "method": method,
                "stored_path": str(stored_path),
                "text_chars": len(text),
                "preview": clip_text(text, 360),
            }
        )
    if not items:
        raise ValueError("; ".join(errors) or "no readable large-log attachments")
    return {
        "request_id": request_id,
        "directory": str(request_dir),
        "items": items,
        "errors": errors,
        "total_bytes": total_bytes,
        "text": "\n\n".join(combined_blocks),
    }


def normalize_log_pattern(line: str) -> str:
    value = line.strip()
    value = IP_PATTERN.sub("<ip>", value)
    value = MAC_PATTERN.sub("<mac>", value)
    value = re.sub(r"\b\d{4}[-/]\d{1,2}[-/]\d{1,2}[ T]\d{1,2}:\d{2}(?::\d{2})?(?:[.,]\d+)?\b", "<ts>", value)
    value = re.sub(r"\b\d+\b", "<num>", value)
    value = re.sub(r"\s+", " ", value)
    return clip_text(value, 220)


def large_log_scan(text: str) -> dict[str, Any]:
    keyword_counts = {key: 0 for key in LOG_KEYWORDS}
    ips: Counter[str] = Counter()
    macs: Counter[str] = Counter()
    hours: Counter[str] = Counter()
    repeated: Counter[str] = Counter()
    examples: dict[str, list[str]] = {key: [] for key in LOG_KEYWORDS}
    line_count = 0
    non_empty_count = 0
    for raw_line in text.splitlines():
        line_count += 1
        line = raw_line.strip()
        if not line:
            continue
        non_empty_count += 1
        lower = line.lower()
        for key, keywords in LOG_KEYWORDS.items():
            if any(keyword in lower for keyword in keywords):
                keyword_counts[key] += 1
                if len(examples[key]) < 5:
                    examples[key].append(clip_text(line, 360))
        for ip in IP_PATTERN.findall(line):
            ips[ip] += 1
        for mac in MAC_PATTERN.findall(line):
            macs[mac.upper().replace("-", ":")] += 1
        hour_match = HOUR_PATTERN.search(line)
        if hour_match:
            hours[f"{int(hour_match.group(1)):02d}:00"] += 1
        pattern = normalize_log_pattern(line)
        if pattern:
            repeated[pattern] += 1
    return {
        "chars": len(text),
        "line_count": line_count,
        "non_empty_line_count": non_empty_count,
        "keyword_counts": keyword_counts,
        "top_ips": ips.most_common(20),
        "top_macs": macs.most_common(20),
        "top_hours": hours.most_common(24),
        "top_repeated_patterns": repeated.most_common(20),
        "examples": examples,
    }


def split_large_log_chunks(text: str) -> list[dict[str, Any]]:
    chunk_size = max(2000, LARGE_LOG_CHUNK_CHARS)
    chunks: list[dict[str, Any]] = []
    start = 0
    index = 1
    while start < len(text):
        end = min(len(text), start + chunk_size)
        if end < len(text):
            newline = text.rfind("\n", start + int(chunk_size * 0.55), end)
            if newline > start:
                end = newline + 1
        chunk_text = text[start:end]
        lower = chunk_text.lower()
        score = sum(
            lower.count(keyword)
            for keywords in LOG_KEYWORDS.values()
            for keyword in keywords
        )
        chunks.append({"index": index, "start": start, "end": end, "score": score, "text": chunk_text})
        index += 1
        start = end
    if len(chunks) <= LARGE_LOG_MAX_CHUNKS:
        return chunks
    selected: dict[int, dict[str, Any]] = {0: chunks[0], len(chunks) - 1: chunks[-1]}
    for chunk in sorted(chunks, key=lambda item: item["score"], reverse=True):
        if len(selected) >= LARGE_LOG_MAX_CHUNKS:
            break
        selected[chunk["index"] - 1] = chunk
    return [selected[key] for key in sorted(selected)]


def render_large_log_scan(scan: dict[str, Any]) -> str:
    repeated = "\n".join(
        f"- {count}회: {pattern}"
        for pattern, count in scan.get("top_repeated_patterns", [])[:10]
    ) or "- 없음"
    ips = ", ".join(f"{ip}({count})" for ip, count in scan.get("top_ips", [])[:10]) or "없음"
    macs = ", ".join(f"{mac}({count})" for mac, count in scan.get("top_macs", [])[:10]) or "없음"
    hours = ", ".join(f"{hour}({count})" for hour, count in scan.get("top_hours", [])[:12]) or "없음"
    return (
        "대용량 로그 1차 스캔\n"
        f"- 문자수: {scan.get('chars')} / 라인수: {scan.get('line_count')} / 유효라인: {scan.get('non_empty_line_count')}\n"
        f"- 키워드 카운트: {json.dumps(scan.get('keyword_counts') or {}, ensure_ascii=False)}\n"
        f"- 상위 IP: {ips}\n"
        f"- 상위 MAC: {macs}\n"
        f"- 상위 시간대: {hours}\n"
        f"- 반복 패턴:\n{repeated}"
    )


def run_large_log_chunk_analysis(
    *,
    text: str,
    scan: dict[str, Any],
    model: str,
    question: str,
    customer_name: str,
    site_code: str,
) -> dict[str, Any]:
    chunks = split_large_log_chunks(text)
    warnings: list[str] = []
    chunk_summaries: list[dict[str, Any]] = []
    scan_text = render_large_log_scan(scan)
    for chunk in chunks:
        messages = [
            {
                "role": "system",
                "content": (
                    "너는 네트워크/서버/NAS 운영 로그 분석가다. 로그에 없는 사실은 만들지 않는다. "
                    "확정 사실, 추정, 누락 데이터를 분리하고 한국어로 답한다."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"고객사={customer_name or '-'}, site={site_code or '-'}\n"
                    f"사용자 질문={question or '장애/보안/랜섬웨어/장비 이상 징후를 찾아라.'}\n\n"
                    f"{scan_text}\n\n"
                    f"아래는 전체 로그 중 선택된 조각 {chunk['index']}번이다. "
                    "시간, 장비, IP, MAC, 반복 오류, 보안 징후, 네트워크 장애 징후를 근거 중심으로 요약해라.\n\n"
                    f"{chunk['text']}"
                ),
            },
        ]
        chat = run_ollama_chat_messages(
            messages,
            model=model,
            timeout=LARGE_LOG_ANALYSIS_TIMEOUT,
            options={
                "temperature": 0.05,
                "top_p": 0.82,
                "repeat_penalty": NMS_ANALYSIS_REPEAT_PENALTY,
                "num_predict": 700,
            },
        )
        if not chat.get("ok"):
            warnings.append(f"조각 {chunk['index']}번 분석 실패: {result_error_text({'data': chat.get('raw')}) or 'unknown'}")
        chunk_summaries.append(
            {
                "index": chunk["index"],
                "start": chunk["start"],
                "end": chunk["end"],
                "score": chunk["score"],
                "ok": bool(chat.get("ok")),
                "elapsed_ms": chat.get("elapsed_ms"),
                "summary": clip_text(chat.get("response") or json.dumps(chat.get("raw") or {}, ensure_ascii=False), 3000),
            }
        )
    summary_text = "\n\n".join(
        f"[조각 {item['index']} score={item['score']} chars={item['end'] - item['start']}]\n{item['summary']}"
        for item in chunk_summaries
    )
    summary_text = clip_text(summary_text, LARGE_LOG_SUMMARY_CHAR_LIMIT)
    final_messages = [
        {
            "role": "system",
            "content": (
                "너는 운영 장애 분석 책임자다. 대용량 로그 전체 스캔 결과와 조각별 요약만 근거로 최종 판단한다. "
                "없는 데이터는 unknown/insufficient_data로 표시한다. 한국어로 답한다."
            ),
        },
        {
            "role": "user",
            "content": (
                f"고객사={customer_name or '-'}, site={site_code or '-'}\n"
                f"사용자 질문={question or '대용량 로그를 근거로 장애/보안/랜섬웨어/장비 이상을 판단해줘.'}\n\n"
                f"{scan_text}\n\n"
                "조각별 LLM 분석 요약:\n"
                f"{summary_text}\n\n"
                "최종 출력 형식:\n"
                "1. 종합 판정과 신뢰도\n"
                "2. 확정 사실\n"
                "3. 의심 원인 우선순위와 반박 근거\n"
                "4. 보안/랜섬웨어/파일작업 위험\n"
                "5. 네트워크 장애 징후\n"
                "6. 바로 확인할 원격 작업\n"
                "7. 현장 확인 작업\n"
                "8. 추가로 필요한 데이터"
            ),
        },
    ]
    final = run_ollama_chat_messages(
        final_messages,
        model=model,
        timeout=LARGE_LOG_ANALYSIS_TIMEOUT,
        options={
            "temperature": 0.06,
            "top_p": 0.82,
            "repeat_penalty": NMS_ANALYSIS_REPEAT_PENALTY,
            "num_predict": LARGE_LOG_NUM_PREDICT,
        },
    )
    return {
        "ok": bool(final.get("ok")),
        "model": final.get("model") or model,
        "warnings": [*warnings, *(final.get("warnings") or [])],
        "scan": scan,
        "chunk_count_total": max(1, (len(text) + max(1, LARGE_LOG_CHUNK_CHARS) - 1) // max(1, LARGE_LOG_CHUNK_CHARS)),
        "chunk_count_analyzed": len(chunks),
        "chunks": chunk_summaries,
        "response": final.get("response") or "",
        "raw": final.get("raw"),
        "elapsed_ms": sum(int(item.get("elapsed_ms") or 0) for item in chunk_summaries) + int(final.get("elapsed_ms") or 0),
    }


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
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS saved_analyses (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                customer_name TEXT NOT NULL DEFAULT '',
                site_code TEXT NOT NULL DEFAULT '',
                site_name TEXT NOT NULL DEFAULT '',
                analysis_kind TEXT NOT NULL DEFAULT 'freeform',
                title TEXT NOT NULL DEFAULT '',
                question TEXT NOT NULL DEFAULT '',
                content TEXT NOT NULL,
                source_excerpt TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                meta_json TEXT NOT NULL DEFAULT '{}'
            )
            """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_saved_analyses_scope_updated
            ON saved_analyses(customer_name, site_code, updated_at DESC)
            """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_saved_analyses_kind_updated
            ON saved_analyses(analysis_kind, updated_at DESC)
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS nms_evidence_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                customer_name TEXT NOT NULL DEFAULT '',
                site_codes TEXT NOT NULL DEFAULT '',
                site_names TEXT NOT NULL DEFAULT '',
                question TEXT NOT NULL DEFAULT '',
                depth TEXT NOT NULL DEFAULT '',
                hours INTEGER NOT NULL DEFAULT 0,
                source_type TEXT NOT NULL DEFAULT 'network-evidence-pack',
                evidence_version TEXT NOT NULL DEFAULT '',
                digest TEXT NOT NULL DEFAULT '',
                deterministic_brief TEXT NOT NULL DEFAULT '',
                compact_context TEXT NOT NULL DEFAULT '',
                full_context TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL,
                meta_json TEXT NOT NULL DEFAULT '{}'
            )
            """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_nms_evidence_scope_created
            ON nms_evidence_snapshots(customer_name, site_codes, created_at DESC)
            """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_nms_evidence_created
            ON nms_evidence_snapshots(created_at DESC)
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


def list_autopilot_history(limit: int = 20) -> list[dict[str, Any]]:
    init_conversation_store()
    safe_limit = max(1, min(int(limit or 20), 100))
    with CONVERSATION_LOCK, sqlite3.connect(CONVERSATION_DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """
            SELECT c.*,
                   (SELECT COUNT(*) FROM conversation_messages m WHERE m.conversation_id = c.id) AS message_count,
                   (SELECT content FROM conversation_messages m
                    WHERE m.conversation_id = c.id AND m.role = 'assistant'
                    ORDER BY m.id DESC LIMIT 1) AS latest_assistant,
                   (SELECT created_at FROM conversation_messages m
                    WHERE m.conversation_id = c.id AND m.role = 'assistant'
                    ORDER BY m.id DESC LIMIT 1) AS latest_assistant_at,
                   (SELECT content FROM conversation_messages m
                    WHERE m.conversation_id = c.id AND m.role = 'user'
                    ORDER BY m.id DESC LIMIT 1) AS latest_user
            FROM conversations c
            WHERE c.id LIKE ?
               OR c.meta_json LIKE ?
               OR c.meta_json LIKE ?
            ORDER BY c.updated_at DESC
            LIMIT ?
            """,
            (
                f"{NMS_AUTOPILOT_CONVERSATION_PREFIX}-%",
                '%"source": "nms-autopilot"%',
                '%"source":"nms-autopilot"%',
                safe_limit,
            ),
        ).fetchall()
        items = []
        for row in rows:
            item = serialize_conversation_row(row)
            meta = item.get("meta") if isinstance(item.get("meta"), dict) else {}
            target = meta.get("target") if isinstance(meta.get("target"), dict) else {}
            item["target"] = target
            item["message_count"] = int(row["message_count"] or 0)
            item["latest_user"] = clip_text(row["latest_user"] or "", 500)
            item["latest_assistant_preview"] = clip_text(row["latest_assistant"] or "", 1200)
            item["latest_assistant_at"] = row["latest_assistant_at"] or ""
            items.append(item)
        return items


def conversation_store_counts() -> dict[str, int]:
    init_conversation_store()
    with CONVERSATION_LOCK, sqlite3.connect(CONVERSATION_DB_PATH) as conn:
        conversation_count = conn.execute("SELECT COUNT(*) FROM conversations").fetchone()[0]
        message_count = conn.execute("SELECT COUNT(*) FROM conversation_messages").fetchone()[0]
        saved_count = conn.execute("SELECT COUNT(*) FROM saved_analyses").fetchone()[0]
    return {
        "conversations": int(conversation_count or 0),
        "messages": int(message_count or 0),
        "saved_analyses": int(saved_count or 0),
    }


def operations_dashboard_payload() -> dict[str, Any]:
    return {
        "success": True,
        "generated_at": utc_now(),
        "conversation_store": conversation_store_counts(),
        "nms_context_cache": nms_context_cache_snapshot(),
        "monitor": NMS_MONITOR.snapshot(),
        "autopilot": NMS_AUTOPILOT.snapshot(),
        "autopilot_history": list_autopilot_history(limit=20),
        "saved_analyses": list_saved_analyses(limit=20, include_content=False),
    }


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
        if not isinstance(message, dict):
            continue
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
            if not isinstance(message, dict):
                continue
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


def normalize_site_codes(value: Any) -> list[str]:
    if isinstance(value, list):
        parts = value
    else:
        parts = str(value or "").split(",")
    seen: set[str] = set()
    codes: list[str] = []
    for part in parts:
        code = str(part or "").strip()
        if not code or code in seen:
            continue
        seen.add(code)
        codes.append(code[:80])
    return codes


def normalize_saved_analysis_id(value: Any) -> int | None:
    try:
        parsed = int(str(value or "").strip())
        return parsed if parsed > 0 else None
    except Exception:
        return None


def serialize_saved_analysis_row(row: sqlite3.Row, include_content: bool = False) -> dict[str, Any]:
    try:
        meta = json.loads(row["meta_json"] or "{}")
    except Exception:
        meta = {}
    content = row["content"] or ""
    item = {
        "id": int(row["id"]),
        "customer_name": row["customer_name"] or "",
        "site_code": row["site_code"] or "",
        "site_name": row["site_name"] or "",
        "analysis_kind": row["analysis_kind"] or "freeform",
        "title": row["title"] or "",
        "question": row["question"] or "",
        "source_excerpt": row["source_excerpt"] or "",
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
        "meta": meta,
        "content_preview": clip_text(content, SAVED_ANALYSIS_PREVIEW_CHAR_LIMIT),
    }
    if include_content:
        item["content"] = content
    return item


def list_saved_analyses(
    customer_name: str = "",
    site_code: Any = "",
    limit: int = 50,
    include_content: bool = False,
) -> list[dict[str, Any]]:
    init_conversation_store()
    safe_limit = max(1, min(int(limit or 50), 200))
    normalized_customer = clip_text(str(customer_name or "").strip(), 200)
    site_codes = normalize_site_codes(site_code)
    conditions: list[str] = []
    params: list[Any] = []
    placeholders = ""
    if normalized_customer:
        conditions.append("customer_name = ?")
        params.append(normalized_customer)
    if site_codes:
        placeholders = ",".join("?" for _ in site_codes)
        conditions.append(f"(site_code = '' OR site_code IN ({placeholders}))")
        params.extend(site_codes)
    query = "SELECT * FROM saved_analyses"
    if conditions:
        query += " WHERE " + " AND ".join(conditions)
    if site_codes:
        query += f" ORDER BY CASE WHEN site_code IN ({placeholders}) THEN 0 WHEN site_code = '' THEN 1 ELSE 2 END, updated_at DESC"
        params.extend(site_codes)
    else:
        query += " ORDER BY updated_at DESC"
    query += " LIMIT ?"
    params.append(safe_limit)
    with CONVERSATION_LOCK, sqlite3.connect(CONVERSATION_DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(query, tuple(params)).fetchall()
        return [serialize_saved_analysis_row(row, include_content=include_content) for row in rows]


def load_saved_analysis(analysis_id: Any) -> dict[str, Any] | None:
    init_conversation_store()
    normalized_id = normalize_saved_analysis_id(analysis_id)
    if normalized_id is None:
        return None
    with CONVERSATION_LOCK, sqlite3.connect(CONVERSATION_DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT * FROM saved_analyses WHERE id = ?", (normalized_id,)).fetchone()
        if not row:
            return None
        return serialize_saved_analysis_row(row, include_content=True)


def create_saved_analysis(body: dict[str, Any]) -> dict[str, Any]:
    init_conversation_store()
    customer_name = clip_text(str(body.get("customer_name") or body.get("customer") or "").strip(), 200)
    if not customer_name:
        raise ValueError("customer_name is required")
    site_codes = normalize_site_codes(body.get("site_code") or body.get("site_codes") or "")
    site_code = site_codes[0] if site_codes else ""
    site_name = clip_text(str(body.get("site_name") or "").strip(), 200)
    analysis_kind = clip_text(str(body.get("analysis_kind") or body.get("template") or "freeform").strip(), 60) or "freeform"
    title = clip_text(
        str(body.get("title") or body.get("question") or f"{customer_name or '미지정 고객'} 분석").strip(),
        CONVERSATION_TITLE_CHAR_LIMIT,
    ) or "저장 분석"
    question = clip_text(str(body.get("question") or "").strip(), 600)
    content = clip_text(body.get("content") or body.get("response") or "", SAVED_ANALYSIS_CONTENT_CHAR_LIMIT).strip()
    source_excerpt = clip_text(body.get("source_excerpt") or body.get("data") or body.get("prompt") or "", SAVED_ANALYSIS_SOURCE_CHAR_LIMIT).strip()
    if not content:
        raise ValueError("content is required")
    meta = body.get("meta") if isinstance(body.get("meta"), dict) else {}
    now = utc_now()
    with CONVERSATION_LOCK, sqlite3.connect(CONVERSATION_DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.execute(
            """
            INSERT INTO saved_analyses (
                customer_name, site_code, site_name, analysis_kind, title, question,
                content, source_excerpt, created_at, updated_at, meta_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                customer_name,
                site_code,
                site_name,
                analysis_kind,
                title,
                question,
                content,
                source_excerpt,
                now,
                now,
                json.dumps(meta, ensure_ascii=False),
            ),
        )
        conn.commit()
        row = conn.execute("SELECT * FROM saved_analyses WHERE id = ?", (cursor.lastrowid,)).fetchone()
        return serialize_saved_analysis_row(row, include_content=True)


def delete_saved_analysis(analysis_id: Any) -> bool:
    init_conversation_store()
    normalized_id = normalize_saved_analysis_id(analysis_id)
    if normalized_id is None:
        return False
    with CONVERSATION_LOCK, sqlite3.connect(CONVERSATION_DB_PATH) as conn:
        result = conn.execute("DELETE FROM saved_analyses WHERE id = ?", (normalized_id,))
        conn.commit()
        return result.rowcount > 0


def build_saved_analysis_context(customer_name: str, site_code: Any = "", limit: int = SAVED_ANALYSIS_CONTEXT_LIMIT) -> str:
    normalized_customer = clip_text(str(customer_name or "").strip(), 200)
    if not normalized_customer:
        return ""
    items = list_saved_analyses(
        customer_name=normalized_customer,
        site_code=site_code,
        limit=max(1, min(int(limit or SAVED_ANALYSIS_CONTEXT_LIMIT), 8)),
        include_content=True,
    )
    if not items:
        return ""
    lines = ["이전 저장 분석 참고자료"]
    for index, item in enumerate(items, start=1):
        scope = item.get("site_name") or item.get("site_code") or "전체 현장"
        lines.append(
            f"[{index}] {item.get('title') or '저장 분석'} / scope={scope} / kind={item.get('analysis_kind')} / updated={item.get('updated_at')}"
        )
        if item.get("question"):
            lines.append(f"당시 요청: {item['question']}")
        if item.get("source_excerpt"):
            lines.append(f"당시 입력 요약:\n{clip_text(item['source_excerpt'], 700)}")
        lines.append(f"당시 분석 결론:\n{clip_text(item.get('content') or '', 1400)}")
    lines.append("위 자료는 참고 이력이다. 현재 입력 데이터와 충돌하면 현재 데이터와 최신 로그를 우선한다.")
    return "\n\n".join(lines)


def json_dumps_compact(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=str)


def nms_evidence_scope_values(
    context: dict[str, Any],
    customer_name: str = "",
    site_codes: Any = "",
) -> tuple[str, str, str]:
    matched = safe_dict(context.get("matched"))
    requested = safe_dict(context.get("requested"))
    customer = str(customer_name or requested.get("customer_name") or "").strip()
    if not customer:
        customer = ", ".join(str(name) for name in (matched.get("customer_names") or []) if name)
    codes = normalize_site_codes(site_codes or requested.get("site_codes") or matched.get("site_codes") or [])
    site_names = ", ".join(str(name) for name in (matched.get("site_names") or []) if name)
    return clip_text(customer, 200), ",".join(codes), clip_text(site_names, 400)


def serialize_nms_evidence_snapshot_row(row: sqlite3.Row, include_full: bool = False) -> dict[str, Any]:
    try:
        meta = json.loads(row["meta_json"] or "{}")
    except Exception:
        meta = {}
    item = {
        "id": int(row["id"]),
        "customer_name": row["customer_name"] or "",
        "site_codes": row["site_codes"] or "",
        "site_names": row["site_names"] or "",
        "question": row["question"] or "",
        "depth": row["depth"] or "",
        "hours": int(row["hours"] or 0),
        "source_type": row["source_type"] or "",
        "evidence_version": row["evidence_version"] or "",
        "digest": row["digest"] or "",
        "deterministic_brief": row["deterministic_brief"] or "",
        "compact_context_preview": clip_text(row["compact_context"] or "", NMS_EVIDENCE_STORE_CONTEXT_LIMIT),
        "full_context_chars": len(row["full_context"] or ""),
        "compact_context_chars": len(row["compact_context"] or ""),
        "created_at": row["created_at"],
        "meta": meta,
    }
    if include_full:
        item["compact_context"] = row["compact_context"] or ""
        item["full_context"] = row["full_context"] or ""
    return item


def list_nms_evidence_snapshots(
    customer_name: str = "",
    site_codes: Any = "",
    limit: int = NMS_EVIDENCE_STORE_RECENT_LIMIT,
    include_full: bool = False,
) -> list[dict[str, Any]]:
    init_conversation_store()
    safe_limit = max(1, min(int(limit or NMS_EVIDENCE_STORE_RECENT_LIMIT), 100))
    normalized_customer = clip_text(str(customer_name or "").strip(), 200)
    normalized_site_codes = ",".join(normalize_site_codes(site_codes))
    conditions: list[str] = []
    params: list[Any] = []
    if normalized_customer:
        conditions.append("customer_name = ?")
        params.append(normalized_customer)
    if normalized_site_codes:
        conditions.append("(site_codes = '' OR site_codes = ?)")
        params.append(normalized_site_codes)
    query = "SELECT * FROM nms_evidence_snapshots"
    if conditions:
        query += " WHERE " + " AND ".join(conditions)
    query += " ORDER BY created_at DESC, id DESC LIMIT ?"
    params.append(safe_limit)
    with CONVERSATION_LOCK, sqlite3.connect(CONVERSATION_DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(query, params).fetchall()
    return [serialize_nms_evidence_snapshot_row(row, include_full=include_full) for row in rows]


def load_nms_evidence_snapshot(snapshot_id: Any, include_full: bool = True) -> dict[str, Any] | None:
    try:
        normalized_id = int(str(snapshot_id or "").strip())
    except Exception:
        return None
    if normalized_id <= 0:
        return None
    init_conversation_store()
    with CONVERSATION_LOCK, sqlite3.connect(CONVERSATION_DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT * FROM nms_evidence_snapshots WHERE id = ?", (normalized_id,)).fetchone()
    return serialize_nms_evidence_snapshot_row(row, include_full=include_full) if row else None


def create_nms_evidence_snapshot(
    context: dict[str, Any],
    customer_name: str = "",
    site_codes: Any = "",
    question: str = "",
    depth: str = "deep",
    hours: Any = 0,
) -> dict[str, Any] | None:
    if not NMS_EVIDENCE_STORE_ENABLED or not context:
        return None
    init_conversation_store()
    scoped_customer, scoped_site_codes, scoped_site_names = nms_evidence_scope_values(context, customer_name, site_codes)
    compact_context = compact_nms_context_for_llm(context)
    compact_text = clip_middle_text(json_dumps_compact(compact_context), NMS_ANALYSIS_CONTEXT_CHAR_LIMIT)
    full_text = clip_middle_text(json_dumps_compact(context), NMS_EVIDENCE_STORE_MAX_FULL_CHARS)
    digest = build_nms_evidence_digest(context)
    deterministic_brief = build_nms_deterministic_brief(context)
    if is_field_analysis_evidence_pack(context):
        source_type = "field-analysis-evidence"
    elif is_network_evidence_pack(context):
        source_type = "network-evidence-pack"
    else:
        source_type = "nms-context"
    evidence_version = str(context.get("evidence_pack_version") or context.get("version") or "")
    context_hash = hashlib.sha256(full_text.encode("utf-8", errors="ignore")).hexdigest()[:16]
    now = utc_now()
    meta = {
        "context_hash": context_hash,
        "requested": safe_dict(context.get("requested")),
        "matched": safe_dict(context.get("matched")),
        "compact_chars": len(compact_text),
        "full_chars": len(full_text),
    }
    with CONVERSATION_LOCK, sqlite3.connect(CONVERSATION_DB_PATH) as conn:
        cursor = conn.execute(
            """
            INSERT INTO nms_evidence_snapshots (
                customer_name, site_codes, site_names, question, depth, hours, source_type,
                evidence_version, digest, deterministic_brief, compact_context, full_context,
                created_at, meta_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                scoped_customer,
                scoped_site_codes,
                scoped_site_names,
                clip_text(question, 500),
                clip_text(depth, 40),
                safe_int(hours, 0),
                source_type,
                clip_text(evidence_version, 80),
                clip_text(digest, 12000),
                clip_text(deterministic_brief, 18000),
                compact_text,
                full_text,
                now,
                json.dumps(meta, ensure_ascii=False),
            ),
        )
        snapshot_id = cursor.lastrowid
        conn.execute(
            """
            DELETE FROM nms_evidence_snapshots
            WHERE customer_name = ?
              AND site_codes = ?
              AND id NOT IN (
                  SELECT id
                  FROM nms_evidence_snapshots
                  WHERE customer_name = ? AND site_codes = ?
                  ORDER BY created_at DESC, id DESC
                  LIMIT ?
              )
            """,
            (
                scoped_customer,
                scoped_site_codes,
                scoped_customer,
                scoped_site_codes,
                max(1, NMS_EVIDENCE_STORE_MAX_ITEMS_PER_SCOPE),
            ),
        )
        conn.commit()
    loaded = load_nms_evidence_snapshot(snapshot_id, include_full=False)
    return loaded or {
        "id": snapshot_id,
        "customer_name": scoped_customer,
        "site_codes": scoped_site_codes,
        "created_at": now,
        "compact_context_chars": len(compact_text),
        "full_context_chars": len(full_text),
        "meta": meta,
    }


def build_nms_evidence_store_context(
    customer_name: str = "",
    site_codes: Any = "",
    limit: int = NMS_EVIDENCE_STORE_RECENT_LIMIT,
) -> str:
    items = list_nms_evidence_snapshots(customer_name=customer_name, site_codes=site_codes, limit=limit)
    if not items:
        return ""
    lines = [
        "118 Evidence Store 참고",
        "원본 evidence는 118 서버 SQLite에 저장되어 있으며, LLM 프롬프트에는 아래 요약/축약본만 포함한다.",
    ]
    for item in items:
        scope = item.get("site_names") or item.get("site_codes") or "전체 현장"
        lines.append(
            f"[snapshot_id={item.get('id')}] created={item.get('created_at')} "
            f"customer={item.get('customer_name') or '-'} scope={scope} "
            f"source={item.get('source_type')} full_chars={item.get('full_context_chars')}"
        )
        if item.get("question"):
            lines.append(f"요청: {clip_text(item.get('question'), 300)}")
        lines.append(f"근거 요약:\n{clip_text(item.get('digest') or '', 1400)}")
        if item.get("deterministic_brief"):
            lines.append(f"규칙 기반 1차 브리핑:\n{clip_text(item.get('deterministic_brief') or '', 1600)}")
    lines.append("저장 evidence와 현재 축약 JSON이 충돌하면 현재 요청의 timestamp와 최신 snapshot_id를 우선한다.")
    return "\n\n".join(lines)


def is_network_evidence_pack(context: dict[str, Any]) -> bool:
    return str((context or {}).get("evidence_pack_version") or "").startswith("network-evidence-pack")


def is_field_analysis_evidence_pack(context: dict[str, Any]) -> bool:
    return str((context or {}).get("evidence_pack_version") or "").startswith("field-analysis-evidence")


def limited_list(value: Any, limit: int) -> list[Any]:
    if not isinstance(value, list):
        return []
    return value[: max(0, limit)]


def compact_evidence_views(value: Any, row_limit: int = 20) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    compact: dict[str, Any] = {}
    for name, payload in value.items():
        if isinstance(payload, dict):
            next_payload = dict(payload)
            if isinstance(next_payload.get("rows"), list):
                next_payload["rows"] = limited_list(next_payload.get("rows"), row_limit)
            compact[str(name)] = next_payload
        elif isinstance(payload, list):
            compact[str(name)] = limited_list(payload, row_limit)
        else:
            compact[str(name)] = payload
    return compact


def compact_compressed_evidence(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    compact = dict(value)
    aggregates = value.get("aggregates") if isinstance(value.get("aggregates"), dict) else {}
    samples = value.get("representative_samples") if isinstance(value.get("representative_samples"), dict) else {}
    compact["aggregates"] = {
        "syslog_by_hour": limited_list(aggregates.get("syslog_by_hour"), 48),
        "syslog_top_devices": limited_list(aggregates.get("syslog_top_devices"), 20),
        "syslog_top_apps": limited_list(aggregates.get("syslog_top_apps"), 20),
        "trap_top_oids": limited_list(aggregates.get("trap_top_oids"), 20),
        "trap_top_sources": limited_list(aggregates.get("trap_top_sources"), 20),
        "polling_top_metrics": limited_list(aggregates.get("polling_top_metrics"), 25),
        "interface_latest_by_device": limited_list(aggregates.get("interface_latest_by_device"), 25),
        "nas_docker_latest": limited_list(aggregates.get("nas_docker_latest"), 40),
        "gateway_probe_latest": limited_list(aggregates.get("gateway_probe_latest"), 30),
        "dns_probe_latest": limited_list(aggregates.get("dns_probe_latest"), 30),
        "arp_latest": limited_list(aggregates.get("arp_latest"), 40),
        "mac_latest": limited_list(aggregates.get("mac_latest"), 40),
        "mdns_latest": limited_list(aggregates.get("mdns_latest"), 40),
        "arp_top_talkers": limited_list(aggregates.get("arp_top_talkers"), 30),
        "mdns_top_names": limited_list(aggregates.get("mdns_top_names"), 30),
        "snmp_disk_latest": limited_list(aggregates.get("snmp_disk_latest"), 30),
        "snmp_interface_errors": limited_list(aggregates.get("snmp_interface_errors"), 30),
        "faulty_devices": limited_list(aggregates.get("faulty_devices"), 30),
        "abnormal_traffic": limited_list(aggregates.get("abnormal_traffic"), 30),
        "flood_summary": limited_list(aggregates.get("flood_summary"), 30),
        "ingress_summary": limited_list(aggregates.get("ingress_summary"), 30),
    }
    compact["representative_samples"] = {
        "syslog_patterns": limited_list(samples.get("syslog_patterns"), 25),
        "syslog_dhcp_patterns": limited_list(samples.get("syslog_dhcp_patterns"), 25),
        "syslog_stp_patterns": limited_list(samples.get("syslog_stp_patterns"), 25),
        "polling_latest_by_device": limited_list(samples.get("polling_latest_by_device"), 25),
    }
    return compact


def compact_event_classification(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    compact = {
        "version": value.get("version"),
        "summary": value.get("summary") or {},
        "data_quality": value.get("data_quality") or {},
        "active_categories": limited_list(value.get("active_categories"), 12),
        "top_events": limited_list(value.get("top_events"), 25),
        "llm_focus": limited_list(value.get("llm_focus"), 12),
    }
    event_types = value.get("event_types") if isinstance(value.get("event_types"), dict) else {}
    compact["event_types"] = {
        key: {
            "label": item.get("label") if isinstance(item, dict) else "",
            "category": item.get("category") if isinstance(item, dict) else "",
            "count": item.get("count") if isinstance(item, dict) else 0,
            "max_severity_rank": item.get("max_severity_rank") if isinstance(item, dict) else 0,
            "representative_events": limited_list(item.get("representative_events"), 2) if isinstance(item, dict) else [],
        }
        for key, item in list(event_types.items())[:20]
    }
    return compact


def compact_network_evidence_for_llm(context: dict[str, Any]) -> dict[str, Any]:
    analysis = context.get("deterministic_analysis") or {}
    event_classification = compact_event_classification(
        context.get("event_classification") or analysis.get("event_classification") or {}
    )
    raw = context.get("raw_evidence") or {}
    logs = raw.get("logs") or {}
    traffic = raw.get("traffic") or {}
    temporal = raw.get("temporal") or {}
    compressed = compact_compressed_evidence(context.get("compressed_evidence") or {})
    compact_analysis = {
        "data_policy": analysis.get("data_policy") or context.get("source_policy") or {},
        "coverage": analysis.get("coverage") or context.get("data_coverage") or {},
        "top_signals": limited_list(analysis.get("top_signals"), 40),
        "event_classification": event_classification,
        "protocol_analysis": analysis.get("protocol_analysis") or {},
        "timeline": limited_list(analysis.get("timeline"), 80),
        "gaps": limited_list(analysis.get("gaps"), 30),
        "root_cause_hypotheses": limited_list(analysis.get("root_cause_hypotheses"), 12),
        "required_next_checks": limited_list(analysis.get("required_next_checks"), 30),
    }
    raw_samples = {
        "sites": limited_list(raw.get("sites"), 20),
        "devices": limited_list(raw.get("devices"), 80),
        "traffic": {
            "interface_count": traffic.get("interface_count", 0),
            "interfaces": limited_list(traffic.get("interfaces"), 30),
        },
        "logs": {
            "syslog_count": logs.get("syslog_count", 0),
            "trap_count": logs.get("trap_count", 0),
            "syslog": limited_list(logs.get("syslog"), 30),
            "traps": limited_list(logs.get("traps"), 20),
        },
        "temporal": temporal,
        "views": compact_evidence_views(raw.get("views") or {}, row_limit=15),
    }
    return {
        "generated_at": context.get("generated_at"),
        "evidence_pack_version": context.get("evidence_pack_version"),
        "requested": context.get("requested") or {},
        "matched": context.get("matched") or {},
        "source_policy": context.get("source_policy") or {},
        "data_coverage": context.get("data_coverage") or {},
        "event_classification": event_classification,
        "compressed_evidence": compressed,
        "deterministic_analysis": compact_analysis,
        "raw_evidence_policy": context.get("raw_evidence_policy") or {
            "role": "sample_only",
            "note": "raw_evidence는 제한 샘플이다. 기간 전체 판단은 compressed_evidence와 deterministic_analysis를 우선한다.",
        },
        "raw_evidence_samples": raw_samples,
        "codex_command_set": context.get("codex_command_set") or {},
        "llm_payload_policy": {
            "rule": "33번 NMS가 기간 전체 원천 데이터를 SQL로 먼저 집계/압축했고, 118번 LLM은 이 압축 evidence와 대표 샘플을 분석한다.",
            "do_not_do": [
                "원문 전체가 없다는 이유로 없는 값을 상상하지 않는다.",
                "raw_evidence_samples의 샘플 건수를 기간 전체 건수로 오해하지 않는다.",
                "compressed_evidence.source_totals가 0이면 해당 소스는 관측되지 않음으로 표시한다.",
            ],
        },
    }


def compact_field_analysis_evidence_for_llm(context: dict[str, Any]) -> dict[str, Any]:
    compressed = context.get("compressed_evidence") if isinstance(context.get("compressed_evidence"), dict) else {}
    aggregates = compressed.get("aggregates") if isinstance(compressed.get("aggregates"), dict) else {}
    samples = compressed.get("representative_samples") if isinstance(compressed.get("representative_samples"), dict) else {}
    analysis = context.get("deterministic_analysis") if isinstance(context.get("deterministic_analysis"), dict) else {}
    return {
        "generated_at": context.get("generated_at"),
        "evidence_pack_version": context.get("evidence_pack_version"),
        "requested": context.get("requested") or {},
        "matched": context.get("matched") or {},
        "target": context.get("target") or {},
        "source_policy": context.get("source_policy") or {},
        "data_coverage": context.get("data_coverage") or {},
        "deterministic_analysis": {
            "data_policy": analysis.get("data_policy") or {},
            "top_signals": limited_list(analysis.get("top_signals"), 30),
            "gaps": limited_list(analysis.get("gaps"), 30),
            "root_cause_hypotheses": limited_list(analysis.get("root_cause_hypotheses"), 10),
            "required_next_checks": limited_list(analysis.get("required_next_checks"), 30),
        },
        "compressed_evidence": {
            "source_totals": compressed.get("source_totals") or {},
            "aggregates": {
                "pulse_status_counts": limited_list(aggregates.get("pulse_status_counts"), 20),
                "pulse_endpoint_counts": limited_list(aggregates.get("pulse_endpoint_counts"), 20),
                "pulse_metric_summary": limited_list(aggregates.get("pulse_metric_summary"), 40),
                "site_collectors": limited_list(aggregates.get("site_collectors"), 20),
                "diagnostic_commands": limited_list(aggregates.get("diagnostic_commands"), 30),
                "polling_latest": limited_list(aggregates.get("polling_latest"), 40),
            },
            "representative_samples": {
                "pulse_observations": limited_list(samples.get("pulse_observations"), 35),
                "test_sessions": limited_list(samples.get("test_sessions"), 20),
                "syslog_events": limited_list(samples.get("syslog_events"), 30),
                "snmp_traps": limited_list(samples.get("snmp_traps"), 20),
            },
        },
        "raw_evidence_policy": context.get("raw_evidence_policy") or {
            "role": "bounded_samples",
            "note": "선택 현장 테스트 대상의 제한 샘플이다. 전체 판단은 source_totals와 deterministic_analysis를 우선한다.",
        },
        "codex_command_set": context.get("codex_command_set") or {},
        "llm_payload_policy": {
            "rule": "33번 NMS가 선택 대상의 원천 데이터를 압축했고, 118번 LLM은 확정 사실과 누락 데이터를 분리해 보고서로 정리한다.",
            "do_not_do": [
                "public_ip와 private_ip를 같은 의미로 섞지 않는다.",
                "없는 PoE/포트/VLAN/LLDP/ARP 값을 상상하지 않는다.",
                "수집되지 않은 diagnostic 결과를 정상 결과로 해석하지 않는다.",
            ],
        },
    }


def compact_nms_context_for_llm(context: dict[str, Any]) -> dict[str, Any]:
    if is_field_analysis_evidence_pack(context):
        return compact_field_analysis_evidence_for_llm(context)
    if is_network_evidence_pack(context):
        return compact_network_evidence_for_llm(context)
    raw_logs = context.get("logs") or {}
    raw_traffic = context.get("traffic") or {}
    return {
        "generated_at": context.get("generated_at"),
        "requested": context.get("requested") or {},
        "matched": context.get("matched") or {},
        "sites": limited_list(context.get("sites"), 20),
        "devices": limited_list(context.get("devices"), 80),
        "traffic": {
            "interface_count": raw_traffic.get("interface_count", 0),
            "interfaces": limited_list(raw_traffic.get("interfaces"), 30),
        },
        "logs": {
            "syslog_count": raw_logs.get("syslog_count", 0),
            "trap_count": raw_logs.get("trap_count", 0),
            "syslog": limited_list(raw_logs.get("syslog"), 30),
            "traps": limited_list(raw_logs.get("traps"), 20),
        },
        "temporal": context.get("temporal") or {},
        "llm_payload_policy": {
            "rule": "NMS 원천 데이터 중 LLM에 필요한 샘플만 포함한다. 전체 건수는 count 필드를 우선한다.",
        },
    }


def compact_messages_for_prompt_budget(messages: list[dict[str, str]], warnings: list[str]) -> list[dict[str, str]]:
    total_chars = sum(len(str(message.get("content") or "")) for message in messages)
    if total_chars <= LLM_OPS_MAX_PROMPT_CHARS:
        return messages

    compacted: list[dict[str, str]] = []
    remaining = max(4000, LLM_OPS_MAX_PROMPT_CHARS)
    for index, message in enumerate(messages):
        role = str(message.get("role") or "user")
        content = str(message.get("content") or "")
        if not content:
            continue
        if role == "system":
            limit = min(len(content), 12000, remaining)
        elif index == len(messages) - 1:
            limit = min(max(8000, int(remaining * 0.7)), LLM_OPS_MAX_SINGLE_MESSAGE_CHARS)
        else:
            limit = min(max(2500, int(remaining * 0.25)), LLM_OPS_MAX_SINGLE_MESSAGE_CHARS)
        clipped = clip_middle_text(content, max(800, limit))
        compacted.append({"role": role, "content": clipped})
        remaining -= len(clipped)
        if remaining <= 2000:
            remaining = 2000

    new_total = sum(len(message["content"]) for message in compacted)
    if new_total < total_chars:
        warnings.append(
            f"프롬프트가 {total_chars:,}자로 너무 커서 {new_total:,}자로 압축했습니다. "
            "NMS 분석은 /api/nms/analyze의 compressed evidence 경로 사용을 권장합니다."
        )
    return compacted


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


def ensure_default_chat_system_message(messages: list[dict[str, Any]]) -> list[dict[str, str]]:
    normalized = []
    has_system = False
    for message in messages:
        role = str(message.get("role") or "").strip()
        content = str(message.get("content") or "")
        if not content:
            continue
        normalized_message = {"role": role, "content": content}
        normalized.append(normalized_message)
        if role == "system":
            has_system = True
    guard_messages = []
    if KOREAN_RESPONSE_GUARD.strip():
        guard_messages.append({"role": "system", "content": KOREAN_RESPONSE_GUARD})
    if has_system or not DEFAULT_CHAT_SYSTEM_PROMPT.strip():
        return [*guard_messages, *normalized]
    return [*guard_messages, {"role": "system", "content": DEFAULT_CHAT_SYSTEM_PROMPT}, *normalized]


def contains_hangul(text: Any) -> bool:
    return any("가" <= ch <= "힣" for ch in str(text or ""))


def starts_with_non_korean_response(text: Any) -> bool:
    stripped = str(text or "").lstrip()
    if not stripped:
        return False
    first = stripped[0]
    chinese_openers = ("明白", "好的", "当然", "理解", "是的", "可以")
    english_openers = ("sure", "okay", "ok,", "yes", "understood", "certainly")
    if any(stripped.startswith(prefix) for prefix in chinese_openers):
        return True
    if first.isascii() and stripped.lower().startswith(english_openers):
        return True
    return not contains_hangul(stripped[:120])


def should_retry_korean_response(messages: list[dict[str, str]], response_text: Any) -> bool:
    if not KOREAN_RETRY_ENABLED:
        return False
    if not starts_with_non_korean_response(response_text):
        return False
    user_text = "\n".join(
        str(message.get("content") or "")
        for message in messages
        if str(message.get("role") or "") == "user"
    )
    if not contains_hangul(user_text):
        return False
    lowered = user_text.lower()
    explicit_foreign = ("영어로", "중국어로", "일본어로", "in english", "in chinese", "in japanese")
    return not any(token in lowered for token in explicit_foreign)


def sanitize_korean_response(text: Any) -> str:
    value = str(text or "").strip()
    has_chinese_fragment = bool(re.search(r"[\u4e00-\u9fff]", value))
    if not value or not (starts_with_non_korean_response(value) or has_chinese_fragment) or not contains_hangul(value):
        return value
    korean_lines = [line.strip() for line in value.splitlines() if contains_hangul(line)]
    if not korean_lines:
        return value
    cleaned = "\n".join(korean_lines).strip()
    cleaned = re.sub(r"^한국어로\s*(다시\s*)?(작성하면|말하면|답하면)[,:，]?\s*", "", cleaned)
    cleaned = re.sub(r"^요약하면[,:，]?\s*", "", cleaned)
    if re.search(r"[\u4e00-\u9fff]", cleaned):
        candidates = [
            candidate.strip().strip("\"'“”.,，。:：")
            for candidate in re.findall(r"[가-힣A-Za-z0-9\s.,!?·:;()\-\"'“”]+", cleaned)
            if contains_hangul(candidate)
        ]
        if candidates:
            cleaned = max(candidates, key=len)
    return cleaned.strip().strip("\"'“”")


def sanitize_uncomputed_cadence_claims(text: Any) -> str:
    value = str(text or "")
    cadence_claim_pattern = re.compile(
        r"(?:\d+\s*분.*(?:간격|주기|단위|마다)|(?:간격|주기|단위|마다).*\d+\s*분)"
    )
    if not value or not cadence_claim_pattern.search(value):
        return value

    replacement = "반복 간격: 계산된 cadence/interval 값이 없어 단정하지 않음. 원문 타임스탬프 샘플만 근거로 반복 발생을 확인."
    cleaned_lines: list[str] = []
    for line in value.splitlines():
        if cadence_claim_pattern.search(line):
            indent = re.match(r"^\s*", line).group(0)
            stripped = line.strip()
            if stripped.startswith('"'):
                suffix = "," if stripped.endswith(",") else ""
                line = f'{indent}"{replacement}"{suffix}'
            else:
                line = f"{indent}- {replacement}"
            if cleaned_lines and replacement in cleaned_lines[-1]:
                continue
        cleaned_lines.append(line)
    return "\n".join(cleaned_lines)


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


def run_command(args: list[str], timeout: int = 8, env: dict[str, str] | None = None) -> dict[str, Any]:
    try:
        proc = subprocess.run(
            args,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
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


def nvidia_compat_env() -> dict[str, str] | None:
    compat_dir = os.getenv("NVIDIA_COMPAT_LIB_DIR", "").strip()
    if not compat_dir or not Path(compat_dir).is_dir():
        return None
    env = os.environ.copy()
    current_paths = [part for part in env.get("LD_LIBRARY_PATH", "").split(":") if part]
    if compat_dir not in current_paths:
        current_paths.insert(0, compat_dir)
    env["LD_LIBRARY_PATH"] = ":".join(current_paths)
    return env


def parse_gpu_csv(output: str) -> list[dict[str, Any]]:
    if not output:
        return []
    reader = csv.reader(io.StringIO(output))
    gpus: list[dict[str, Any]] = []
    for row in reader:
        if len(row) < 5:
            continue
        name, total, used, util, temp = [item.strip() for item in row[:5]]
        total_mib = int(total.split()[0]) if total else None
        used_mib = int(used.split()[0]) if used else None
        used_percent = round((used_mib / total_mib) * 100, 1) if total_mib and used_mib is not None else None
        gpus.append(
            {
                "name": name,
                "memory_total_mib": total_mib,
                "memory_used_mib": used_mib,
                "memory_used_percent": used_percent,
                "utilization_gpu_percent": int(util.split()[0]) if util else None,
                "temperature_c": int(temp.split()[0]) if temp else None,
            }
        )
    return gpus


def gpu_status() -> dict[str, Any]:
    args = [
        "nvidia-smi",
        "--query-gpu=name,memory.total,memory.used,utilization.gpu,temperature.gpu",
        "--format=csv,noheader",
    ]
    cmd = run_command(args, timeout=8)
    used_compat = False
    if not cmd["ok"]:
        compat_env = nvidia_compat_env()
        if compat_env:
            compat_cmd = run_command(args, timeout=8, env=compat_env)
            if compat_cmd["ok"]:
                cmd = compat_cmd
                used_compat = True
    return {
        "available": cmd["ok"],
        "gpus": parse_gpu_csv(cmd["stdout"]) if cmd["ok"] else [],
        "error": None if cmd["ok"] else cmd["stderr"],
        "compat_library_path": os.getenv("NVIDIA_COMPAT_LIB_DIR", "").strip() if used_compat else "",
    }


def ollama_request(path: str, payload: dict[str, Any] | None = None, timeout: int | None = None) -> dict[str, Any]:
    if not OLLAMA_ENABLED:
        return {"ok": False, "status": 503, "data": {"error": "Ollama disabled"}}
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


def safe_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def result_data(result: dict[str, Any]) -> dict[str, Any]:
    return safe_dict(result.get("data"))


def result_message_content(result: dict[str, Any]) -> str:
    data = result_data(result)
    message = safe_dict(data.get("message"))
    return str(message.get("content") or "")


def result_error_text(result: dict[str, Any]) -> str:
    data = result.get("data")
    if isinstance(data, dict):
        return str(data.get("error") or "")
    return str(data or "")


def read_runtime_env(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        return values
    except Exception:
        return values
    for raw_line in lines:
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            values[key] = value
    return values


def parse_runtime_int(value: Any) -> int | None:
    try:
        return int(str(value).strip())
    except Exception:
        return None


def parse_runtime_float(value: Any) -> float | None:
    try:
        return float(str(value).strip())
    except Exception:
        return None


def sglang_runtime_config() -> dict[str, Any]:
    env = read_runtime_env(SGLANG_RUNTIME_ENV_FILE)
    mem_fraction = parse_runtime_float(env.get("SGLANG_MEM_FRACTION_STATIC"))
    context_length = parse_runtime_int(env.get("SGLANG_CONTEXT_LENGTH"))
    return {
        "env_file": str(SGLANG_RUNTIME_ENV_FILE),
        "model_path": env.get("SGLANG_MODEL_PATH") or SGLANG_MODEL,
        "host": env.get("SGLANG_HOST") or "127.0.0.1",
        "port": parse_runtime_int(env.get("SGLANG_PORT")) or 30000,
        "context_length": context_length,
        "mem_fraction_static": mem_fraction,
        "target_vram_percent": round(mem_fraction * 100, 1) if mem_fraction is not None else None,
        "attention_backend": env.get("SGLANG_ATTENTION_BACKEND") or "",
        "sampling_backend": env.get("SGLANG_SAMPLING_BACKEND") or "",
        "extra_args": env.get("SGLANG_EXTRA_ARGS") or "",
    }


def nms_headers() -> dict[str, str]:
    return {"X-NMS-ERP-Context-Token": NMS_CONTEXT_TOKEN} if NMS_CONTEXT_TOKEN else {}


def nms_context_cache_snapshot() -> dict[str, Any]:
    with NMS_CONTEXT_CACHE_LOCK:
        return {
            "enabled": NMS_CONTEXT_CACHE_ENABLED,
            "ttl_seconds": NMS_CONTEXT_CACHE_TTL_SECONDS,
            "max_items": NMS_CONTEXT_CACHE_MAX_ITEMS,
            "items": len(NMS_CONTEXT_CACHE),
            **NMS_CONTEXT_CACHE_STATS,
        }


def nms_context_cache_get(key: str) -> dict[str, Any] | None:
    if not NMS_CONTEXT_CACHE_ENABLED or NMS_CONTEXT_CACHE_TTL_SECONDS <= 0:
        return None
    now = time.monotonic()
    with NMS_CONTEXT_CACHE_LOCK:
        cached = NMS_CONTEXT_CACHE.get(key)
        if not cached:
            NMS_CONTEXT_CACHE_STATS["misses"] += 1
            return None
        stored_at, value = cached
        if now - stored_at > NMS_CONTEXT_CACHE_TTL_SECONDS:
            NMS_CONTEXT_CACHE.pop(key, None)
            NMS_CONTEXT_CACHE_STATS["misses"] += 1
            return None
        NMS_CONTEXT_CACHE.move_to_end(key)
        NMS_CONTEXT_CACHE_STATS["hits"] += 1
        return json.loads(json.dumps(value, ensure_ascii=False))


def nms_context_cache_set(key: str, value: dict[str, Any]) -> None:
    if not NMS_CONTEXT_CACHE_ENABLED or NMS_CONTEXT_CACHE_TTL_SECONDS <= 0 or NMS_CONTEXT_CACHE_MAX_ITEMS <= 0:
        return
    with NMS_CONTEXT_CACHE_LOCK:
        NMS_CONTEXT_CACHE[key] = (time.monotonic(), json.loads(json.dumps(value, ensure_ascii=False)))
        NMS_CONTEXT_CACHE.move_to_end(key)
        NMS_CONTEXT_CACHE_STATS["stores"] += 1
        while len(NMS_CONTEXT_CACHE) > NMS_CONTEXT_CACHE_MAX_ITEMS:
            NMS_CONTEXT_CACHE.popitem(last=False)
            NMS_CONTEXT_CACHE_STATS["evictions"] += 1


def nms_get(path: str, params: dict[str, Any] | None = None, timeout: int | None = None) -> dict[str, Any]:
    if not NMS_CONTEXT_BASE_URL or not NMS_CONTEXT_TOKEN:
        return {"ok": False, "status": 503, "data": {"error": "NMS context integration is not configured"}}
    query = urllib.parse.urlencode({k: v for k, v in (params or {}).items() if v is not None and v != ""}, doseq=True)
    url = f"{NMS_CONTEXT_BASE_URL}{path}"
    if query:
        url = f"{url}?{query}"
    cache_key = f"{path}?{query}"
    cached = nms_context_cache_get(cache_key)
    if cached is not None:
        return {"ok": True, "status": 200, "data": cached, "cached": True}
    result = http_json_request(url, headers=nms_headers(), timeout=timeout or NMS_CONTEXT_TIMEOUT)
    if result.get("ok") and isinstance(result.get("data"), dict):
        nms_context_cache_set(cache_key, result["data"])
    return result


def ollama_status() -> dict[str, Any]:
    if not OLLAMA_ENABLED:
        return {
            "base_url": OLLAMA_BASE_URL,
            "reachable": False,
            "service_active": None,
            "models": [],
            "running": [],
            "error": {"error": "Ollama disabled"},
        }
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


def sglang_model_profile(model: str) -> dict[str, Any]:
    name = str(model or SGLANG_MODEL).strip() or SGLANG_MODEL
    return {
        "context_length": SGLANG_MODEL_CONTEXT_LENGTHS.get(name),
        "mem_fraction_static": SGLANG_MODEL_MEM_FRACTIONS.get(name),
    }


def sglang_model_entry(model: str | None = None) -> dict[str, Any]:
    name = str(model or SGLANG_MODEL).strip() or SGLANG_MODEL
    profile = sglang_model_profile(name)
    return {
        "name": name,
        "model": name,
        "provider": "sglang",
        "details": {
            "family": "SGLang",
            "base_url": SGLANG_BASE_URL,
            "context_length": profile.get("context_length"),
            "mem_fraction_static": profile.get("mem_fraction_static"),
        },
    }


def is_sglang_model(model: str) -> bool:
    return SGLANG_ENABLED and str(model or "").strip() in set(SGLANG_AVAILABLE_MODELS)


def resolve_model_alias(model: str) -> tuple[str, str]:
    requested = str(model or "").strip()
    return MODEL_ALIASES.get(requested, requested), requested


def openai_model_ids() -> list[str]:
    if PUBLIC_MODEL_IDS:
        return list(dict.fromkeys(PUBLIC_MODEL_IDS))
    ids: list[str] = []
    for name in [*MODEL_ALIASES.keys(), DEFAULT_MODEL, FAST_MODEL, *SGLANG_AVAILABLE_MODELS]:
        if name and name not in ids:
            ids.append(name)
    return ids


def sglang_status() -> dict[str, Any]:
    runtime_config = sglang_runtime_config()
    configured_models = [sglang_model_entry(model) for model in SGLANG_AVAILABLE_MODELS]
    if not SGLANG_ENABLED:
        return {
            "base_url": SGLANG_BASE_URL,
            "reachable": False,
            "models": [],
            "running": [],
            "runtime_config": runtime_config,
            "error": {"error": "SGLang disabled"},
        }
    health = http_json_request(f"{SGLANG_BASE_URL}/health", timeout=4)
    current_model = str(runtime_config.get("model_path") or SGLANG_MODEL)
    running = [sglang_model_entry(current_model)] if health["ok"] else []
    return {
        "base_url": SGLANG_BASE_URL,
        "reachable": health["ok"],
        "models": configured_models,
        "running": running,
        "runtime_config": runtime_config,
        "available_models": SGLANG_AVAILABLE_MODELS,
        "error": None if health["ok"] else health["data"],
    }


def ensure_sglang_running(warnings: list[str] | None = None) -> dict[str, Any]:
    status = sglang_status()
    if status["reachable"] or not SGLANG_ENABLED:
        return status
    if not SGLANG_START_SCRIPT or not Path(SGLANG_START_SCRIPT).exists():
        return status
    started = run_command([SGLANG_START_SCRIPT], timeout=20)
    if warnings is not None:
        if started["ok"]:
            warnings.append("SGLang 서버가 꺼져 있어 자동 시작했습니다.")
        else:
            warnings.append(f"SGLang 자동 시작 실패: {started['stderr'] or started['stdout']}")
    deadline = time.monotonic() + max(SGLANG_START_WAIT_SECONDS, 1)
    while time.monotonic() < deadline:
        status = sglang_status()
        if status["reachable"]:
            return status
        time.sleep(2)
    return sglang_status()


def write_sglang_runtime_env(updates: dict[str, Any]) -> None:
    current_lines = SGLANG_RUNTIME_ENV_FILE.read_text(encoding="utf-8").splitlines() if SGLANG_RUNTIME_ENV_FILE.exists() else []
    rendered: list[str] = []
    remaining = {key: str(value) for key, value in updates.items() if value is not None and str(value) != ""}
    for line in current_lines:
        if "=" not in line or line.lstrip().startswith("#"):
            rendered.append(line)
            continue
        key = line.split("=", 1)[0].strip()
        if key in remaining:
            rendered.append(f"{key}={remaining.pop(key)}")
        else:
            rendered.append(line)
    for key, value in remaining.items():
        rendered.append(f"{key}={value}")
    tmp_path = SGLANG_RUNTIME_ENV_FILE.with_name(f"{SGLANG_RUNTIME_ENV_FILE.name}.tmp-{uuid.uuid4().hex}")
    tmp_path.write_text("\n".join(rendered).rstrip() + "\n", encoding="utf-8")
    os.replace(tmp_path, SGLANG_RUNTIME_ENV_FILE)


def restart_sglang_runtime(warnings: list[str] | None = None) -> dict[str, Any]:
    main_pid = run_command(["systemctl", "show", "-p", "MainPID", "--value", "metro-sglang.service"], timeout=5)
    pid = (main_pid.get("stdout") or "").strip()
    if pid and pid != "0" and pid.isdigit():
        stop = run_command(["kill", "-TERM", pid], timeout=10)
    else:
        stop = {"ok": False, "returncode": 1, "stdout": "", "stderr": "SGLang MainPID not found"}
    if warnings is not None:
        if stop["ok"]:
            warnings.append("SGLang MainPID를 종료했고 systemd가 새 설정으로 재기동합니다.")
        elif stop["returncode"] == 1:
            warnings.append("실행 중인 SGLang 프로세스가 없어 새 기동 상태를 기다립니다.")
        else:
            warnings.append(f"SGLang 프로세스 종료 확인 필요: {stop['stderr'] or stop['stdout']}")
    deadline = time.monotonic() + max(SGLANG_START_WAIT_SECONDS, 1)
    while time.monotonic() < deadline:
        status = sglang_status()
        if status["reachable"]:
            return status
        time.sleep(3)
    return sglang_status()


def ensure_sglang_model(model: str, warnings: list[str] | None = None) -> dict[str, Any]:
    requested = str(model or SGLANG_MODEL).strip() or SGLANG_MODEL
    if not is_sglang_model(requested):
        return sglang_status()
    with SGLANG_SWITCH_LOCK:
        runtime = sglang_runtime_config()
        current_model = str(runtime.get("model_path") or SGLANG_MODEL)
        status = sglang_status()
        if current_model == requested and status.get("reachable"):
            return status
        if not SGLANG_AUTO_SWITCH_ENABLED:
            if warnings is not None and current_model != requested:
                warnings.append(
                    f"요청 모델 {requested}은 설치 후보지만 현재 실행 모델은 {current_model}입니다. "
                    "운영 안정성을 위해 요청 중 자동 재기동은 하지 않습니다."
                )
            if status.get("reachable"):
                return status
            return ensure_sglang_running(warnings)
        profile = sglang_model_profile(requested)
        updates: dict[str, Any] = {"SGLANG_MODEL_PATH": requested}
        if profile.get("context_length"):
            updates["SGLANG_CONTEXT_LENGTH"] = profile["context_length"]
        if profile.get("mem_fraction_static") is not None:
            updates["SGLANG_MEM_FRACTION_STATIC"] = profile["mem_fraction_static"]
        write_sglang_runtime_env(updates)
        if warnings is not None and current_model != requested:
            warnings.append(
                f"SGLang 모델을 {current_model}에서 {requested}(으)로 전환합니다. 최초 로딩은 수 분 걸릴 수 있습니다."
            )
        if status.get("reachable"):
            return restart_sglang_runtime(warnings)
        return ensure_sglang_running(warnings)


def stop_sglang_if_running(warnings: list[str] | None = None) -> None:
    if not SGLANG_ENABLED or not SGLANG_STOP_SCRIPT or not Path(SGLANG_STOP_SCRIPT).exists():
        return
    status = sglang_status()
    if not status["reachable"]:
        return
    result = run_command([SGLANG_STOP_SCRIPT], timeout=20)
    if warnings is not None:
        if result["ok"]:
            warnings.append("Ollama 모델 실행을 위해 SGLang 서버를 중지했습니다.")
        else:
            warnings.append(f"SGLang 중지 실패: {result['stderr'] or result['stdout']}")


def sglang_chat_request(payload: dict[str, Any], timeout: int | None = None) -> dict[str, Any]:
    result = http_json_request(
        f"{SGLANG_BASE_URL}/v1/chat/completions",
        payload,
        timeout=timeout or SGLANG_REQUEST_TIMEOUT,
    )
    if not result["ok"]:
        return result
    choices = result_data(result).get("choices") or []
    message = choices[0].get("message") if choices and isinstance(choices[0], dict) else {}
    content = message.get("content") if isinstance(message, dict) else ""
    return {
        "ok": True,
        "status": result["status"],
        "data": {
            "message": {"role": "assistant", "content": content or ""},
            "runtime": "sglang",
            "raw": result_data(result),
        },
    }


def is_ollama_cpu_only_model(model: str) -> bool:
    return str(model or "").strip() in OLLAMA_CPU_ONLY_MODELS


def should_keep_sglang_for_ollama_model(model: str) -> bool:
    name = str(model or "").strip()
    return name in OLLAMA_KEEP_SGLANG_MODELS or name in OLLAMA_CPU_ONLY_MODELS


def should_disable_ollama_think(model: str) -> bool:
    return str(model or "").strip() in OLLAMA_DISABLE_THINK_MODELS


def apply_ollama_model_policy(
    model: str,
    options: dict[str, Any] | None,
    warnings: list[str] | None = None,
) -> dict[str, Any]:
    name = str(model or "").strip()
    effective = dict(options or {})
    if not is_ollama_cpu_only_model(name):
        return effective

    max_ctx = OLLAMA_MODEL_NUM_CTX.get(name)
    if max_ctx:
        requested_ctx = int(effective.get("num_ctx") or 0)
        if requested_ctx <= 0 or requested_ctx > max_ctx:
            effective["num_ctx"] = max_ctx
    if name in OLLAMA_MODEL_NUM_GPU:
        effective["num_gpu"] = OLLAMA_MODEL_NUM_GPU[name]
    else:
        effective["num_gpu"] = 0
    if name in OLLAMA_MODEL_NUM_THREAD:
        effective["num_thread"] = OLLAMA_MODEL_NUM_THREAD[name]
    if "num_predict" not in effective and name in OLLAMA_MODEL_NUM_PREDICT:
        effective["num_predict"] = OLLAMA_MODEL_NUM_PREDICT[name]
    if warnings is not None:
        warnings.append(
            f"{name}은 정밀 Q4 CPU/RAM 모델로 실행합니다. GPU SGLang 모델은 유지하고 num_gpu={effective.get('num_gpu')}로 충돌을 피합니다."
        )
    return effective


def build_analysis_prompt(body: dict[str, Any]) -> list[dict[str, str]]:
    template = body.get("template") or "freeform"
    customer = body.get("customer") or "미지정 고객"
    question = body.get("question") or ""
    data = body.get("data") or body.get("logs") or ""
    saved_context = str(body.get("saved_analysis_context") or "").strip()
    attachments_context = str(body.get("attachments_context") or "").strip()
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
        f"{f'이전 저장 분석 참고:\\n{saved_context}\\n\\n' if saved_context else ''}"
        f"{f'외부 첨부자료 추출본:\\n{attachments_context}\\n\\n' if attachments_context else ''}"
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
    if is_field_analysis_evidence_pack(context):
        target = context.get("target") or {}
        matched = context.get("matched") or {}
        coverage = context.get("data_coverage") or {}
        analysis = context.get("deterministic_analysis") or {}
        compressed = context.get("compressed_evidence") or {}
        source_totals = compressed.get("source_totals") or {}
        signals = analysis.get("top_signals") or []
        gaps = analysis.get("gaps") or []
        lines = [
            f"- 분석 기준: {context.get('evidence_pack_version') or 'field-analysis-evidence'}. 33번 NMS 원천 evidence pack 기준이다.",
            (
                "- 선택 대상: "
                f"{target.get('target_name') or '-'} / type={target.get('target_type') or matched.get('target_type') or '-'} / "
                f"model={target.get('model_name') or target.get('model') or '-'}"
            ),
            (
                "- 위치/주소: "
                f"customer={matched.get('customer_name') or '-'}, site={matched.get('site_name') or matched.get('site_code') or '-'}, "
                f"public_ip={target.get('public_ip') or '-'}, private_ip={target.get('private_ip') or '-'}"
            ),
            (
                "- 커버리지: "
                f"pulse_obs={coverage.get('pulse_observation_count', 0)}, "
                f"pulse_metrics={coverage.get('pulse_metric_sample_count', 0)}, "
                f"syslog={coverage.get('syslog_sample_count', 0)}, "
                f"trap={coverage.get('trap_sample_count', 0)}, "
                f"diagnostics={coverage.get('diagnostic_sample_count', 0)}, "
                f"last_age_min={coverage.get('latest_age_minutes')}"
            ),
            (
                "- 원천 집계: "
                f"pulse={((source_totals.get('pulse_observations') or {}).get('total') or 0)}, "
                f"metrics={((source_totals.get('pulse_metrics') or {}).get('total') or 0)}, "
                f"collector_diag={((source_totals.get('collector_diagnostics') or {}).get('sample_count') or 0)}"
            ),
            f"- 상위 신호: {len(signals)}건",
        ]
        for item in signals[:8]:
            lines.append(
                f"  · {item.get('severity')} / {item.get('source_type')} / {item.get('signal_type')} / "
                f"{item.get('evidence') or '-'}"
            )
        if gaps:
            lines.append("- 누락/주의 데이터:")
            lines.extend([f"  · {item}" for item in gaps[:10]])
        return "\n".join(lines)

    if is_network_evidence_pack(context):
        matched = context.get("matched") or {}
        coverage = context.get("data_coverage") or {}
        analysis = context.get("deterministic_analysis") or {}
        compressed = context.get("compressed_evidence") or {}
        source_totals = compressed.get("source_totals") or {}
        aggregates = compressed.get("aggregates") or {}
        event_classification = context.get("event_classification") or analysis.get("event_classification") or {}
        event_summary = event_classification.get("summary") or {}
        active_categories = event_classification.get("active_categories") or []
        top_signals = analysis.get("top_signals") or []
        hypotheses = analysis.get("root_cause_hypotheses") or []
        gaps = analysis.get("gaps") or []
        lines = [
            f"- 분석 기준: {context.get('evidence_pack_version') or 'network evidence pack'}. NMS 요약은 참고 자료이며 최종 근거는 고객사별 원천 데이터 집계와 규칙 판정이다.",
            f"- 매칭 고객/현장: customers={matched.get('customer_count', 0)}, sites={matched.get('site_count', 0)}, site_codes={matched.get('site_codes') or []}",
            (
                "- 데이터 커버리지: "
                f"devices={coverage.get('device_count', 0)}, "
                f"syslog={coverage.get('syslog_sample_count', 0)}, "
                f"trap={coverage.get('trap_sample_count', 0)}, "
                f"interfaces={coverage.get('interface_sample_count', 0)}"
            ),
            (
                "- 기간 전체 SQL 집계: "
                f"syslog={((source_totals.get('syslog') or {}).get('total') or 0)}, "
                f"trap={((source_totals.get('snmp_trap') or {}).get('total') or 0)}, "
                f"polling={((source_totals.get('polling') or {}).get('total') or 0)}, "
                f"interface={((source_totals.get('interface_metric') or {}).get('total') or 0)}"
            ),
            f"- 상위 이상 신호: {len(top_signals)}건",
        ]
        if event_summary:
            lines.append(
                "- 이벤트 분류: "
                f"primary_category={event_summary.get('primary_category') or '-'}, "
                f"primary_event={event_summary.get('primary_event_type') or '-'}, "
                f"risk={event_summary.get('risk_level') or '-'}, "
                f"signals={event_summary.get('total_signal_count', 0)}"
            )
        if active_categories:
            lines.append("- 분류별 분석 레인:")
            for item in active_categories[:8]:
                lines.append(
                    f"  · {item.get('label') or item.get('category')}: "
                    f"{item.get('event_count', 0)}건 / max={item.get('max_severity') or '-'} / "
                    f"types={item.get('event_types') or {}}"
                )
        for item in top_signals[:8]:
            lines.append(
                f"  · {item.get('severity')} / {item.get('source_type')} / {item.get('signal_type')} / "
                f"{item.get('site_name') or '-'} / {item.get('device_name') or '-'} / {item.get('evidence') or '-'}"
            )
        top_apps = aggregates.get("syslog_top_apps") or []
        if top_apps:
            lines.append("- Syslog 앱별 상위:")
            for item in top_apps[:6]:
                lines.append(
                    f"  · {item.get('app_name')}: {item.get('count')}건 / "
                    f"last={item.get('last_seen_at')} / {item.get('sample_device') or '-'}"
                )
        if hypotheses:
            lines.append("- 원인 후보 초안:")
            for item in hypotheses[:5]:
                lines.append(f"  · {item.get('rank')}. {item.get('cause')} / confidence={item.get('confidence')} / {item.get('basis')}")
        if gaps:
            lines.append("- 누락/주의 데이터:")
            lines.extend([f"  · {item}" for item in gaps[:8]])
        return "\n".join(lines)

    matched = context.get("matched") or {}
    logs = context.get("logs") or {}
    traffic = context.get("traffic") or {}
    temporal = context.get("temporal") or {}
    off_hours = temporal.get("off_hours") or {}
    nas_file_activity = temporal.get("nas_file_activity") or {}
    nas_file_risk = temporal.get("nas_file_risk") or {}
    nas_ransomware_findings = temporal.get("nas_ransomware_findings") or []
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
        f"- NAS 랜섬웨어 규칙 경보: {len(nas_ransomware_findings)}건",
        f"- 이벤트 집중 시간대: {top_hours[:8]}",
    ]
    return "\n".join(lines)


def build_nms_deterministic_brief(context: dict[str, Any]) -> str:
    if is_field_analysis_evidence_pack(context):
        requested = context.get("requested") or {}
        matched = context.get("matched") or {}
        target = context.get("target") or {}
        coverage = context.get("data_coverage") or {}
        analysis = context.get("deterministic_analysis") or {}
        compressed = context.get("compressed_evidence") or {}
        source_totals = compressed.get("source_totals") or {}
        signals = analysis.get("top_signals") or []
        gaps = analysis.get("gaps") or []
        next_checks = analysis.get("required_next_checks") or []
        lines = [
            "현장 테스트 대상 Evidence Pack",
            (
                f"- 대상: {target.get('target_name') or '-'} "
                f"({target.get('model_name') or target.get('model') or '-'}) / type={requested.get('target_type') or matched.get('target_type') or '-'}"
            ),
            f"- 위치: {matched.get('customer_name') or '-'} / {matched.get('site_name') or matched.get('site_code') or '-'}",
            f"- 주소 분리: public_ip={target.get('public_ip') or '-'} / private_ip={target.get('private_ip') or '-'}",
            f"- 상태: {target.get('target_status') or '-'} / last_age_min={target.get('last_observed_age_minutes')}",
            (
                "- 커버리지: "
                f"Pulse observation={coverage.get('pulse_observation_count', 0)}, "
                f"Pulse metric={coverage.get('pulse_metric_sample_count', 0)}, "
                f"Syslog={coverage.get('syslog_sample_count', 0)}, "
                f"Trap={coverage.get('trap_sample_count', 0)}, "
                f"Diagnostic={coverage.get('diagnostic_sample_count', 0)}"
            ),
            (
                "- 기간 집계: "
                f"pulse_total={((source_totals.get('pulse_observations') or {}).get('total') or 0)}, "
                f"metric_total={((source_totals.get('pulse_metrics') or {}).get('total') or 0)}, "
                f"collector_diag={((source_totals.get('collector_diagnostics') or {}).get('sample_count') or 0)}"
            ),
        ]
        if signals:
            lines.append("- 상위 신호:")
            for item in signals[:10]:
                lines.append(f"  · {item.get('severity')} / {item.get('source_type')} / {item.get('signal_type')} / {item.get('evidence') or '-'}")
        if gaps:
            lines.append("- 누락/주의 데이터:")
            lines.extend([f"  · {item}" for item in gaps[:10]])
        if next_checks:
            lines.append("- 다음 확인 작업:")
            lines.extend([f"  · {item}" for item in next_checks[:8]])
        return "\n".join(lines)

    if is_network_evidence_pack(context):
        requested = context.get("requested") or {}
        matched = context.get("matched") or {}
        coverage = context.get("data_coverage") or {}
        analysis = context.get("deterministic_analysis") or {}
        compressed = context.get("compressed_evidence") or {}
        source_totals = compressed.get("source_totals") or {}
        aggregates = compressed.get("aggregates") or {}
        event_classification = context.get("event_classification") or analysis.get("event_classification") or {}
        event_summary = event_classification.get("summary") or {}
        active_categories = event_classification.get("active_categories") or []
        llm_focus = event_classification.get("llm_focus") or []
        signals = analysis.get("top_signals") or []
        timeline = analysis.get("timeline") or []
        hypotheses = analysis.get("root_cause_hypotheses") or []
        next_checks = analysis.get("required_next_checks") or []
        gaps = analysis.get("gaps") or []
        protocol = analysis.get("protocol_analysis") or {}
        dhcp = protocol.get("dhcp") or {}
        dhcp_summary = dhcp.get("summary") or {}
        dhcp_macs = dhcp.get("macs") or []
        lines = [
            "검증된 고객사별 네트워크 Evidence Pack",
            f"- 대상: {', '.join(matched.get('customer_names') or []) or requested.get('customer_name') or '전체'} / site_codes={matched.get('site_codes') or []}",
            f"- 분석 범위: 최근 {requested.get('hours', '-')}시간, from={requested.get('from', '-')}",
            "- 원칙: NMS 결과는 참고자료이며, 최종 판단은 원천 로그/수치/타임스탬프/규칙 판정에 둔다.",
            (
                "- 데이터 커버리지: "
                f"site={coverage.get('site_count', 0)}, device={coverage.get('device_count', 0)}, "
                f"syslog={coverage.get('syslog_sample_count', 0)}, trap={coverage.get('trap_sample_count', 0)}, "
                f"interface={coverage.get('interface_sample_count', 0)}"
            ),
            (
                "- 기간 전체 SQL 집계: "
                f"syslog={((source_totals.get('syslog') or {}).get('total') or 0)}, "
                f"trap={((source_totals.get('snmp_trap') or {}).get('total') or 0)}, "
                f"polling={((source_totals.get('polling') or {}).get('total') or 0)}, "
                f"interface={((source_totals.get('interface_metric') or {}).get('total') or 0)}"
            ),
        ]
        if event_summary:
            lines.append(
                "- 이벤트 분류 요약: "
                f"primary_category={event_summary.get('primary_category') or '-'}, "
                f"primary_event={event_summary.get('primary_event_type') or '-'}, "
                f"risk={event_summary.get('risk_level') or '-'}, "
                f"signals={event_summary.get('total_signal_count', 0)}"
            )
        if active_categories:
            lines.append("- 분류별 이벤트:")
            for item in active_categories[:8]:
                lines.append(
                    f"  · {item.get('label') or item.get('category')}: "
                    f"{item.get('event_count', 0)}건 / max={item.get('max_severity') or '-'} / "
                    f"types={item.get('event_types') or {}}"
                )
        if llm_focus:
            lines.append("- 118 LLM 세분화 지시:")
            for item in llm_focus[:8]:
                lines.append(f"  · {item.get('label') or item.get('category')}: {item.get('instruction') or '-'}")
        if dhcp_summary.get("total_events"):
            lines.append(
                "- DHCP 분석: "
                f"risk={dhcp_summary.get('risk_level')}, events={dhcp_summary.get('total_events')}, "
                f"unique_macs={dhcp_summary.get('unique_mac_count')}, note={dhcp_summary.get('note')}"
            )
            for item in dhcp_macs[:5]:
                lines.append(
                    f"  · MAC {item.get('mac')}: count={item.get('count')}, "
                    f"events={item.get('event_types')}, interfaces={item.get('interfaces')}, vlans={item.get('vlans')}"
                )
        if signals:
            lines.append("- 상위 이상 신호:")
            for item in signals[:10]:
                lines.append(
                    f"  · {item.get('severity')} / {item.get('source_type')} / {item.get('signal_type')} / "
                    f"{item.get('site_name') or '-'} / {item.get('device_name') or '-'} / {item.get('evidence') or '-'}"
                )
        top_apps = aggregates.get("syslog_top_apps") or []
        if top_apps:
            lines.append("- Syslog 앱별 상위:")
            for item in top_apps[:8]:
                lines.append(
                    f"  · {item.get('app_name')}: {item.get('count')}건 / "
                    f"last={item.get('last_seen_at')} / {item.get('sample_device') or '-'} / "
                    f"{clip_text(item.get('sample_message') or '', 90)}"
                )
        if hypotheses:
            lines.append("- 원인 후보 초안:")
            for item in hypotheses[:5]:
                lines.append(f"  · {item.get('rank')}. {item.get('cause')} / confidence={item.get('confidence')} / {item.get('basis')}")
        if timeline:
            lines.append("- 타임라인 샘플:")
            for item in timeline[:8]:
                lines.append(
                    f"  · {item.get('at')} / {item.get('source_type')} / {item.get('device_name') or '-'} / {item.get('message') or '-'}"
                )
        if next_checks:
            lines.append("- 다음 확인 작업:")
            lines.extend([f"  · {item}" for item in next_checks[:8]])
        if gaps:
            lines.append("- 누락/주의 데이터:")
            lines.extend([f"  · {item}" for item in gaps[:8]])
        return "\n".join(lines)

    requested = context.get("requested") or {}
    matched = context.get("matched") or {}
    logs = context.get("logs") or {}
    traffic = context.get("traffic") or {}
    temporal = context.get("temporal") or {}
    off_hours = temporal.get("off_hours") or {}
    nas_file_activity = temporal.get("nas_file_activity") or {}
    nas_file_risk = temporal.get("nas_file_risk") or {}
    nas_ransomware_findings = temporal.get("nas_ransomware_findings") or []
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
        f"- NAS 랜섬웨어 규칙 경보: {len(nas_ransomware_findings)}건",
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

    if nas_ransomware_findings:
        lines.append("- NAS 랜섬웨어 규칙 경보:")
        for item in nas_ransomware_findings[:5]:
            lines.append(
                f"  · {item.get('severity')} / {item.get('title')} / "
                f"{item.get('device_name') or '-'} / count={item.get('count')} / "
                f"last={item.get('last_seen_at')} / {item.get('summary') or item.get('sample_message') or '-'}"
            )

    if int(nas_file_risk.get("file_operation_count") or 0) == 0 and int(nas_file_risk.get("suspicious_keyword_count") or 0) == 0:
        lines.append("- 1차 판정: 새벽 시간대 랜섬웨어성 파일작업 직접 징후는 현재 관측되지 않음")
    if nas_ransomware_findings:
        lines.append("- 1차 판정: 33번 NMS 규칙 엔진이 랜섬웨어 조기탐지 조건을 충족한 경보를 관측함")
    if int(nas_file_activity.get("file_operation_count") or 0) >= 100:
        lines.append("- 1차 판정: 전체 시간대 파일명 변경이 많으므로 정상 업무/프로그램 임시파일 패턴인지 사용자와 경로 기준으로 확인 필요")

    return "\n".join(lines)


def build_nms_analysis_prompt(
    context: dict[str, Any],
    question: str = "",
    depth: str = "deep",
    saved_context: str = "",
    attachments_context: str = "",
    stored_evidence_context: str = "",
    evidence_snapshot: dict[str, Any] | None = None,
) -> list[dict[str, str]]:
    requested = context.get("requested") or {}
    matched = context.get("matched") or {}
    customer_names = ", ".join(matched.get("customer_names") or []) or requested.get("customer_name") or "전체"
    evidence_digest = build_nms_evidence_digest(context)
    llm_context = compact_nms_context_for_llm(context)
    evidence_limit = min(NMS_ANALYSIS_CONTEXT_CHAR_LIMIT, NMS_LLM_EVIDENCE_CHAR_LIMIT)
    if evidence_snapshot:
        evidence_limit = min(evidence_limit, NMS_EVIDENCE_STORE_CONTEXT_LIMIT)
    data_text = clip_text(json.dumps(llm_context, ensure_ascii=False, indent=2), evidence_limit)
    snapshot_notice = ""
    if evidence_snapshot:
        snapshot_notice = (
            "이번 원본 evidence는 118 서버 Evidence Store에 저장했다. "
            f"snapshot_id={evidence_snapshot.get('id')}, "
            f"full_chars={evidence_snapshot.get('full_context_chars')}, "
            f"compact_chars={evidence_snapshot.get('compact_context_chars')}. "
            "LLM 프롬프트에는 요약/축약 JSON만 포함되므로, 없는 원문 값을 상상하지 말고 "
            "필요하면 snapshot_id 기준으로 추가 조회가 필요하다고 표시한다."
        )
    if is_field_analysis_evidence_pack(context):
        command_set = context.get("codex_command_set") or {}
        system_prompt = (
            "너는 Codex가 지휘하는 현장 네트워크 테스트 장비 분석 보고 담당자다. "
            "NETSCOUT Pulse, Ubuntu Collector, 향후 고객사 현장 테스트 장비의 evidence pack만 근거로 판단한다. "
            "public_ip는 중앙 서버가 본 고객사 공인 IP이고 private_ip 또는 pulse_reported_ip는 장비/collector가 보고한 내부 IP다. "
            "두 주소를 같은 의미로 섞지 않는다. "
            "PoE, 링크, 포트, VLAN, LLDP, ARP, MAC, diagnostic 결과가 없으면 unknown 또는 자료 없음으로 표시한다. "
            "확정 사실/추정/반박 근거/누락 데이터/원격 확인/현장 확인/고객 보고 문안을 반드시 분리한다. "
            "JSON에 없는 결과를 만들지 말고, timestamp와 count가 있는 항목만 수치 근거로 인용한다."
        )
        command_prompt = command_set.get("recommended_prompt") or ""
    elif is_network_evidence_pack(context):
        command_set = context.get("codex_command_set") or {}
        system_prompt = (
            "너는 Codex가 지휘하는 고객사별 네트워크 장애 분석 보고 담당자다. "
            "NMS 결과를 결론으로 추론하지 말고, network evidence pack의 기간 전체 SQL 압축집계, "
            "대표 원천 샘플, 규칙 기반 신호, 타임라인, 누락 데이터, Codex 명령 셋을 근거로만 판단한다. "
            "확정 사실/추정/누락 데이터/원격 확인/현장 확인/고객 보고 문안을 반드시 분리한다. "
            "raw_evidence_samples는 샘플이고 기간 전체 판단은 compressed_evidence를 우선한다. "
            "DHCP DISCOVER/OFFER 반복만으로 보안 위협이나 CRITICAL을 단정하지 않는다. "
            "반복 간격/주기는 JSON에 계산된 cadence/interval 값이 있을 때만 말하고, 없으면 정확한 타임스탬프 샘플만 인용한다. "
            "traffic.interfaces 또는 interface_metrics가 없으면 트래픽 정상/비정상을 단정하지 않는다. "
            "event_classification이 있으면 llm_focus 지시를 우선 적용하고 네트워크 트래픽, 인터넷/WAN, 장비, 클라이언트, NAS 상태, NAS 파일변화, NAS collector를 별도 섹션으로 분리한다. "
            "JSON에 없는 내용은 단정하지 말고 '자료 없음'으로 표시한다."
        )
        command_prompt = command_set.get("recommended_prompt") or ""
    else:
        system_prompt = (
            "너는 33번 NMS 상시 관제와 ERP 업무 이력을 함께 보는 실무 분석 담당자다. "
            "NMS 시스로그, SNMP Trap, 장비 상태, 트래픽/프로브/수집기 값, Grafana가 참조하는 "
            "NMS 원천 데이터를 근거로 장애 징후를 판단한다. 답변은 빠른 추측이 아니라 근거 기반 "
            "심층 분석이어야 한다. JSON에 없는 내용은 단정하지 말고 '자료 없음' 또는 '추정'으로 "
            "표시해라. 같은 말을 반복하지 말고, 시간/장비/수치/로그 문구를 근거로 인용해라. "
            "숫자가 0인 항목은 반드시 '관측되지 않음'으로 해석하고, 없는 데이터를 만들어내지 마라. "
            "반복 간격/주기는 JSON에 계산된 cadence/interval 값이 있을 때만 말하고, 없으면 정확한 타임스탬프 샘플만 인용해라. "
            "DHCP DISCOVER/OFFER 반복만으로 보안 위협을 단정하지 말고, 데이터 없음과 정상을 분리해라."
        )
        command_prompt = ""
    user_prompt = (
        f"분석 대상: {customer_names}\n"
        f"분석 모드: {depth}\n"
        f"요청: {question or '최근 관제 데이터 기준으로 이상 징후와 조치 우선순위를 판단해줘.'}\n\n"
        f"{f'Evidence Store 안내:\\n{snapshot_notice}\\n\\n' if snapshot_notice else ''}"
        f"우선 적용할 근거 요약:\n{evidence_digest}\n\n"
        f"{f'Codex 표준 명령:\\n{command_prompt}\\n\\n' if command_prompt else ''}"
        f"{f'이전 저장 분석 참고:\\n{saved_context}\\n\\n' if saved_context else ''}"
        f"{f'저장 Evidence Snapshot 참고:\\n{stored_evidence_context}\\n\\n' if stored_evidence_context else ''}"
        f"{f'외부 첨부자료 추출본:\\n{attachments_context}\\n\\n' if attachments_context else ''}"
        f"LLM Safe Evidence Pack JSON:\n{data_text}\n\n"
        "분석 지침:\n"
        "- NMS 요약을 그대로 결론으로 쓰지 말고 고객사별 evidence pack의 원천 근거와 규칙 판정을 먼저 검증한다.\n"
        "- 먼저 compressed_evidence.source_totals와 aggregates를 보고 기간 전체 경향을 잡은 뒤 raw_evidence_samples로 대표 로그를 검증한다.\n"
        "- event_classification.summary, active_categories, top_events, llm_focus가 있으면 이것을 먼저 읽고 분류별 분석 순서를 정한다.\n"
        "- 네트워크 트래픽, 인터넷/WAN, 네트워크 장비, 클라이언트/단말, NAS 상태, NAS 파일변화, NAS collector, 데이터 정합성을 한 문단에 섞지 않는다.\n"
        "- raw_evidence_samples의 샘플 건수를 전체 건수로 해석하지 않는다.\n"
        "- 이전 저장 분석은 참고용이다. 현재 로그/트래픽과 다른 부분이 있으면 차이를 먼저 설명한다.\n"
        "- 첨부자료가 있으면 현재 NMS 원천값과 모순되는지 먼저 비교하고, 첨부자료의 수치/표/문장을 근거에 함께 반영한다.\n"
        "- temporal.off_hours와 temporal.nas_file_risk가 있으면 새벽 이벤트, 랜섬웨어성 파일 작업, 반복 시간대를 별도 판단한다.\n"
        "- temporal.nas_file_activity는 전체 시간대 NAS 파일작업이고, temporal.nas_file_risk는 새벽 시간대 파일작업이다. 둘을 반드시 분리해서 설명해라.\n"
        "- temporal.nas_ransomware_findings가 있으면 33번 NMS 규칙 엔진의 1차 판정이다. LLM 판단보다 우선 근거로 삼고, severity/title/count/sample_message를 그대로 인용해라.\n"
        "- Grafana 화면은 NMS DB를 시각화한 것이므로, Grafana 값이라고 표현할 때도 JSON의 traffic/logs/temporal 원천값을 근거로 삼는다.\n"
        "- temporal.top_event_hours는 이벤트 발생 시간 분포다. traffic.interfaces가 없으면 트래픽량으로 해석하지 마라.\n"
        "- deterministic_analysis.protocol_analysis.dhcp가 있으면 동일 MAC 반복, 다수 MAC 폭증, 다중 OFFER, gateway/DNS 변경, MAC 이동 후보를 분리해 해석한다.\n"
        "- 반복 간격, 주기, '몇 분 간격' 표현은 JSON에 cadence/interval로 계산된 값이 있을 때만 사용한다. 샘플 타임스탬프만 있으면 '반복 발생'으로 쓰고, 원문 시각을 그대로 인용한다.\n"
        "- DHCP DISCOVER/OFFER 반복만 있으면 정상 IP 할당 과정일 수 있으므로 INFO 또는 낮은 신뢰도 주의로만 본다. 다중 DHCP OFFER, gateway/DNS 변경, MAC 다중 VLAN/포트 이동, STP/포트 flap/트래픽 오류가 동반될 때만 위험도를 올린다.\n"
        "- SNMP Trap 0건은 장비 장애 근거가 아니라 trap 관측 없음이다. 데이터 없음과 정상은 다르다.\n"
        "- nas_file_risk.file_operation_count가 0이면 새벽 파일 삭제/이동/이름변경은 관측되지 않았다고 써라. 전체 시간대 파일작업 여부는 nas_file_activity로 따로 판단해라.\n"
        "- 이벤트가 많아도 정상 주기 작업일 가능성과 장애/보안 이벤트 가능성을 분리한다.\n"
        "- 근거가 부족하면 어떤 데이터가 더 필요한지 명확히 적는다.\n\n"
        "출력 형식:\n"
        "1. 종합 판정: 정상/주의/위험 중 하나와 신뢰도\n"
        "2. 핵심 근거: 시간, 장비명, 수치, 로그 문구를 포함해 5개 이상\n"
        "3. 이벤트 분류별 분석: 네트워크 트래픽/인터넷WAN/장비/클라이언트/NAS상태/NAS파일변화/NAS collector/데이터정합성을 분리\n"
        "4. 새벽 시간대 이벤트 분석: 집중 시간, 반복성, NAS 파일작업/랜섬웨어 의심 여부\n"
        "5. NMS/Grafana 관측값 해석: 장비 상태, 트래픽, 프로브, Trap/Syslog를 분리\n"
        "6. 의심 원인 우선순위: 가능성 높은 순서와 반박 근거\n"
        "7. 즉시 원격 확인 작업: 명령 또는 화면 경로 중심\n"
        "8. 현장 확인 필요 작업: 케이블/스위치/공유기/NAS/단말 기준\n"
        "9. 추가로 수집해야 할 데이터"
    )
    return [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]


def select_nms_analysis_profile(depth: str, context: dict[str, Any], explicit_model: Any = None) -> dict[str, Any]:
    if str(explicit_model or "").strip():
        return {
            "name": "explicit",
            "model": str(explicit_model).strip(),
            "num_ctx": NMS_ANALYSIS_NUM_CTX,
            "num_predict": NMS_ANALYSIS_NUM_PREDICT,
            "timeout": NMS_ANALYSIS_TIMEOUT,
            "reason": "request model was explicitly provided",
        }

    normalized_depth = str(depth or "").strip().lower()
    analysis = context.get("deterministic_analysis") or {}
    event_classification = context.get("event_classification") or analysis.get("event_classification") or {}
    event_summary = event_classification.get("summary") or {}
    total_signals = int(event_summary.get("total_signal_count") or len(analysis.get("top_signals") or []) or 0)
    risk_level = str(event_summary.get("risk_level") or "").lower()
    quick_depths = {"quick", "fast", "brief", "summary", "shallow", "간단", "요약", "빠른"}
    deep_depths = {"deep", "detailed", "field", "report", "심층", "정밀", "현장", "보고서"}
    low_signal = total_signals <= 2 and risk_level in {"", "ok", "info", "unknown"}

    if normalized_depth in quick_depths or (normalized_depth not in deep_depths and low_signal):
        return {
            "name": "fast",
            "model": NMS_FAST_ANALYSIS_MODEL,
            "num_ctx": min(NMS_ANALYSIS_NUM_CTX, NMS_FAST_ANALYSIS_NUM_CTX),
            "num_predict": min(NMS_ANALYSIS_NUM_PREDICT, NMS_FAST_ANALYSIS_NUM_PREDICT),
            "timeout": min(NMS_ANALYSIS_TIMEOUT, NMS_FAST_ANALYSIS_TIMEOUT),
            "reason": f"depth={depth or '-'}, signals={total_signals}, risk={risk_level or '-'}",
        }

    return {
        "name": "deep",
        "model": NMS_DEEP_ANALYSIS_MODEL,
        "num_ctx": NMS_ANALYSIS_NUM_CTX,
        "num_predict": NMS_ANALYSIS_NUM_PREDICT,
        "timeout": NMS_ANALYSIS_TIMEOUT,
        "reason": f"depth={depth or '-'}, signals={total_signals}, risk={risk_level or '-'}",
    }


def run_ollama_chat_messages(
    messages: list[dict[str, str]],
    model: str = DEFAULT_MODEL,
    timeout: int = REQUEST_TIMEOUT,
    options: dict[str, Any] | None = None,
    keep_alive: str = KEEP_ALIVE,
) -> dict[str, Any]:
    if not is_sglang_model(model) and not OLLAMA_ENABLED and SGLANG_ENABLED:
        model = DEFAULT_MODEL if is_sglang_model(DEFAULT_MODEL) else SGLANG_MODEL
    if is_sglang_model(model):
        warnings: list[str] = []
        messages = compact_messages_for_prompt_budget(messages, warnings)
        unload_ok = unload_running_models_except(model, warnings)
        if not unload_ok:
            return {
                "ok": False,
                "status": HTTPStatus.BAD_GATEWAY,
                "model": model,
                "elapsed_ms": 0,
                "response": "",
                "warnings": warnings,
                "raw": {"error": "ollama model is still running; SGLang start was skipped to avoid VRAM conflict"},
            }
        ensure_sglang_model(model, warnings)
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": False,
        }
        if options:
            if "temperature" in options:
                payload["temperature"] = options["temperature"]
            if "top_p" in options:
                payload["top_p"] = options["top_p"]
            if "num_predict" in options:
                payload["max_tokens"] = options["num_predict"]
        started = time.monotonic()
        result = sglang_chat_request(payload, timeout=timeout)
        elapsed_ms = int((time.monotonic() - started) * 1000)
        response_text = result_message_content(result)
        return {
            "ok": result["ok"],
            "status": result["status"],
            "model": model,
            "elapsed_ms": elapsed_ms,
            "response": response_text or "",
            "warnings": warnings,
            "raw": result.get("data") if isinstance(result.get("data"), dict) else {"error": result.get("data")},
        }

    warnings: list[str] = []
    options = apply_ollama_model_policy(model, options, warnings)
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "keep_alive": keep_alive,
    }
    if should_disable_ollama_think(model):
        payload["think"] = False
    if options:
        payload["options"] = options
    started = time.monotonic()
    result = ollama_request("/api/chat", payload, timeout=timeout)
    elapsed_ms = int((time.monotonic() - started) * 1000)
    response_text = result_message_content(result)
    return {
        "ok": result["ok"],
        "status": result["status"],
        "model": model,
        "elapsed_ms": elapsed_ms,
        "response": response_text or "",
        "warnings": warnings,
        "raw": result.get("data") if isinstance(result.get("data"), dict) else {"error": result.get("data")},
    }


def model_summary(model: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": model.get("name") or model.get("model"),
        "provider": model.get("provider") or "ollama",
        "size": model.get("size"),
        "modified_at": model.get("modified_at"),
        "details": model.get("details"),
        "expires_at": model.get("expires_at"),
        "size_vram": model.get("size_vram"),
    }


def unload_running_models_except(target_model: str, warnings: list[str] | None = None) -> bool:
    if not MODEL_AUTO_UNLOAD_BEFORE_SWITCH_ENABLED:
        return True
    target = str(target_model or "").strip()
    if not target:
        return True
    if not OLLAMA_ENABLED:
        return True
    target_is_sglang = is_sglang_model(target)
    status = ollama_status()
    running = [
        str(item.get("name") or item.get("model") or "").strip()
        for item in status.get("running") or []
        if str(item.get("name") or item.get("model") or "").strip()
    ]
    if target_is_sglang:
        sglang = sglang_status()
        current_sglang_model = str((sglang.get("runtime_config") or {}).get("model_path") or SGLANG_MODEL)
        if sglang.get("reachable") and current_sglang_model == target:
            if running and warnings is not None:
                warnings.append("SGLang 대상 모델이 이미 실행 중이라 잔류 Ollama 모델 언로드 대기를 건너뜁니다.")
            return True
    if not target_is_sglang and not should_keep_sglang_for_ollama_model(target):
        stop_sglang_if_running(warnings)
    for running_model in running:
        if running_model == target:
            continue
        if shutil.which("ollama"):
            stop_cmd = run_command(["ollama", "stop", running_model], timeout=MODEL_SWITCH_UNLOAD_TIMEOUT)
            result = {
                "ok": stop_cmd["ok"],
                "data": {"error": stop_cmd["stderr"] or stop_cmd["stdout"]},
            }
        else:
            result = ollama_request(
                "/api/generate",
                {"model": running_model, "prompt": "", "stream": False, "keep_alive": 0},
                timeout=MODEL_SWITCH_UNLOAD_TIMEOUT,
            )
        if warnings is not None:
            if result.get("ok"):
                warnings.append(f"선택 모델 전환을 위해 기존 실행 모델 {running_model}을 언로드했습니다.")
            else:
                error = result_error_text(result) or "알 수 없는 오류"
                warnings.append(f"기존 실행 모델 {running_model} 언로드 실패: {error}")
    if target_is_sglang and running:
        deadline = time.monotonic() + max(MODEL_SWITCH_UNLOAD_TIMEOUT, 1)
        while time.monotonic() < deadline:
            still_running = [
                str(item.get("name") or item.get("model") or "").strip()
                for item in ollama_status().get("running") or []
                if str(item.get("name") or item.get("model") or "").strip()
            ]
            if not still_running:
                return True
            time.sleep(2)
        if warnings is not None:
            warnings.append("Ollama 모델이 제한시간 안에 완전히 내려가지 않아 SGLang 시작을 보류했습니다.")
        return False
    return True


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
        saved_context = build_saved_analysis_context(customer_name, site_codes, limit=SAVED_ANALYSIS_CONTEXT_LIMIT)
        evidence_snapshot = create_nms_evidence_snapshot(
            context,
            customer_name=customer_name,
            site_codes=site_codes,
            question=question,
            depth="autopilot",
            hours=NMS_AUTOPILOT_WINDOW_HOURS,
        )
        stored_evidence_context = build_nms_evidence_store_context(
            customer_name,
            site_codes,
            limit=NMS_EVIDENCE_STORE_RECENT_LIMIT,
        )
        prompt_messages = build_nms_analysis_prompt(
            context,
            question,
            "deep",
            saved_context,
            "",
            stored_evidence_context,
            evidence_snapshot,
        )
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
        raw_response_text = chat.get("response") or ""
        response_text = sanitize_uncomputed_cadence_claims(raw_response_text)
        if response_text != raw_response_text:
            chat.setdefault("warnings", []).append("계산 근거 없는 반복 간격/주기 표현을 제거했습니다.")
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
                        f"snapshot_id={(evidence_snapshot or {}).get('id') or '-'}, "
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
            "evidence_snapshot_id": (evidence_snapshot or {}).get("id"),
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
                    "nms_fast_analysis_model": NMS_FAST_ANALYSIS_MODEL,
                    "nms_deep_analysis_model": NMS_DEEP_ANALYSIS_MODEL,
                    "nms_fast_analysis_num_ctx": NMS_FAST_ANALYSIS_NUM_CTX,
                    "nms_fast_analysis_num_predict": NMS_FAST_ANALYSIS_NUM_PREDICT,
                    "public_models": PUBLIC_MODEL_IDS,
                    "model_fallback_enabled": MODEL_FALLBACK_ENABLED,
                    "model_running_switch_guard_enabled": MODEL_RUNNING_SWITCH_GUARD_ENABLED,
                    "model_auto_unload_before_switch_enabled": MODEL_AUTO_UNLOAD_BEFORE_SWITCH_ENABLED,
                    "ollama_enabled": OLLAMA_ENABLED,
                    "ollama_base_url": OLLAMA_BASE_URL,
                    "sglang_enabled": SGLANG_ENABLED,
                    "sglang_base_url": SGLANG_BASE_URL,
                    "sglang_model": SGLANG_MODEL,
                    "sglang_available_models": SGLANG_AVAILABLE_MODELS,
                    "model_aliases": MODEL_ALIASES,
                    "sglang_auto_switch_enabled": SGLANG_AUTO_SWITCH_ENABLED,
                    "sglang_runtime": sglang_runtime_config(),
                    "ollama_cpu_only_models": sorted(OLLAMA_CPU_ONLY_MODELS),
                    "ollama_keep_sglang_models": sorted(OLLAMA_KEEP_SGLANG_MODELS),
                    "ollama_disable_think_models": sorted(OLLAMA_DISABLE_THINK_MODELS),
                    "default_num_ctx": LLM_OPS_DEFAULT_NUM_CTX,
                    "attachment_limits": {
                        "max_count": ATTACHMENT_MAX_COUNT,
                        "max_file_bytes": ATTACHMENT_MAX_FILE_BYTES,
                        "max_total_bytes": ATTACHMENT_MAX_TOTAL_BYTES,
                    },
                    "large_log_limits": {
                        "directory": str(LARGE_LOG_DIR),
                        "max_file_bytes": LARGE_LOG_MAX_FILE_BYTES,
                        "max_total_bytes": LARGE_LOG_MAX_TOTAL_BYTES,
                        "chunk_chars": LARGE_LOG_CHUNK_CHARS,
                        "max_chunks": LARGE_LOG_MAX_CHUNKS,
                    },
                    "nms_context_configured": bool(NMS_CONTEXT_BASE_URL and NMS_CONTEXT_TOKEN),
                    "nms_context_base_url": NMS_CONTEXT_BASE_URL,
                    "nms_context_cache": nms_context_cache_snapshot(),
                    "nms_evidence_store": {
                        "enabled": NMS_EVIDENCE_STORE_ENABLED,
                        "database": str(CONVERSATION_DB_PATH),
                        "prompt_context_limit": NMS_EVIDENCE_STORE_CONTEXT_LIMIT,
                        "max_full_chars": NMS_EVIDENCE_STORE_MAX_FULL_CHARS,
                        "recent_limit": NMS_EVIDENCE_STORE_RECENT_LIMIT,
                    },
                    "keep_alive": KEEP_ALIVE,
                    "time": utc_now(),
                }
            )
        if route == "/api/health":
            return self.json_response(self.health_payload())
        if route == "/api/sglang/runtime":
            if not self.authorized():
                return self.json_response({"error": "unauthorized"}, HTTPStatus.UNAUTHORIZED)
            return self.json_response(
                {
                    "ok": True,
                    "runtime": sglang_runtime_config(),
                    "status": sglang_status(),
                    "gpu": gpu_status(),
                    "time": utc_now(),
                }
            )
        if route == "/api/models":
            status = ollama_status()
            sglang = sglang_status()
            models = [model_summary(m) for m in status["models"]]
            if SGLANG_ENABLED:
                public_sglang_ids = PUBLIC_MODEL_IDS if PUBLIC_MODEL_IDS else [
                    str(item.get("name") or item.get("model") or "").strip()
                    for item in sglang["models"]
                    if str(item.get("name") or item.get("model") or "").strip()
                ]
                models.extend(model_summary(sglang_model_entry(model_id)) for model_id in public_sglang_ids)
            running = [model_summary(m) for m in status["running"]]
            running.extend(model_summary(m) for m in sglang["running"])
            return self.json_response(
                {
                    "ok": status["reachable"] or sglang["reachable"],
                    "models": models,
                    "running": running,
                    "default_model": DEFAULT_MODEL,
                    "fast_model": FAST_MODEL,
                    "sglang_model": SGLANG_MODEL if SGLANG_ENABLED else "",
                    "sglang_available_models": SGLANG_AVAILABLE_MODELS if SGLANG_ENABLED else [],
                    "public_models": PUBLIC_MODEL_IDS,
                    "model_aliases": MODEL_ALIASES,
                    "sglang_auto_switch_enabled": SGLANG_AUTO_SWITCH_ENABLED,
                    "precision_models": sorted(OLLAMA_CPU_ONLY_MODELS),
                    "ollama_cpu_only_models": sorted(OLLAMA_CPU_ONLY_MODELS),
                    "ollama_disable_think_models": sorted(OLLAMA_DISABLE_THINK_MODELS),
                    "model_fallback_enabled": MODEL_FALLBACK_ENABLED,
                    "model_running_switch_guard_enabled": MODEL_RUNNING_SWITCH_GUARD_ENABLED,
                    "model_auto_unload_before_switch_enabled": MODEL_AUTO_UNLOAD_BEFORE_SWITCH_ENABLED,
                    "error": None if status["reachable"] or sglang["reachable"] else {"ollama": status["error"], "sglang": sglang["error"]},
                },
                HTTPStatus.OK if status["reachable"] or sglang["reachable"] else HTTPStatus.BAD_GATEWAY,
            )
        if route == "/api/conversations" or route.startswith("/api/conversations/"):
            if not self.authorized():
                return self.json_response({"error": "unauthorized"}, HTTPStatus.UNAUTHORIZED)
            return self.handle_conversations_get(route, requestUrl=urllib.parse.urlparse(self.path))
        if route == "/api/saved-analyses" or route.startswith("/api/saved-analyses/"):
            if not self.authorized():
                return self.json_response({"error": "unauthorized"}, HTTPStatus.UNAUTHORIZED)
            return self.handle_saved_analyses_get(route, requestUrl=urllib.parse.urlparse(self.path))
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
        if route == "/api/saved-analyses":
            return self.handle_create_saved_analysis(body)
        if route == "/api/chat":
            attachment_context = self.prepare_attachment_context(body)
            if attachment_context is None:
                return
            body["attachment_context"] = attachment_context
            if attachment_context.get("text"):
                body["attachments_context"] = attachment_context.get("text") or ""
                if isinstance(body.get("messages"), list):
                    body["messages"] = [
                        {
                            "role": "user",
                            "content": (
                                "첨부자료 추출본을 먼저 참고해라. 현재 대화와 충돌하면 사용자의 최신 지시를 우선한다.\n\n"
                                f"{body['attachments_context']}"
                            ),
                        },
                        *[
                            message
                            for message in body.get("messages") or []
                            if isinstance(message, dict)
                        ],
                    ]
                else:
                    prompt = str(body.get("prompt") or "").strip()
                    body["prompt"] = (
                        f"첨부자료 추출본:\n{body['attachments_context']}\n\n"
                        f"사용자 입력:\n{prompt or '첨부자료를 참고해서 답해줘.'}"
                    )
            return self.handle_chat(body)
        if route == "/api/analyze":
            attachment_context = self.prepare_attachment_context(body)
            if attachment_context is None:
                return
            body["attachment_context"] = attachment_context
            body["attachments_context"] = attachment_context.get("text") or ""
            body["saved_analysis_context"] = build_saved_analysis_context(
                str(body.get("customer") or body.get("customer_name") or "").strip(),
                body.get("site_code") or body.get("site_codes") or "",
                limit=int(body.get("saved_analysis_limit") or SAVED_ANALYSIS_CONTEXT_LIMIT),
            )
            body["messages"] = build_analysis_prompt(body)
            return self.handle_chat(body)
        if route == "/api/large-log/analyze":
            return self.handle_large_log_analyze(body)
        if route == "/api/nms/analyze":
            return self.handle_nms_analyze(body)
        if route == "/api/nms/field-analyze":
            return self.handle_nms_field_analyze(body)
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
        if route.startswith("/api/saved-analyses/"):
            analysis_id = route.rsplit("/", 1)[-1]
            deleted = delete_saved_analysis(analysis_id)
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

    def handle_saved_analyses_get(self, route: str, requestUrl: urllib.parse.ParseResult) -> None:
        params = urllib.parse.parse_qs(requestUrl.query)
        first = lambda key, default="": (params.get(key) or [default])[0]
        if route == "/api/saved-analyses":
            items = list_saved_analyses(
                customer_name=first("customer_name"),
                site_code=first("site_code") or first("site_codes"),
                limit=int(first("limit", "50") or "50"),
                include_content=False,
            )
            return self.json_response({"success": True, "count": len(items), "items": items})
        analysis_id = route.rsplit("/", 1)[-1]
        item = load_saved_analysis(analysis_id)
        if not item:
            return self.json_response({"error": "saved analysis not found"}, HTTPStatus.NOT_FOUND)
        return self.json_response({"success": True, "item": item})

    def handle_create_saved_analysis(self, body: dict[str, Any]) -> None:
        try:
            item = create_saved_analysis(body)
        except ValueError as exc:
            return self.json_response({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
        return self.json_response({"success": True, "item": item}, HTTPStatus.CREATED)

    def prepare_attachment_context(self, body: dict[str, Any]) -> dict[str, Any] | None:
        try:
            context = extract_attachments_context(body.get("attachments") or [])
            if body.get("attachments") and not context.get("items"):
                detail = "; ".join(context.get("errors") or []) or "no readable attachments"
                raise ValueError(f"no readable attachments: {detail}")
            return context
        except ValueError as exc:
            self.json_response({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
            return None

    def handle_large_log_analyze(self, body: dict[str, Any]) -> None:
        try:
            context = extract_large_log_context(body.get("attachments") or [])
        except ValueError as exc:
            return self.json_response({"error": str(exc)}, HTTPStatus.BAD_REQUEST)

        requested_model = str(body.get("model") or DEFAULT_MODEL).strip() or DEFAULT_MODEL
        model, warnings = self.resolve_model_for_request(requested_model)
        text = str(context.get("text") or "")
        scan = large_log_scan(text)
        question = str(body.get("question") or body.get("prompt") or "").strip()
        customer_name = str(body.get("customer_name") or body.get("customer") or "").strip()
        site_code = str(body.get("site_code") or body.get("site_codes") or "").strip()
        analysis = run_large_log_chunk_analysis(
            text=text,
            scan=scan,
            model=model,
            question=question,
            customer_name=customer_name,
            site_code=site_code,
        )
        all_warnings = [*warnings, *(context.get("errors") or []), *(analysis.get("warnings") or [])]
        conversation_id = normalize_conversation_id(body.get("conversation_id") or body.get("thread_id") or "")
        remember = bool(conversation_id or body.get("remember") or body.get("persist"))
        if remember and analysis.get("response"):
            conversation = ensure_conversation(
                conversation_id,
                title=clip_text(question or f"{customer_name or '대용량 로그'} 분석", CONVERSATION_TITLE_CHAR_LIMIT),
                meta={"source": body.get("source") or "large-log-analysis", "customer_name": customer_name, "site_code": site_code},
            )
            conversation_id = conversation["id"]
            append_conversation_messages(
                conversation_id,
                [
                    {
                        "role": "user",
                        "content": (
                            f"대용량 로그 분석 요청: customer={customer_name or '-'}, site={site_code or '-'}, "
                            f"files={len(context.get('items') or [])}, chars={scan.get('chars')}, question={question or '-'}"
                        ),
                    },
                    {"role": "assistant", "content": str(analysis.get("response") or "")},
                ],
                meta={"model": analysis.get("model") or model, "source": "large-log-analysis", "elapsed_ms": analysis.get("elapsed_ms")},
            )
        payload = {
            "ok": bool(analysis.get("ok")),
            "model": analysis.get("model") or model,
            "model_requested": requested_model,
            "warnings": all_warnings,
            "conversation_id": conversation_id or None,
            "stored": {
                "request_id": context.get("request_id"),
                "directory": context.get("directory"),
                "items": context.get("items"),
                "total_bytes": context.get("total_bytes"),
            },
            "scan": analysis.get("scan"),
            "chunk_count_total": analysis.get("chunk_count_total"),
            "chunk_count_analyzed": analysis.get("chunk_count_analyzed"),
            "chunks": analysis.get("chunks"),
            "response": analysis.get("response"),
            "raw": analysis.get("raw"),
            "elapsed_ms": analysis.get("elapsed_ms"),
            "gpu": gpu_status(),
            "time": utc_now(),
        }
        return self.json_response(payload, HTTPStatus.OK if analysis.get("ok") else HTTPStatus.BAD_GATEWAY)

    def handle_openai_models(self) -> None:
        status = ollama_status()
        sglang = sglang_status()
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
        if SGLANG_ENABLED:
            for model_name in openai_model_ids():
                data.append({
                    "id": model_name,
                    "object": "model",
                    "created": 0,
                    "owned_by": "metro-sglang",
                })
        ok = status["reachable"] or sglang["reachable"] or bool(data)
        return self.json_response({"object": "list", "data": data}, HTTPStatus.OK if ok else HTTPStatus.BAD_GATEWAY)

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
            params = {
                "customer_name": first("customer_name"),
                "site_codes": first("site_codes"),
                "hours": first("hours", str(NMS_DEEP_ANALYSIS_WINDOW_HOURS)),
                "limit": first("limit", str(NMS_DEEP_ANALYSIS_LIMIT)),
            }
            result = nms_get(
                "/api/integrations/erp/network-evidence-pack",
                params,
                timeout=max(NMS_CONTEXT_TIMEOUT, 30),
            )
            if not result["ok"] and int(result.get("status") or 0) in {404, 501}:
                result = nms_get(
                    "/api/integrations/erp/nms-context",
                    params,
                    timeout=max(NMS_CONTEXT_TIMEOUT, 30),
                )
            return self.json_response(result["data"], HTTPStatus.OK if result["ok"] else HTTPStatus.BAD_GATEWAY)
        if route == "/api/nms/field-targets":
            result = nms_get(
                "/api/integrations/erp/field-analysis-targets",
                {
                    "q": first("q"),
                    "target_types": first("target_types"),
                    "limit": first("limit", "100"),
                },
                timeout=max(NMS_CONTEXT_TIMEOUT, 20),
            )
            return self.json_response(result["data"], HTTPStatus.OK if result["ok"] else HTTPStatus.BAD_GATEWAY)
        if route == "/api/nms/field-evidence":
            result = nms_get(
                "/api/integrations/erp/field-analysis-evidence",
                {
                    "target_type": first("target_type"),
                    "target_id": first("target_id"),
                    "hours": first("hours", "24"),
                    "limit": first("limit", "80"),
                    "from": first("from"),
                },
                timeout=max(NMS_CONTEXT_TIMEOUT, 30),
            )
            return self.json_response(result["data"], HTTPStatus.OK if result["ok"] else HTTPStatus.BAD_GATEWAY)
        if route == "/api/nms/evidence-snapshots":
            items = list_nms_evidence_snapshots(
                customer_name=first("customer_name"),
                site_codes=first("site_codes"),
                limit=int(first("limit", str(NMS_EVIDENCE_STORE_RECENT_LIMIT)) or str(NMS_EVIDENCE_STORE_RECENT_LIMIT)),
                include_full=first("include_full").lower() in {"1", "true", "yes", "on"},
            )
            return self.json_response({"success": True, "count": len(items), "items": items})
        if route.startswith("/api/nms/evidence-snapshots/"):
            snapshot_id = route.rsplit("/", 1)[-1]
            item = load_nms_evidence_snapshot(
                snapshot_id,
                include_full=first("include_full", "true").lower() not in {"0", "false", "no", "off"},
            )
            if not item:
                return self.json_response({"error": "evidence snapshot not found"}, HTTPStatus.NOT_FOUND)
            return self.json_response({"success": True, "item": item})
        if route == "/api/nms/operations":
            return self.json_response(operations_dashboard_payload())
        if route == "/api/nms/autopilot/history":
            limit = int(first("limit", "20") or "20")
            return self.json_response({"success": True, "items": list_autopilot_history(limit=limit)})
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

    def resolve_model_for_request(self, requested_model: str) -> tuple[str, list[str]]:
        requested = str(requested_model or DEFAULT_MODEL).strip() or DEFAULT_MODEL
        model, alias = resolve_model_alias(requested)
        warnings: list[str] = []
        if alias and alias != model:
            warnings.append(f"요청 모델 별칭 {alias}을 현재 운영 모델 {model}로 연결했습니다.")
        if is_sglang_model(model):
            current_model = str(sglang_runtime_config().get("model_path") or SGLANG_MODEL)
            if current_model != model and not SGLANG_AUTO_SWITCH_ENABLED:
                fallback = current_model if is_sglang_model(current_model) else SGLANG_MODEL
                warnings.append(
                    f"요청 모델 {model}은 현재 실행 중이 아니어서 활성 SGLang 모델 {fallback}로 실행했습니다."
                )
                return fallback, warnings
            return model, warnings
        if SGLANG_ENABLED and not OLLAMA_ENABLED:
            fallback = DEFAULT_MODEL if is_sglang_model(DEFAULT_MODEL) else SGLANG_MODEL
            warnings.append(f"Ollama가 비활성화되어 요청 모델 {model} 대신 SGLang 모델 {fallback}로 실행했습니다.")
            return fallback, warnings
        if not MODEL_FALLBACK_ENABLED or not DEFAULT_MODEL or model == DEFAULT_MODEL:
            return model, warnings

        status = ollama_status()
        available = {
            str(item.get("name") or item.get("model") or "").strip()
            for item in status.get("models") or []
            if str(item.get("name") or item.get("model") or "").strip()
        }
        running = {
            str(item.get("name") or item.get("model") or "").strip()
            for item in status.get("running") or []
            if str(item.get("name") or item.get("model") or "").strip()
        }
        if DEFAULT_MODEL not in available:
            return model, warnings
        if model not in available:
            warnings.append(f"요청 모델 {model}이 설치되어 있지 않아 운영 기본 모델 {DEFAULT_MODEL}로 실행했습니다.")
            return DEFAULT_MODEL, warnings
        if MODEL_RUNNING_SWITCH_GUARD_ENABLED and DEFAULT_MODEL in running and model not in running:
            warnings.append(
                f"현재 GPU에는 {DEFAULT_MODEL}가 실행 중입니다. 모델 전환 timeout을 피하기 위해 {model} 대신 운영 기본 모델로 실행했습니다."
            )
            return DEFAULT_MODEL, warnings
        return model, warnings

    def run_chat_completion(self, body: dict[str, Any]) -> tuple[dict[str, Any], HTTPStatus]:
        requested_model = str(body.get("model") or DEFAULT_MODEL).strip() or DEFAULT_MODEL
        model, warnings = self.resolve_model_for_request(requested_model)
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

        current_messages = ensure_default_chat_system_message(
            [
                {"role": str(message.get("role") or ""), "content": str(message.get("content") or "")}
                for message in messages
                if isinstance(message, dict) and str(message.get("content") or "")
            ]
        )
        messages = current_messages
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

        messages = compact_messages_for_prompt_budget(messages, warnings)
        options = body.get("options") if isinstance(body.get("options"), dict) else {}
        for key in ("temperature", "num_ctx", "num_predict", "top_p", "top_k", "repeat_penalty"):
            if key in body:
                options[key] = body[key]
        if "num_ctx" not in options and LLM_OPS_DEFAULT_NUM_CTX > 0:
            options["num_ctx"] = LLM_OPS_DEFAULT_NUM_CTX
        runtime = "sglang" if is_sglang_model(model) else "ollama"
        if runtime == "ollama":
            options = apply_ollama_model_policy(model, options, warnings)
        if runtime == "sglang":
            payload = {
                "model": model,
                "messages": messages,
                "stream": False,
            }
            if "temperature" in options:
                payload["temperature"] = options["temperature"]
            if "top_p" in options:
                payload["top_p"] = options["top_p"]
            max_tokens = options.get("num_predict") or body.get("max_tokens") or body.get("max_completion_tokens")
            if max_tokens:
                payload["max_tokens"] = max_tokens
        else:
            payload = {
                "model": model,
                "messages": messages,
                "stream": False,
                "keep_alive": body.get("keep_alive", KEEP_ALIVE),
            }
            if should_disable_ollama_think(model):
                payload["think"] = False
            if options:
                payload["options"] = options
        unload_ok = unload_running_models_except(model, warnings)
        if runtime == "sglang" and not unload_ok:
            return {
                "ok": False,
                "model": model,
                "model_requested": requested_model,
                "warnings": warnings,
                "error": "ollama model is still running; SGLang start was skipped to avoid VRAM conflict",
                "gpu": gpu_status(),
                "time": utc_now(),
            }, HTTPStatus.BAD_GATEWAY
        if runtime == "sglang":
            ensure_sglang_model(model, warnings)
        started = time.monotonic()
        if runtime == "sglang":
            result = sglang_chat_request(payload, timeout=int(body.get("timeout") or SGLANG_REQUEST_TIMEOUT))
        else:
            result = ollama_request("/api/chat", payload, timeout=int(body.get("timeout") or REQUEST_TIMEOUT))
        elapsed_ms = int((time.monotonic() - started) * 1000)
        response_text = result_message_content(result)
        if not result["ok"] and MODEL_FALLBACK_ENABLED and model != DEFAULT_MODEL and DEFAULT_MODEL:
            error_text = result_error_text(result)
            if runtime == "sglang" or "timed out" in error_text.lower():
                warnings.append(f"{model} 응답 실패로 {DEFAULT_MODEL}로 한 번 더 실행했습니다.")
                model = DEFAULT_MODEL
                runtime = "sglang" if is_sglang_model(model) else "ollama"
                payload["model"] = model
                if runtime == "sglang":
                    payload.pop("keep_alive", None)
                    payload.pop("options", None)
                    if "temperature" in options:
                        payload["temperature"] = options["temperature"]
                    if "top_p" in options:
                        payload["top_p"] = options["top_p"]
                    max_tokens = options.get("num_predict") or body.get("max_tokens") or body.get("max_completion_tokens")
                    if max_tokens:
                        payload["max_tokens"] = max_tokens
                    ensure_sglang_model(model, warnings)
                else:
                    payload["keep_alive"] = body.get("keep_alive", KEEP_ALIVE)
                    payload["options"] = options
                started = time.monotonic()
                unload_running_models_except(model, warnings)
                if runtime == "sglang":
                    result = sglang_chat_request(payload, timeout=int(body.get("timeout") or SGLANG_REQUEST_TIMEOUT))
                else:
                    result = ollama_request("/api/chat", payload, timeout=int(body.get("timeout") or REQUEST_TIMEOUT))
                elapsed_ms = int((time.monotonic() - started) * 1000)
                response_text = result_message_content(result)
        if result["ok"] and should_retry_korean_response(messages, response_text):
            warnings.append("응답이 한국어가 아닌 언어로 시작되어 한국어 재작성 요청을 자동 실행했습니다.")
            retry_payload = dict(payload)
            retry_payload["messages"] = [
                {
                    "role": "system",
                    "content": (
                        "방금 응답은 한국어가 아닌 언어로 시작했다. 최종 답변은 반드시 한국어만 사용한다. "
                        "중국어/영어 표현 없이 자연스러운 한국어로 다시 작성한다."
                    ),
                },
                *messages,
                {"role": "user", "content": "위 요청에 대한 최종 답변을 한국어로만 다시 작성해라."},
            ]
            started = time.monotonic()
            if runtime == "sglang":
                retry_result = sglang_chat_request(retry_payload, timeout=int(body.get("timeout") or SGLANG_REQUEST_TIMEOUT))
            else:
                retry_result = ollama_request("/api/chat", retry_payload, timeout=int(body.get("timeout") or REQUEST_TIMEOUT))
            retry_elapsed_ms = int((time.monotonic() - started) * 1000)
            retry_response = result_message_content(retry_result)
            if retry_result.get("ok") and retry_response:
                result = retry_result
                elapsed_ms += retry_elapsed_ms
                response_text = retry_response
        sanitized_response = sanitize_korean_response(response_text)
        if sanitized_response != (response_text or ""):
            warnings.append("최종 응답 앞부분의 비한국어 문장을 제거하고 한국어 답변만 표시했습니다.")
            response_text = sanitized_response
            if isinstance(result.get("data"), dict) and isinstance(result["data"].get("message"), dict):
                result["data"]["message"]["content"] = response_text
        if body.get("forbid_uncomputed_cadence"):
            cadence_sanitized_response = sanitize_uncomputed_cadence_claims(response_text)
            if cadence_sanitized_response != (response_text or ""):
                warnings.append("계산 근거 없는 반복 간격/주기 표현을 제거했습니다.")
                response_text = cadence_sanitized_response
                if isinstance(result.get("data"), dict) and isinstance(result["data"].get("message"), dict):
                    result["data"]["message"]["content"] = response_text
        prepend_report = str(body.get("prepend_report") or "").strip()
        if prepend_report and response_text:
            response_text = f"{prepend_report}\n\n---\n\nLLM 심층 분석\n{response_text}"
            if isinstance(result.get("data"), dict) and isinstance(result["data"].get("message"), dict):
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
            "model_requested": requested_model,
            "warnings": warnings,
            "elapsed_ms": elapsed_ms,
            "conversation_id": conversation_id or None,
            "message": result_data(result).get("message"),
            "response": response_text,
            "raw": result.get("data") if isinstance(result.get("data"), dict) else {"error": result.get("data")},
            "gpu": gpu_status(),
            "attachment_summary": (body.get("attachment_context") or {}).get("items") if isinstance(body.get("attachment_context"), dict) else [],
            "attachment_errors": (body.get("attachment_context") or {}).get("errors") if isinstance(body.get("attachment_context"), dict) else [],
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
            "warnings": payload.get("warnings") or [],
            "model_requested": payload.get("model_requested"),
            "gpu": payload.get("gpu"),
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

        context_params = {
            "customer_name": customer_name,
            "site_codes": site_codes,
            "hours": body.get("hours") or default_hours,
            "limit": body.get("limit") or default_limit,
        }
        result = nms_get(
            "/api/integrations/erp/network-evidence-pack",
            context_params,
            timeout=max(NMS_CONTEXT_TIMEOUT, 30),
        )
        if not result["ok"] and int(result.get("status") or 0) in {404, 501}:
            result = nms_get(
                "/api/integrations/erp/nms-context",
                context_params,
                timeout=max(NMS_CONTEXT_TIMEOUT, 30),
            )
        if not result["ok"]:
            return self.json_response(result_data(result) or {"error": result.get("data")}, HTTPStatus.BAD_GATEWAY)

        context_data = result_data(result)
        matched = safe_dict(context_data.get("matched"))
        effective_customer_name = customer_name or ", ".join((matched.get("customer_names") or []))
        effective_site_codes = site_codes or ",".join(str(value) for value in (matched.get("site_codes") or []) if value)
        evidence_snapshot = create_nms_evidence_snapshot(
            context_data,
            customer_name=effective_customer_name,
            site_codes=effective_site_codes,
            question=str(body.get("question") or ""),
            depth=depth,
            hours=context_params.get("hours") or default_hours,
        )
        stored_evidence_context = build_nms_evidence_store_context(
            effective_customer_name,
            effective_site_codes,
            limit=int(body.get("evidence_snapshot_limit") or NMS_EVIDENCE_STORE_RECENT_LIMIT),
        )
        attachment_context = self.prepare_attachment_context(body)
        if attachment_context is None:
            return
        body["attachment_context"] = attachment_context
        body["attachments_context"] = attachment_context.get("text") or ""
        body["saved_analysis_context"] = build_saved_analysis_context(
            effective_customer_name,
            effective_site_codes,
            limit=int(body.get("saved_analysis_limit") or SAVED_ANALYSIS_CONTEXT_LIMIT),
        )
        body["messages"] = build_nms_analysis_prompt(
            context_data,
            str(body.get("question") or ""),
            depth,
            str(body.get("saved_analysis_context") or ""),
            str(body.get("attachments_context") or ""),
            stored_evidence_context,
            evidence_snapshot,
        )
        body["prepend_report"] = build_nms_deterministic_brief(context_data)
        body["forbid_uncomputed_cadence"] = True
        body["conversation_store_messages"] = [
            {
                "role": "user",
                "content": (
                    f"NMS 심층 분석 요청: customer={customer_name or '-'}, "
                    f"site_codes={site_codes or '-'}, hours={body.get('hours') or default_hours}, "
                    f"snapshot_id={(evidence_snapshot or {}).get('id') or '-'}, "
                    f"question={str(body.get('question') or '').strip() or '-'}"
                ),
            }
        ]
        analysis_profile = select_nms_analysis_profile(depth, context_data, body.get("model"))
        body.setdefault("model", analysis_profile["model"])
        body.setdefault("temperature", 0.08)
        body.setdefault("top_p", 0.82)
        body.setdefault("repeat_penalty", NMS_ANALYSIS_REPEAT_PENALTY)
        body.setdefault("num_ctx", analysis_profile["num_ctx"])
        body.setdefault("num_predict", analysis_profile["num_predict"])
        body.setdefault("timeout", analysis_profile["timeout"])
        payload, status = self.run_chat_completion(body)
        payload["nms_analysis_profile"] = analysis_profile
        payload["evidence_snapshot"] = evidence_snapshot
        if evidence_snapshot:
            payload.setdefault("warnings", []).append(
                f"NMS 원본 evidence를 118 Evidence Store snapshot_id={evidence_snapshot.get('id')}에 저장하고 LLM에는 축약본만 전달했습니다."
            )
        return self.json_response(payload, status)

    def handle_nms_field_analyze(self, body: dict[str, Any]) -> None:
        target_type = str(body.get("target_type") or "").strip()
        target_id = str(body.get("target_id") or "").strip()
        if not target_type or not target_id:
            return self.json_response({"error": "target_type and target_id are required"}, HTTPStatus.BAD_REQUEST)

        depth = str(body.get("depth") or body.get("analysis_depth") or "field").strip().lower()
        context_params = {
            "target_type": target_type,
            "target_id": target_id,
            "hours": body.get("hours") or 24,
            "limit": body.get("limit") or 80,
            "from": body.get("from") or "",
        }
        result = nms_get(
            "/api/integrations/erp/field-analysis-evidence",
            context_params,
            timeout=max(NMS_CONTEXT_TIMEOUT, 30),
        )
        if not result["ok"]:
            return self.json_response(result_data(result) or {"error": result.get("data")}, HTTPStatus.BAD_GATEWAY)

        context_data = result_data(result)
        matched = safe_dict(context_data.get("matched"))
        target = safe_dict(context_data.get("target"))
        effective_customer_name = str(matched.get("customer_name") or target.get("customer_name") or "").strip()
        effective_site_codes = ",".join(str(value) for value in (matched.get("site_codes") or []) if value)
        if not effective_site_codes and matched.get("site_code"):
            effective_site_codes = str(matched.get("site_code"))
        default_question = (
            "선택한 현장 테스트 장비/Collector의 최근 evidence를 기준으로 고객사 네트워크 장애 여부, "
            "누락 데이터, 현장 확인 항목, 고객 보고 문안을 정리해줘."
        )
        question = str(body.get("question") or default_question).strip()
        evidence_snapshot = create_nms_evidence_snapshot(
            context_data,
            customer_name=effective_customer_name,
            site_codes=effective_site_codes,
            question=question,
            depth=depth,
            hours=context_params.get("hours") or 24,
        )
        stored_evidence_context = build_nms_evidence_store_context(
            effective_customer_name,
            effective_site_codes,
            limit=int(body.get("evidence_snapshot_limit") or NMS_EVIDENCE_STORE_RECENT_LIMIT),
        )
        attachment_context = self.prepare_attachment_context(body)
        if attachment_context is None:
            return
        body["attachment_context"] = attachment_context
        body["attachments_context"] = attachment_context.get("text") or ""
        body["saved_analysis_context"] = build_saved_analysis_context(
            effective_customer_name,
            effective_site_codes,
            limit=int(body.get("saved_analysis_limit") or SAVED_ANALYSIS_CONTEXT_LIMIT),
        )
        body["messages"] = build_nms_analysis_prompt(
            context_data,
            question,
            depth,
            str(body.get("saved_analysis_context") or ""),
            str(body.get("attachments_context") or ""),
            stored_evidence_context,
            evidence_snapshot,
        )
        body["prepend_report"] = build_nms_deterministic_brief(context_data)
        body["forbid_uncomputed_cadence"] = True
        body["conversation_store_messages"] = [
            {
                "role": "user",
                "content": (
                    f"현장 테스트 대상 분석 요청: target_type={target_type}, target_id={target_id}, "
                    f"target_name={target.get('target_name') or '-'}, hours={context_params.get('hours')}, "
                    f"snapshot_id={(evidence_snapshot or {}).get('id') or '-'}, question={question}"
                ),
            }
        ]
        analysis_profile = select_nms_analysis_profile(depth, context_data, body.get("model"))
        body.setdefault("model", analysis_profile["model"])
        body.setdefault("temperature", 0.08)
        body.setdefault("top_p", 0.82)
        body.setdefault("repeat_penalty", NMS_ANALYSIS_REPEAT_PENALTY)
        body.setdefault("num_ctx", analysis_profile["num_ctx"])
        body.setdefault("num_predict", analysis_profile["num_predict"])
        body.setdefault("timeout", analysis_profile["timeout"])
        payload, status = self.run_chat_completion(body)
        payload["nms_analysis_profile"] = analysis_profile
        payload["evidence_snapshot"] = evidence_snapshot
        site_names = matched.get("site_names") if isinstance(matched.get("site_names"), list) else []
        payload["field_target"] = {
            "target_type": target_type,
            "target_id": target_id,
            "target_name": target.get("target_name") or target.get("name"),
            "model_name": target.get("model_name") or target.get("model"),
            "customer_name": effective_customer_name,
            "site_name": target.get("site_name") or matched.get("site_name") or (site_names[0] if site_names else ""),
            "site_codes": effective_site_codes,
            "target_status": target.get("target_status") or target.get("status"),
            "public_ip": target.get("public_ip"),
            "private_ip": target.get("private_ip"),
        }
        if evidence_snapshot:
            payload.setdefault("warnings", []).append(
                f"현장 테스트 대상 evidence를 118 Evidence Store snapshot_id={evidence_snapshot.get('id')}에 저장하고 LLM에는 축약본만 전달했습니다."
            )
        return self.json_response(payload, status)

    def handle_model_lifecycle(self, body: dict[str, Any], keep_alive: str | int) -> None:
        model = str(body.get("model") or DEFAULT_MODEL)
        warnings: list[str] = []
        if is_sglang_model(model) or (SGLANG_ENABLED and not OLLAMA_ENABLED):
            if not is_sglang_model(model):
                fallback = DEFAULT_MODEL if is_sglang_model(DEFAULT_MODEL) else SGLANG_MODEL
                warnings.append(f"Ollama가 비활성화되어 요청 모델 {model} 대신 SGLang 모델 {fallback}로 처리했습니다.")
                model = fallback
            if keep_alive == 0 or str(keep_alive) == "0":
                stop_sglang_if_running(warnings)
            else:
                ensure_sglang_model(model, warnings)
            status = sglang_status()
            return self.json_response(
                {
                    "ok": status["reachable"] or (keep_alive == 0 or str(keep_alive) == "0"),
                    "model": model,
                    "keep_alive": keep_alive,
                    "warnings": warnings,
                    "sglang": status,
                    "running": status.get("running"),
                    "gpu": gpu_status(),
                    "time": utc_now(),
                },
                HTTPStatus.OK if status["reachable"] or (keep_alive == 0 or str(keep_alive) == "0") else HTTPStatus.BAD_GATEWAY,
            )
        if not OLLAMA_ENABLED:
            return self.json_response(
                {
                    "ok": False,
                    "model": model,
                    "keep_alive": keep_alive,
                    "warnings": warnings,
                    "error": "Ollama is disabled and requested model is not the configured SGLang model.",
                    "sglang_model": SGLANG_MODEL if SGLANG_ENABLED else "",
                    "gpu": gpu_status(),
                    "time": utc_now(),
                },
                HTTPStatus.BAD_GATEWAY,
            )
        if keep_alive != 0 and str(keep_alive) != "0":
            unload_running_models_except(model, warnings)
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
                "warnings": warnings,
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
                "nms_fast_analysis_model": NMS_FAST_ANALYSIS_MODEL,
                "nms_deep_analysis_model": NMS_DEEP_ANALYSIS_MODEL,
                "nms_fast_analysis_num_ctx": NMS_FAST_ANALYSIS_NUM_CTX,
                "nms_fast_analysis_num_predict": NMS_FAST_ANALYSIS_NUM_PREDICT,
                "public_models": PUBLIC_MODEL_IDS,
                "model_fallback_enabled": MODEL_FALLBACK_ENABLED,
                "model_running_switch_guard_enabled": MODEL_RUNNING_SWITCH_GUARD_ENABLED,
                "model_auto_unload_before_switch_enabled": MODEL_AUTO_UNLOAD_BEFORE_SWITCH_ENABLED,
                "default_num_ctx": LLM_OPS_DEFAULT_NUM_CTX,
                "ollama_enabled": OLLAMA_ENABLED,
                "ollama_cpu_only_models": sorted(OLLAMA_CPU_ONLY_MODELS),
                "ollama_keep_sglang_models": sorted(OLLAMA_KEEP_SGLANG_MODELS),
                "ollama_disable_think_models": sorted(OLLAMA_DISABLE_THINK_MODELS),
                "sglang_available_models": SGLANG_AVAILABLE_MODELS,
                "model_aliases": MODEL_ALIASES,
                "sglang_auto_switch_enabled": SGLANG_AUTO_SWITCH_ENABLED,
                "large_log": {
                    "directory": str(LARGE_LOG_DIR),
                    "max_file_bytes": LARGE_LOG_MAX_FILE_BYTES,
                    "max_total_bytes": LARGE_LOG_MAX_TOTAL_BYTES,
                    "chunk_chars": LARGE_LOG_CHUNK_CHARS,
                    "max_chunks": LARGE_LOG_MAX_CHUNKS,
                },
            },
            "nms_context_cache": nms_context_cache_snapshot(),
            "nms_monitor": NMS_MONITOR.snapshot(),
            "nms_autopilot": NMS_AUTOPILOT.snapshot(),
            "ollama": ollama_status(),
            "sglang": sglang_status(),
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
    .muted { color: var(--muted); font-size: 12px; }
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
    button:disabled { opacity: 0.45; cursor: not-allowed; }
    details.advanced {
      border: 1px solid var(--line);
      border-radius: 16px;
      padding: 10px 12px;
      background: rgba(255, 255, 255, 0.04);
    }
    details.advanced summary {
      cursor: pointer;
      color: var(--muted);
      font-size: 13px;
      font-weight: 700;
    }
    details.advanced .row { margin-top: 10px; }
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
    .ops-list { display: grid; gap: 8px; max-height: 360px; overflow: auto; padding-right: 2px; }
    .ops-item {
      display: grid;
      gap: 8px;
      padding: 11px;
      border: 1px solid var(--line);
      border-radius: 16px;
      background: rgba(255, 255, 255, 0.045);
    }
    .ops-title { font-weight: 800; letter-spacing: -0.02em; }
    .ops-meta, .ops-preview, .ops-empty { color: var(--muted); font-size: 12px; line-height: 1.5; }
    .ops-preview { color: #c7d7d4; }
    .ops-actions { display: flex; gap: 8px; flex-wrap: wrap; }
    .ops-actions button { min-width: 0; padding: 8px 10px; font-size: 12px; }
    .ops-section-title { color: var(--muted); font-size: 12px; font-weight: 800; margin-top: 2px; }
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
      <div class="subtitle">흐름: 모델 선택 → 업체/현장 선택 → 분석 기능 선택 → 분석 실행 → 이어 묻기/결과 저장</div>
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
            <select id="model" onchange="rememberSelectedModel()"></select>
          </div>
          <div>
            <label>분석 기능</label>
            <select id="template">
              <option value="nms_deep">업체 NMS 심층분석</option>
              <option value="field_target_report">현장 테스트 장비 분석</option>
              <option value="nms_context">선택 데이터 보기</option>
              <option value="maintenance_report">유지보수 보고서</option>
              <option value="customer_history">고객사 이력 요약</option>
              <option value="freeform">본문/첨부 일반분석</option>
              <option value="codex_task">Codex 작업 지시 초안</option>
            </select>
          </div>
        </div>
        <div class="row">
          <button onclick="saveToken()">토큰 저장</button>
          <button class="secondary" onclick="newConversation()">새 대화</button>
          <button class="secondary" onclick="refreshAll()">상태/목록 새로고침</button>
        </div>
        <details class="advanced">
          <summary>고급 모델 작업</summary>
          <div class="row">
            <button class="secondary" onclick="loadConversations()">대화 새로고침</button>
            <button class="secondary" onclick="preload()">프리로드</button>
            <button class="secondary" onclick="unload()">언로드</button>
            <button class="secondary" onclick="benchmark()">벤치마크</button>
          </div>
        </details>
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
            <input id="nmsCustomerSearch" placeholder="업체명, 코드, 현장명으로 검색" oninput="renderNmsCustomers()" />
            <select id="nmsCustomer" onchange="selectNmsCustomer()">
              <option value="">업체 목록 불러오기 필요</option>
            </select>
            <small id="nmsCustomerSearchStatus" class="muted">검색어 없이 전체 표시</small>
          </div>
          <div>
            <label>현장</label>
            <select id="nmsSite" onchange="selectNmsSite()">
              <option value="">전체 현장</option>
            </select>
          </div>
        </div>
        <div class="row">
          <button id="nmsBackButton" class="secondary" onclick="goBackNmsMenu()" disabled>이전 메뉴</button>
          <button class="secondary" onclick="loadNmsCustomers()">업체/상태 새로고침</button>
          <button class="secondary" onclick="toggleNmsMonitor()">상태카드 자동갱신</button>
        </div>
        <div id="nmsMonitorSummary" class="metric">
          <span>모니터링 상태</span><b>확인 전</b>
        </div>
      </div>
      <div class="card stack">
        <div class="row" style="justify-content:space-between">
          <strong>현장 테스트 분석 대상</strong>
          <span id="fieldTargetPill" class="pill warn">대기</span>
        </div>
        <div>
          <label>대상 검색</label>
          <input id="fieldTargetSearch" placeholder="NETSCOUT, Collector, 고객사, 현장명" oninput="renderFieldTargets()" />
          <select id="fieldTargetSelect" onchange="selectFieldTarget()">
            <option value="">대상 목록 불러오기 필요</option>
          </select>
          <small id="fieldTargetStatus" class="muted">NETSCOUT Pulse와 Ubuntu Collector를 별도 분석 대상으로 선택합니다.</small>
        </div>
        <div class="row">
          <button class="secondary" onclick="loadFieldTargets()">대상 새로고침</button>
          <button class="secondary" onclick="loadFieldEvidence()">선택 근거 보기</button>
          <button onclick="analyzeFieldTarget()">선택 대상 보고서</button>
        </div>
      </div>
      <div class="card stack">
        <div class="row" style="justify-content:space-between">
          <strong>운영 데이터 열람</strong>
          <span id="opsDataPill" class="pill warn">대기</span>
        </div>
        <div id="opsDataSummary" class="metric">
          <span>자동분석/저장 분석</span><b style="font-size:14px">확인 전</b>
        </div>
        <div class="row">
          <button class="secondary" onclick="loadOperationsDashboard()">운영 데이터 새로고침</button>
          <button class="secondary" onclick="toggleOpsAutoRefresh()">운영 데이터 자동갱신</button>
          <button class="secondary" onclick="loadNmsContext()">선택 원천 데이터 보기</button>
        </div>
        <div class="ops-section-title">자동분석 이력</div>
        <div id="opsAutopilotList" class="ops-list">
          <div class="ops-empty">토큰 저장 후 자동분석 이력을 불러옵니다.</div>
        </div>
        <div class="ops-section-title">최근 저장 분석</div>
        <div id="opsSavedList" class="ops-list">
          <div class="ops-empty">저장된 분석 결과가 여기에 표시됩니다.</div>
        </div>
      </div>
      <div class="card stack">
        <div class="row" style="justify-content:space-between">
          <strong>업체/현장 분석 보관함</strong>
          <span id="savedAnalysisCount" class="pill warn">0건</span>
        </div>
        <div class="grid2">
          <div>
            <label>저장 제목</label>
            <input id="savedAnalysisTitle" placeholder="예: 돈우 5월 장애 분석 1차" />
          </div>
          <div>
            <label>저장 분석</label>
            <select id="savedAnalysisSelect">
              <option value="">고객사/현장 선택 후 불러오기</option>
            </select>
          </div>
        </div>
        <div id="savedAnalysisStatus" class="metric">
          <span>현재 범위</span><b style="font-size:14px">고객사/현장을 선택하세요.</b>
        </div>
        <div class="row">
          <button class="secondary" onclick="loadSavedAnalyses()">목록 새로고침</button>
          <button class="secondary" onclick="saveCurrentAnalysis()">현재 결과 저장</button>
          <button class="secondary" onclick="loadSavedAnalysis()">선택 불러오기</button>
          <button class="secondary" onclick="deleteSavedAnalysis()">선택 삭제</button>
        </div>
      </div>
    </section>
    <section class="stack">
      <div class="card stack">
        <div class="grid2">
          <div>
            <label>고객사</label>
            <input id="customer" placeholder="예: (주)농업회사법인돈우" onchange="loadSavedAnalyses(true)" />
          </div>
          <div>
            <label>요청 / 후속 질문</label>
            <input id="question" placeholder="예: 최근 24시간 장애 가능성을 판단해줘. 이후에는 여기에 이어서 질문합니다." />
          </div>
        </div>
        <div>
          <label>추가 자료 / 로그 / 메모</label>
          <textarea id="prompt" placeholder="필요할 때만 NMS 이벤트, 작업일지, 로그, 첨부 설명을 넣습니다. 업체 NMS 분석은 비워도 됩니다."></textarea>
        </div>
        <div class="grid2">
          <div>
            <label>외부 첨부자료 (CSV, XLS, XLSX, DOC, DOCX, TXT)</label>
            <input id="analysisAttachments" type="file" multiple accept=".csv,.xls,.xlsx,.doc,.docx,.txt,.log" onchange="renderAttachmentSummary()" />
          </div>
          <div id="attachmentSummary" class="metric">
            <span>첨부 상태</span><b style="font-size:14px">선택된 파일 없음</b>
          </div>
        </div>
        <div class="row">
          <button onclick="runSelectedAnalysis()">분석 실행</button>
          <button class="secondary" onclick="analyzeLargeLog()">대용량 로그 분석</button>
          <button class="secondary" onclick="continueConversation()">이어 묻기</button>
          <button class="secondary" onclick="saveCurrentAnalysis()">결과 저장</button>
          <button class="secondary" onclick="buildCodexTaskDraft()">Codex 지시 초안</button>
          <button class="secondary" onclick="clearAttachments()">첨부 비우기</button>
        </div>
      </div>
      <div class="card result"><pre id="result">대기 중입니다.</pre></div>
    </section>
  </main>
  <script>
    const $ = (id) => document.getElementById(id);
    const tokenKey = "metro_llm_ops_token";
    const conversationKey = "metro_llm_ops_conversation_id";
    const maxAttachmentCount = 5;
    const maxAttachmentFileBytes = 5 * 1024 * 1024;
    const maxAttachmentTotalBytes = 12 * 1024 * 1024;
    const maxLargeLogFileBytes = 64 * 1024 * 1024;
    const maxLargeLogTotalBytes = 128 * 1024 * 1024;
    let nmsCustomers = [];
    let fieldTargets = [];
    let nmsMonitorTimer = null;
    let healthTimer = null;
    let conversations = [];
    let savedAnalyses = [];
    let operationsDashboard = null;
    let opsRefreshTimer = null;
    let nmsNavigationStack = [];
    let currentNmsMenuState = null;
    let isRestoringNmsMenu = false;
    let currentConversationId = localStorage.getItem(conversationKey) || "";
    $("token").value = localStorage.getItem(tokenKey) || "";

    function saveToken() {
      localStorage.setItem(tokenKey, $("token").value.trim());
      setResult("토큰을 브라우저에 저장했습니다.");
      loadConversations();
      loadNmsCustomers();
      refreshNmsMonitor();
      loadOperationsDashboard(true);
      loadSavedAnalyses(true);
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

    function escapeHtml(value) {
      return String(value ?? "").replace(/[&<>"']/g, (ch) => ({
        "&": "&amp;",
        "<": "&lt;",
        ">": "&gt;",
        '"': "&quot;",
        "'": "&#39;",
      }[ch]));
    }

    function compactTime(value) {
      return String(value || "").replace("T", " ").replace("Z", "").slice(0, 16) || "-";
    }

    function formatBytes(bytes) {
      if (!bytes) return "0 B";
      const units = ["B", "KB", "MB", "GB"];
      let value = bytes;
      let unitIndex = 0;
      while (value >= 1024 && unitIndex < units.length - 1) {
        value /= 1024;
        unitIndex += 1;
      }
      return `${value.toFixed(value >= 10 || unitIndex === 0 ? 0 : 1)} ${units[unitIndex]}`;
    }

    function attachmentFiles() {
      return Array.from($("analysisAttachments").files || []);
    }

    function attachmentSummaryText() {
      return attachmentFiles().map((file) => `${file.name} (${formatBytes(file.size)})`).join(", ");
    }

    function renderAttachmentSummary(message = "") {
      const files = attachmentFiles();
      const totalBytes = files.reduce((sum, file) => sum + (file.size || 0), 0);
      const text = files.length
        ? `${files.length}개 / ${formatBytes(totalBytes)} · ${attachmentSummaryText()}`
        : (message || "선택된 파일 없음");
      $("attachmentSummary").innerHTML = `
        <span>첨부 상태</span>
        <b style="font-size:14px">${text}</b>
      `;
    }

    function clearAttachments() {
      $("analysisAttachments").value = "";
      renderAttachmentSummary("선택된 파일 없음");
    }

    function arrayBufferToBase64(buffer) {
      const bytes = new Uint8Array(buffer);
      const chunkSize = 0x8000;
      let binary = "";
      for (let index = 0; index < bytes.length; index += chunkSize) {
        binary += String.fromCharCode(...bytes.subarray(index, index + chunkSize));
      }
      return btoa(binary);
    }

    async function collectAttachmentPayloads(mode = "normal") {
      const files = attachmentFiles();
      if (!files.length) return [];
      const maxFileBytes = mode === "large-log" ? maxLargeLogFileBytes : maxAttachmentFileBytes;
      const maxTotalBytes = mode === "large-log" ? maxLargeLogTotalBytes : maxAttachmentTotalBytes;
      if (files.length > maxAttachmentCount) {
        throw new Error(`첨부파일은 최대 ${maxAttachmentCount}개까지 가능합니다.`);
      }
      const totalBytes = files.reduce((sum, file) => sum + (file.size || 0), 0);
      if (totalBytes > maxTotalBytes) {
        throw new Error(`첨부파일 전체 크기는 ${formatBytes(maxTotalBytes)} 이하여야 합니다.`);
      }
      const payloads = [];
      for (const file of files) {
        if ((file.size || 0) > maxFileBytes) {
          throw new Error(`${file.name} 파일이 너무 큽니다. 파일당 ${formatBytes(maxFileBytes)} 이하만 가능합니다.`);
        }
        const buffer = await file.arrayBuffer();
        payloads.push({
          name: file.name,
          type: file.type || "",
          size: file.size || 0,
          content_base64: arrayBufferToBase64(buffer),
        });
      }
      return payloads;
    }

    function renderResponseWithAttachmentWarnings(data) {
      let text = data.response || "";
      if (!text && data.ok === false) {
        const rawError = data.raw?.error || data.error || "알 수 없는 오류";
        text = `LLM 응답 실패: ${rawError}`;
      }
      if (!text) text = JSON.stringify(data, null, 2);
      if (data.model || data.model_requested) {
        const requested = data.model_requested || data.model || "운영 기본 모델";
        const used = data.model || "알 수 없음";
        text = `[모델]\n- 요청: ${requested}\n- 실행: ${used}\n\n${text}`;
      }
      if (Array.isArray(data.warnings) && data.warnings.length) {
        text = `[실행 보정]\n- ${data.warnings.join("\n- ")}\n\n${text}`;
      }
      if (Array.isArray(data.attachment_errors) && data.attachment_errors.length) {
        text += `\n\n[첨부 처리 경고]\n- ${data.attachment_errors.join("\n- ")}`;
      }
      return text;
    }

    function describeApiError(data, res) {
      if (!data) return `HTTP ${res.status} ${res.statusText}`;
      if (typeof data.error === "string") return data.error;
      if (data.error?.message) return data.error.message;
      if (data.raw?.error) return data.raw.error;
      if (data.response) return data.response;
      return `HTTP ${res.status} ${JSON.stringify(data)}`;
    }

    async function api(path, options = {}) {
      const res = await fetch(path, options);
      const text = await res.text();
      let data = {};
      try {
        data = text ? JSON.parse(text) : {};
      } catch (err) {
        data = {error: text || String(err)};
      }
      if (!res.ok) throw new Error(describeApiError(data, res));
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

    function selectedSiteDetails() {
      const customer = selectedCustomer();
      const siteCode = $("nmsSite").value;
      if (!customer || !siteCode) {
        return {site_code: siteCode || "", site_name: ""};
      }
      const site = (customer.sites || []).find((item) => item.site_code === siteCode) || {};
      return {
        site_code: siteCode,
        site_name: site.site_name || "",
      };
    }

    function analysisScope() {
      const customerName = $("customer").value.trim() || (selectedCustomer()?.customer_name || "").trim();
      const site = selectedSiteDetails();
      return {
        customer_name: customerName,
        site_code: site.site_code || "",
        site_name: site.site_name || "",
      };
    }

    function updateSavedAnalysisStatus(extra = "") {
      const scope = analysisScope();
      const customerLabel = scope.customer_name || "고객사 미지정";
      const siteLabel = scope.site_name || scope.site_code || "전체 현장";
      $("savedAnalysisCount").className = savedAnalyses.length ? "pill" : "pill warn";
      $("savedAnalysisCount").textContent = `${savedAnalyses.length}건`;
      $("savedAnalysisStatus").innerHTML = `
        <span>현재 범위</span>
        <b style="font-size:14px">${customerLabel} / ${siteLabel}${extra ? ` · ${extra}` : ""}</b>
      `;
    }

    function renderSavedAnalyses(selectedId = "") {
      const select = $("savedAnalysisSelect");
      select.innerHTML = '<option value="">저장 분석 선택</option>';
      for (const item of savedAnalyses) {
        const opt = document.createElement("option");
        opt.value = String(item.id);
        const scope = item.site_name || item.site_code || "전체";
        const updatedAt = String(item.updated_at || "").replace("T", " ").replace("Z", "").slice(0, 16);
        opt.textContent = `${item.title || "저장 분석"} / ${scope} / ${updatedAt}`;
        if (String(item.id) === String(selectedId)) opt.selected = true;
        select.appendChild(opt);
      }
      updateSavedAnalysisStatus(savedAnalyses.length ? `최근 ${savedAnalyses[0].updated_at || "-"}` : "저장 없음");
    }

    async function loadSavedAnalyses(silent = false) {
      const scope = analysisScope();
      if (!scope.customer_name) {
        savedAnalyses = [];
        renderSavedAnalyses();
        updateSavedAnalysisStatus("고객사 입력 필요");
        return;
      }
      try {
        const query = new URLSearchParams({
          customer_name: scope.customer_name,
          limit: "50",
        });
        if (scope.site_code) query.set("site_code", scope.site_code);
        const data = await api(`/api/saved-analyses?${query.toString()}`, {headers: headers()});
        const selectedId = $("savedAnalysisSelect").value;
        savedAnalyses = data.items || [];
        renderSavedAnalyses(selectedId);
        if (!silent) {
          updateSavedAnalysisStatus(savedAnalyses.length ? "저장 분석을 불러왔습니다." : "저장 분석이 없습니다.");
        }
      } catch (err) {
        if (!silent) setResult(`저장 분석을 불러오지 못했습니다.\n${String(err)}`);
        console.warn("saved analysis load failed", err);
      }
    }

    async function saveCurrentAnalysis() {
      const scope = analysisScope();
      const content = $("result").textContent.trim();
      if (!scope.customer_name) {
        setResult("저장할 고객사를 먼저 선택하거나 입력하세요.");
        return;
      }
      if (!content || content === "대기 중입니다." || content === "분석 중..." || content === "생성 중...") {
        setResult("저장할 분석 결과가 없습니다.");
        return;
      }
      const title = $("savedAnalysisTitle").value.trim() || $("question").value.trim() || `${scope.customer_name} 분석`;
      const attachmentNote = attachmentSummaryText();
      const data = await api("/api/saved-analyses", {
        method: "POST",
        headers: headers(),
        body: JSON.stringify({
          customer_name: scope.customer_name,
          site_code: scope.site_code,
          site_name: scope.site_name,
          analysis_kind: $("template").value,
          title,
          question: $("question").value.trim(),
          content,
          source_excerpt: [
            attachmentNote ? `[첨부파일]\n${attachmentNote}` : "",
            $("prompt").value.trim(),
          ].filter(Boolean).join("\n\n"),
          meta: {
            model: $("model").value,
            template: $("template").value,
            conversation_id: currentConversationId || "",
            source: "llm-ops-ui",
          },
        }),
      });
      $("savedAnalysisTitle").value = data.item.title || title;
      await loadSavedAnalyses(true);
      renderSavedAnalyses(data.item.id);
      updateSavedAnalysisStatus("현재 결과를 저장했습니다.");
    }

    async function loadSavedAnalysis() {
      const analysisId = $("savedAnalysisSelect").value;
      if (!analysisId) {
        setResult("불러올 저장 분석을 선택하세요.");
        return;
      }
      const data = await api(`/api/saved-analyses/${analysisId}`, {headers: headers()});
      const item = data.item || {};
      if (item.customer_name) {
        $("customer").value = item.customer_name;
        const customer = nmsCustomers.find((entry) => entry.customer_name === item.customer_name);
        if (customer) {
          $("nmsCustomer").value = customer.customer_name;
          selectNmsCustomer();
          if (item.site_code) {
            $("nmsSite").value = item.site_code;
            selectNmsSite();
          }
        }
      }
      if (item.analysis_kind && [...$("template").options].some((opt) => opt.value === item.analysis_kind)) {
        $("template").value = item.analysis_kind;
      }
      $("savedAnalysisTitle").value = item.title || "";
      $("question").value = item.question || $("question").value;
      $("prompt").value = item.source_excerpt || $("prompt").value;
      setResult(item.content || "저장 내용이 없습니다.");
      updateSavedAnalysisStatus(`선택 분석 ${item.id}번을 불러왔습니다.`);
      await loadSavedAnalyses(true);
      renderSavedAnalyses(item.id);
    }

    async function deleteSavedAnalysis() {
      const analysisId = $("savedAnalysisSelect").value;
      if (!analysisId) {
        setResult("삭제할 저장 분석을 선택하세요.");
        return;
      }
      await api(`/api/saved-analyses/${analysisId}`, {
        method: "DELETE",
        headers: headers(),
      });
      await loadSavedAnalyses(true);
      $("savedAnalysisTitle").value = "";
      updateSavedAnalysisStatus("선택 저장 분석을 삭제했습니다.");
    }

    function opsTargetLabel(item) {
      const target = item?.target || {};
      return [
        target.customer_name || item.customer_name || "",
        target.site_name || target.site_code || item.site_name || item.site_code || "",
      ].filter(Boolean).join(" / ") || item.title || "대상 미확인";
    }

    function renderOperationsDashboard(data) {
      const monitor = data.monitor || {};
      const autopilot = data.autopilot || {};
      const summary = monitor.summary || {};
      const autoSummary = autopilot.summary || {};
      const history = data.autopilot_history || [];
      const saved = data.saved_analyses || [];
      const store = data.conversation_store || {};
      $("opsDataPill").className = monitor.ok ? "pill" : "pill warn";
      $("opsDataPill").textContent = monitor.ok ? "열람 가능" : "NMS 확인 필요";
      $("opsDataSummary").innerHTML = `
        <span>생성 ${compactTime(data.generated_at)} · 자동 ${compactTime(autopilot.last_run_at)}</span>
        <b style="font-size:14px">
          고객 ${summary.customer_count || 0} · 이벤트 ${summary.recent_event_count || 0}
          · 자동분석 ${history.length}건 · 저장 ${store.saved_analyses || saved.length || 0}건
          · 후보 ${autoSummary.candidate_count || 0}
        </b>
      `;
      $("opsAutopilotList").innerHTML = history.length ? history.map((item) => `
        <div class="ops-item">
          <div class="ops-title">${escapeHtml(opsTargetLabel(item))}</div>
          <div class="ops-meta">
            ${escapeHtml(compactTime(item.updated_at))} · 메시지 ${item.message_count || 0} · ${escapeHtml(item.id)}
          </div>
          <div class="ops-preview">${escapeHtml(item.latest_assistant_preview || item.latest_user || "최근 자동분석 내용 없음")}</div>
          <div class="ops-actions">
            <button class="secondary" onclick="openAutopilotConversation('${escapeHtml(item.id)}')">대화 열기</button>
          </div>
        </div>
      `).join("") : '<div class="ops-empty">자동분석 대화 이력이 아직 없습니다.</div>';
      $("opsSavedList").innerHTML = saved.length ? saved.slice(0, 8).map((item) => `
        <div class="ops-item">
          <div class="ops-title">${escapeHtml(item.title || "저장 분석")}</div>
          <div class="ops-meta">
            ${escapeHtml(item.customer_name || "고객사 미지정")} / ${escapeHtml(item.site_name || item.site_code || "전체 현장")}
            · ${escapeHtml(compactTime(item.updated_at))}
          </div>
          <div class="ops-preview">${escapeHtml(item.content_preview || item.question || "미리보기 없음")}</div>
          <div class="ops-actions">
            <button class="secondary" onclick="loadSavedAnalysisById('${escapeHtml(item.id)}')">저장분석 열기</button>
          </div>
        </div>
      `).join("") : '<div class="ops-empty">최근 저장 분석이 없습니다.</div>';
    }

    async function loadOperationsDashboard(silent = false) {
      try {
        const data = await api("/api/nms/operations", {headers: headers()});
        operationsDashboard = data;
        renderOperationsDashboard(data);
        if (!silent) {
          setResult({
            status: "운영 데이터 열람 갱신 완료",
            generated_at: data.generated_at,
            monitor_summary: data.monitor?.summary || {},
            autopilot_summary: data.autopilot?.summary || {},
            saved_analyses: (data.saved_analyses || []).length,
          });
        }
      } catch (err) {
        $("opsDataPill").className = "pill bad";
        $("opsDataPill").textContent = "열람 실패";
        $("opsDataSummary").innerHTML = `<span>오류</span><b style="font-size:14px">${escapeHtml(String(err))}</b>`;
        if (!silent) setResult(`운영 데이터를 불러오지 못했습니다.\n${String(err)}`);
      }
    }

    function toggleOpsAutoRefresh() {
      if (opsRefreshTimer) {
        clearInterval(opsRefreshTimer);
        opsRefreshTimer = null;
        $("opsDataPill").textContent = "자동갱신 중지";
        return;
      }
      loadOperationsDashboard(true);
      opsRefreshTimer = setInterval(() => loadOperationsDashboard(true), 60000);
      $("opsDataPill").textContent = "자동갱신 중";
    }

    async function openAutopilotConversation(conversationId) {
      if (!conversationId) return;
      const data = await api(`/api/conversations/${encodeURIComponent(conversationId)}?limit=80`, {headers: headers()});
      const conversation = data.conversation || {};
      currentConversationId = conversation.id || conversationId;
      localStorage.setItem(conversationKey, currentConversationId);
      const target = conversation.meta?.target || {};
      if (target.customer_name) $("customer").value = target.customer_name;
      if (target.customer_name) {
        const customer = nmsCustomers.find((entry) => entry.customer_name === target.customer_name);
        if (customer) {
          $("nmsCustomer").value = customer.customer_name;
          selectNmsCustomer();
          if (target.site_code) {
            $("nmsSite").value = target.site_code;
            selectNmsSite();
          }
        }
      }
      $("savedAnalysisTitle").value ||= `${opsTargetLabel({target, title: conversation.title})} 자동분석`;
      const messages = (conversation.messages || []).slice(-12).map((message) => {
        const role = message.role === "assistant" ? "LLM 분석" : "요청/근거";
        return `[${role} · ${compactTime(message.created_at)}]\n${message.content || ""}`;
      }).join("\n\n---\n\n");
      setResult(messages || "자동분석 대화 내용이 없습니다.");
      await loadConversations();
    }

    async function loadSavedAnalysisById(analysisId) {
      if (!analysisId) return;
      const select = $("savedAnalysisSelect");
      const exists = Array.from(select.options || []).some((opt) => String(opt.value) === String(analysisId));
      if (exists) {
        select.value = String(analysisId);
        return loadSavedAnalysis();
      }
      const data = await api(`/api/saved-analyses/${encodeURIComponent(analysisId)}`, {headers: headers()});
      const item = data.item || {};
      if (item.customer_name) $("customer").value = item.customer_name;
      if (item.analysis_kind && [...$("template").options].some((opt) => opt.value === item.analysis_kind)) {
        $("template").value = item.analysis_kind;
      }
      $("savedAnalysisTitle").value = item.title || "";
      $("question").value = item.question || $("question").value;
      $("prompt").value = item.source_excerpt || $("prompt").value;
      setResult(item.content || "저장 내용이 없습니다.");
      updateSavedAnalysisStatus(`저장 분석 ${item.id || analysisId}번을 불러왔습니다.`);
    }

    function renderHealthMetrics(health, runningModels = null) {
      const sglang = health.sglang || {};
      const runtime = sglang.runtime_config || {};
      const reachable = Boolean(sglang.reachable || (health.ollama && health.ollama.reachable));
      $("statusPill").className = reachable ? "pill" : "pill bad";
      $("statusPill").textContent = sglang.reachable ? "SGLang 연결 정상" : (reachable ? "LLM 연결 정상" : "LLM 연결 실패");
      const gpu = (health.gpu.gpus || [])[0] || {};
      const runningSource = Array.isArray(runningModels) ? runningModels : (sglang.running || (health.ollama && health.ollama.running) || []);
      const running = runningSource.map(m => m.name).join(", ") || "없음";
      const vramPercent = gpu.memory_used_percent ?? (
        gpu.memory_total_mib ? Math.round(((gpu.memory_used_mib || 0) / gpu.memory_total_mib) * 1000) / 10 : 0
      );
      const targetPercent = runtime.target_vram_percent ?? (
        runtime.mem_fraction_static ? Math.round(runtime.mem_fraction_static * 1000) / 10 : "-"
      );
      $("metrics").innerHTML = `
        <div class="metric"><span>GPU</span><b>${gpu.name || "미확인"}</b></div>
        <div class="metric"><span>VRAM</span><b>${gpu.memory_used_mib || 0} / ${gpu.memory_total_mib || 0} MiB (${vramPercent}%)</b></div>
        <div class="metric"><span>SGLang 목표</span><b>${targetPercent}% · ctx ${runtime.context_length || health.app.default_num_ctx || "-"}</b></div>
        <div class="metric"><span>GPU 사용률</span><b>${gpu.utilization_gpu_percent ?? 0}%</b></div>
        <div class="metric"><span>온도</span><b>${gpu.temperature_c ?? "-"}℃</b></div>
        <div class="metric"><span>실행 모델</span><b style="font-size:14px">${running}</b></div>
      `;
    }

    async function refreshHealthOnly() {
      const health = await api("/api/health");
      renderHealthMetrics(health);
    }

    function rememberSelectedModel() {
      const model = $("model").value;
      if (model) localStorage.setItem("llmOpsSelectedModel", model);
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
        const currentModel = $("model").value || localStorage.getItem("llmOpsSelectedModel") || "";
        const health = await api("/api/health");
        const models = await api("/api/models");
        $("model").innerHTML = "";
        const modelNames = (models.models || []).map((m) => m.name).filter(Boolean);
        const selectedModel = currentModel && modelNames.includes(currentModel) ? currentModel : models.default_model;
        for (const m of models.models || []) {
          const opt = document.createElement("option");
          opt.value = m.name;
          if ((models.precision_models || []).includes(m.name)) {
            opt.textContent = `정밀 분석(Q4 RAM) · ${m.name}`;
          } else if (m.provider === "sglang") {
            opt.textContent = `SGLang · ${m.name}`;
          } else if (m.name === models.default_model) {
            opt.textContent = `운영 기본 · ${m.name}`;
          } else if (m.name === models.fast_model) {
            opt.textContent = `빠른 응답 · ${m.name}`;
          } else {
            opt.textContent = `고급 모델 · ${m.name}`;
          }
          if (m.name === selectedModel) opt.selected = true;
          $("model").appendChild(opt);
        }
        if (selectedModel) localStorage.setItem("llmOpsSelectedModel", selectedModel);
        renderHealthMetrics(health, models.running || []);
      } catch (err) {
        $("statusPill").className = "pill bad";
        $("statusPill").textContent = "상태 확인 실패";
        setResult(String(err));
      }
    }

    async function refreshAll() {
      await refresh();
      await loadConversations();
      await refreshNmsMonitor();
      await loadNmsCustomers();
      await loadFieldTargets(true);
      await loadOperationsDashboard(true);
      await loadSavedAnalyses(true);
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
        const message = String(err);
        const tokenMissing = !$("token").value.trim();
        $("nmsMonitorPill").textContent = /unauthorized/i.test(message)
          ? (tokenMissing ? "118 API 토큰 필요" : "118 API 토큰 확인 필요")
          : "33 NMS 연결 확인 필요";
        $("nmsMonitorSummary").innerHTML = `<span>오류</span><b style="font-size:14px">${escapeHtml(message)}</b>`;
      }
    }

    function normalizeSearchText(value) {
      return String(value || "").toLowerCase().replace(/\s+/g, "");
    }

    function fieldTargetTypeLabel(type) {
      if (type === "netscout_pulse_device") return "NETSCOUT Pulse";
      if (type === "ubuntu_collector") return "Ubuntu Collector";
      return type || "대상";
    }

    function fieldTargetSearchText(target) {
      return normalizeSearchText([
        target.target_type,
        target.target_name,
        target.model_name,
        target.customer_name,
        target.site_name,
        target.hostname,
        target.target_status,
      ].filter(Boolean).join(" "));
    }

    function fieldTargetOptionText(target) {
      const typeLabel = fieldTargetTypeLabel(target.target_type);
      const location = [target.customer_name, target.site_name || target.site_code].filter(Boolean).join(" / ") || "위치 미지정";
      const model = target.model_name || target.model || "-";
      const status = target.target_status || "unknown";
      const age = target.last_observed_age_minutes === null || target.last_observed_age_minutes === undefined
        ? ""
        : ` · ${target.last_observed_age_minutes}분 전`;
      return `${typeLabel} · ${target.target_name || "-"} · ${model} · ${location} · ${status}${age}`;
    }

    function renderFieldTargets() {
      const select = $("fieldTargetSelect");
      if (!select) return;
      const previous = select.value;
      const keyword = normalizeSearchText($("fieldTargetSearch")?.value || "");
      const filtered = keyword ? fieldTargets.filter((target) => fieldTargetSearchText(target).includes(keyword)) : fieldTargets.slice();
      select.innerHTML = '<option value="">현장 테스트 대상 선택</option>';
      for (const target of filtered) {
        const opt = document.createElement("option");
        opt.value = `${target.target_type}:${target.target_id}`;
        opt.textContent = fieldTargetOptionText(target);
        select.appendChild(opt);
      }
      if (previous && filtered.some((target) => `${target.target_type}:${target.target_id}` === previous)) {
        select.value = previous;
      }
      const status = $("fieldTargetStatus");
      if (status) {
        status.textContent = keyword
          ? `${filtered.length}/${fieldTargets.length}개 대상 표시`
          : `${fieldTargets.length}개 대상 표시`;
      }
    }

    function selectedFieldTarget() {
      const value = $("fieldTargetSelect")?.value || "";
      if (!value) return null;
      return fieldTargets.find((target) => `${target.target_type}:${target.target_id}` === value) || null;
    }

    function selectFieldTarget() {
      const target = selectedFieldTarget();
      if (!target) return;
      $("template").value = "field_target_report";
      if (target.customer_name) $("customer").value = target.customer_name;
      $("question").value ||= "선택한 현장 테스트 장비/Collector의 최근 관측값을 보고서로 정리해줘.";
      $("fieldTargetPill").className = target.target_status === "active" ? "pill" : "pill warn";
      $("fieldTargetPill").textContent = `${fieldTargetTypeLabel(target.target_type)} 선택`;
      loadSavedAnalyses(true);
    }

    async function loadFieldTargets(silent = false) {
      try {
        const query = new URLSearchParams({
          q: $("fieldTargetSearch")?.value || "",
          limit: "200",
        });
        const data = await api(`/api/nms/field-targets?${query.toString()}`, {headers: headers()});
        fieldTargets = data.targets || [];
        renderFieldTargets();
        $("fieldTargetPill").className = "pill";
        $("fieldTargetPill").textContent = `${fieldTargets.length}개 대상`;
        if (!silent) {
          setResult({
            status: "현장 테스트 분석 대상 목록 갱신 완료",
            count: fieldTargets.length,
            targets: fieldTargets.slice(0, 12).map((target) => ({
              type: target.target_type,
              name: target.target_name,
              model: target.model_name,
              customer: target.customer_name,
              site: target.site_name,
              status: target.target_status,
              last_observed_at: target.last_observed_at,
              public_ip: target.public_ip,
              private_ip: target.private_ip,
            })),
          });
        }
      } catch (err) {
        $("fieldTargetPill").className = "pill bad";
        $("fieldTargetPill").textContent = "대상 실패";
        if (!silent) setResult(`현장 테스트 분석 대상 목록을 불러오지 못했습니다.\n${String(err)}`);
      }
    }

    async function loadFieldEvidence() {
      const target = selectedFieldTarget();
      if (!target) {
        setResult("현장 테스트 분석 대상을 먼저 선택하세요.");
        return;
      }
      try {
        const query = new URLSearchParams({
          target_type: target.target_type,
          target_id: target.target_id,
          hours: "24",
          limit: "80",
        });
        const data = await api(`/api/nms/field-evidence?${query.toString()}`, {headers: headers()});
        $("template").value = "field_target_report";
        $("customer").value = data.matched?.customer_name || target.customer_name || $("customer").value;
        $("prompt").value = JSON.stringify(data, null, 2);
        setResult({
          target: data.target,
          data_coverage: data.data_coverage,
          gaps: data.deterministic_analysis?.gaps || [],
          top_signals: data.deterministic_analysis?.top_signals || [],
        });
      } catch (err) {
        setResult(String(err));
      }
    }

    async function analyzeFieldTarget() {
      const target = selectedFieldTarget();
      if (!target) {
        setResult("현장 테스트 분석 대상을 먼저 선택하세요.");
        return;
      }
      try {
        setResult(`현장 테스트 대상 분석 중... ${fieldTargetTypeLabel(target.target_type)} / ${target.target_name || target.target_id}`);
        const attachments = await collectAttachmentPayloads();
        const data = await api("/api/nms/field-analyze", {
          method: "POST",
          headers: headers(),
          body: JSON.stringify(conversationPayload({
            model: $("model").value,
            target_type: target.target_type,
            target_id: target.target_id,
            question: $("question").value,
            hours: 24,
            limit: 80,
            depth: "field",
            attachments,
            source: "nms-field-analysis",
          })),
        });
        applyConversationFromResponse(data);
        setResult(renderResponseWithAttachmentWarnings(data));
        $("savedAnalysisTitle").value ||= `${target.target_name || "현장 테스트 대상"} 분석 보고서`;
        refresh();
      } catch (err) {
        setResult(String(err));
      }
    }

    function nmsCustomerSearchText(customer) {
      const siteText = (customer.sites || [])
        .map((site) => `${site.site_name || ""} ${site.site_code || ""}`)
        .join(" ");
      return normalizeSearchText(`${customer.customer_name || ""} ${customer.customer_code || ""} ${siteText}`);
    }

    function nmsMenuState() {
      return {
        customerSearch: $("nmsCustomerSearch")?.value || "",
        customerName: $("nmsCustomer")?.value || "",
        siteCode: $("nmsSite")?.value || "",
        template: $("template")?.value || "",
        question: $("question")?.value || "",
        scrollY: Math.max(0, window.scrollY || 0),
      };
    }

    function sameNmsMenuState(left, right) {
      if (!left || !right) return false;
      return left.customerSearch === right.customerSearch
        && left.customerName === right.customerName
        && left.siteCode === right.siteCode
        && left.template === right.template
        && left.question === right.question;
    }

    function updateNmsBackButton() {
      const button = $("nmsBackButton");
      if (!button) return;
      button.disabled = nmsNavigationStack.length === 0;
      button.title = nmsNavigationStack.length ? "직전 업체/현장 메뉴로 돌아갑니다." : "이전 메뉴가 없습니다.";
    }

    function pushNmsMenuState(state) {
      if (!state) return;
      const last = nmsNavigationStack[nmsNavigationStack.length - 1];
      if (sameNmsMenuState(last, state)) return;
      nmsNavigationStack.push({...state});
      if (nmsNavigationStack.length > 25) {
        nmsNavigationStack.shift();
      }
      updateNmsBackButton();
    }

    function trackNmsMenuChange() {
      if (isRestoringNmsMenu) return;
      const next = nmsMenuState();
      if (currentNmsMenuState && !sameNmsMenuState(currentNmsMenuState, next)) {
        pushNmsMenuState(currentNmsMenuState);
      }
    }

    function settleNmsMenuState() {
      currentNmsMenuState = nmsMenuState();
      updateNmsBackButton();
    }

    function renderNmsCustomers(options = {}) {
      const select = $("nmsCustomer");
      const previousValue = options.selectedCustomerName ?? select.value;
      const keyword = normalizeSearchText($("nmsCustomerSearch")?.value || "");
      const filtered = keyword
        ? nmsCustomers.filter((customer) => nmsCustomerSearchText(customer).includes(keyword))
        : nmsCustomers.slice();
      select.innerHTML = '<option value="">업체 선택</option>';
      for (const customer of filtered) {
        const opt = document.createElement("option");
        opt.value = customer.customer_name;
        opt.textContent = `${customer.customer_name} (${customer.site_count || 0}현장 / ${customer.recent_event_count || 0}이벤트)`;
        opt.dataset.customerCode = customer.customer_code || "";
        select.appendChild(opt);
      }
      if (previousValue && filtered.some((customer) => customer.customer_name === previousValue)) {
        select.value = previousValue;
      }
      const status = $("nmsCustomerSearchStatus");
      if (status) {
        status.textContent = keyword
          ? `${filtered.length}/${nmsCustomers.length}개 업체 표시`
          : `${nmsCustomers.length}개 업체 표시`;
      }
      selectNmsCustomer({trackHistory: false, preserveSiteCode: options.preserveSiteCode || ""});
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

    function selectNmsCustomer(options = {}) {
      if (options.trackHistory !== false) {
        trackNmsMenuChange();
      }
      const customer = selectedCustomer();
      const siteSelect = $("nmsSite");
      const preservedSiteCode = options.preserveSiteCode || "";
      siteSelect.innerHTML = '<option value="">전체 현장</option>';
      if (!customer) {
        settleNmsMenuState();
        loadSavedAnalyses(true);
        return;
      }
      $("customer").value = customer.customer_name;
      if ($("template").value === "freeform") {
        $("template").value = "nms_deep";
      }
      for (const site of customer.sites || []) {
        const opt = document.createElement("option");
        opt.value = site.site_code;
        opt.textContent = `${site.site_name} / ${site.site_code} (${site.recent_event_count || 0})`;
        siteSelect.appendChild(opt);
      }
      if (preservedSiteCode && (customer.sites || []).some((site) => site.site_code === preservedSiteCode)) {
        siteSelect.value = preservedSiteCode;
      }
      settleNmsMenuState();
      loadSavedAnalyses(true);
    }

    function selectNmsSite(options = {}) {
      if (options.trackHistory !== false) {
        trackNmsMenuChange();
      }
      const customer = selectedCustomer();
      if (customer) $("customer").value = customer.customer_name;
      settleNmsMenuState();
      loadSavedAnalyses(true);
    }

    function goBackNmsMenu() {
      const previous = nmsNavigationStack.pop();
      updateNmsBackButton();
      if (!previous) {
        setResult("이전 업체/현장 메뉴가 없습니다.");
        return;
      }
      isRestoringNmsMenu = true;
      try {
        $("nmsCustomerSearch").value = previous.customerSearch || "";
        renderNmsCustomers({
          selectedCustomerName: previous.customerName || "",
          preserveSiteCode: previous.siteCode || "",
        });
        if (previous.template && [...$("template").options].some((opt) => opt.value === previous.template)) {
          $("template").value = previous.template;
        }
        $("question").value = previous.question || "";
        if (previous.siteCode && $("nmsSite").value !== previous.siteCode) {
          $("nmsSite").value = previous.siteCode;
        }
        selectNmsSite({trackHistory: false});
        if (Number.isFinite(previous.scrollY)) {
          window.scrollTo({top: previous.scrollY, behavior: "auto"});
        }
        settleNmsMenuState();
      } finally {
        isRestoringNmsMenu = false;
      }
    }

    async function loadNmsContext() {
      const customer = selectedCustomer();
      const siteCode = $("nmsSite").value;
      if (!customer && !siteCode) {
        setResult("업체 또는 현장을 먼저 선택하세요.");
        return;
      }
      $("template").value = "nms_context";
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
      try {
        setResult(`33 NMS 데이터 분석 중... 선택 모델(${ $("model").value || "운영 기본 모델" })로 심층 분석합니다.`);
        const attachments = await collectAttachmentPayloads();
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
            attachments,
            source: "nms-analysis",
          })),
        });
        applyConversationFromResponse(data);
        setResult(renderResponseWithAttachmentWarnings(data));
        $("savedAnalysisTitle").value ||= $("question").value.trim() || `${$("customer").value.trim() || "고객"} NMS 분석`;
        refresh();
      } catch (err) {
        setResult(String(err));
      }
    }

    function toggleNmsMonitor() {
      if (nmsMonitorTimer) {
        clearInterval(nmsMonitorTimer);
        nmsMonitorTimer = null;
        $("nmsMonitorPill").textContent = "상태카드 자동갱신 중지";
        return;
      }
      refreshNmsMonitor();
      nmsMonitorTimer = setInterval(refreshNmsMonitor, 60000);
      $("nmsMonitorPill").textContent = "상태카드 자동갱신";
    }

    function backendTemplate(mode) {
      if (mode === "maintenance_report") return "maintenance_report";
      if (mode === "customer_history") return "customer_history";
      return "freeform";
    }

    async function runSelectedAnalysis() {
      const mode = $("template").value;
      if (mode === "nms_deep") return analyzeNms();
      if (mode === "field_target_report") return analyzeFieldTarget();
      if (mode === "nms_context") return loadNmsContext();
      if (mode === "codex_task") return buildCodexTaskDraft();
      return analyze();
    }

    function currentResultText() {
      const value = $("result").textContent.trim();
      if (!value || value === "대기 중입니다." || value.startsWith("생성 중") || value.startsWith("분석 중")) {
        return "";
      }
      return value;
    }

    function buildCodexTaskDraft() {
      const scope = analysisScope();
      const customer = scope.customer_name || "고객사 미지정";
      const site = scope.site_name || scope.site_code || "전체 현장";
      const question = $("question").value.trim() || "현재 선택된 고객/현장 데이터를 기준으로 원인 분석과 개선 작업을 진행해줘.";
      const result = currentResultText();
      const prompt = $("prompt").value.trim();
      const draft = [
        "Codex 작업 지시 초안",
        "",
        "목표:",
        `- ${question}`,
        "",
        "대상:",
        `- 고객사: ${customer}`,
        `- 현장: ${site}`,
        `- 모델: ${$("model").value || "운영 기본 모델"}`,
        "",
        "참고 자료:",
        result ? result.slice(0, 5000) : "- 현재 저장된 분석 결과 없음",
        prompt ? `\n추가 메모/로그:\n${prompt.slice(0, 3000)}` : "",
        "",
        "Codex에게 요청할 작업:",
        "- 제공된 근거를 먼저 요약한다.",
        "- 확정 사실과 추정 판단을 분리한다.",
        "- 필요한 코드/설정/운영 변경이 있으면 파일, 서비스, 검증 명령까지 제시한다.",
        "- 실행 전 위험 요소와 되돌리는 방법을 같이 남긴다.",
      ].filter(Boolean).join("\n");
      $("savedAnalysisTitle").value ||= `${customer} Codex 작업 지시`;
      setResult(draft);
      return draft;
    }

    async function chat() {
      try {
        setResult(`생성 중... 선택 모델(${ $("model").value || "운영 기본 모델" })로 실행합니다.`);
        const attachments = await collectAttachmentPayloads();
        const data = await api("/api/chat", {
          method: "POST",
          headers: headers(),
          body: JSON.stringify(conversationPayload({
            model: $("model").value,
            prompt: $("prompt").value,
            attachments,
            temperature: 0.2,
            source: "free-chat",
          })),
        });
        applyConversationFromResponse(data);
        setResult(renderResponseWithAttachmentWarnings(data));
        $("savedAnalysisTitle").value ||= $("question").value.trim() || `${$("customer").value.trim() || "고객"} 분석`;
        refresh();
      } catch (err) {
        setResult(String(err));
      }
    }

    async function continueConversation() {
      try {
        const question = $("question").value.trim();
        const extra = $("prompt").value.trim();
        if (!question && !extra) {
          setResult("이어 묻기 내용을 '요청 / 후속 질문' 또는 '추가 자료'에 입력하세요.");
          return;
        }
        setResult("이전 분석 맥락을 이어서 답변 생성 중...");
        const attachments = await collectAttachmentPayloads();
        const prompt = [
          "이전 분석과 대화 이력을 이어서 답해라.",
          question ? `후속 질문:\n${question}` : "",
          extra ? `추가 자료:\n${extra}` : "",
        ].filter(Boolean).join("\n\n");
        const data = await api("/api/chat", {
          method: "POST",
          headers: headers(),
          body: JSON.stringify(conversationPayload({
            model: $("model").value,
            prompt,
            attachments,
            temperature: 0.16,
            source: "follow-up",
          })),
        });
        applyConversationFromResponse(data);
        setResult(renderResponseWithAttachmentWarnings(data));
        $("savedAnalysisTitle").value ||= $("question").value.trim() || `${$("customer").value.trim() || "고객"} 후속 질의`;
        refresh();
      } catch (err) {
        setResult(String(err));
      }
    }

    async function analyze() {
      try {
        setResult(`분석 중... 선택 모델(${ $("model").value || "운영 기본 모델" })로 실행합니다.`);
        const attachments = await collectAttachmentPayloads();
        const data = await api("/api/analyze", {
          method: "POST",
          headers: headers(),
          body: JSON.stringify(conversationPayload({
            model: $("model").value,
            template: backendTemplate($("template").value),
            customer: $("customer").value,
            question: $("question").value,
            data: $("prompt").value,
            attachments,
            temperature: 0.15,
            source: "template-analysis",
          })),
        });
        applyConversationFromResponse(data);
        setResult(renderResponseWithAttachmentWarnings(data));
        $("savedAnalysisTitle").value ||= $("question").value.trim() || `${$("customer").value.trim() || "고객"} 분석`;
        refresh();
      } catch (err) {
        setResult(String(err));
      }
    }

    async function analyzeLargeLog() {
      try {
        const files = attachmentFiles();
        if (!files.length) {
          setResult("대용량 로그 분석에는 TXT/LOG/CSV 등 로그 파일 첨부가 필요합니다.");
          return;
        }
        setResult(`대용량 로그를 저장하고 분할 분석 중... 선택 모델(${ $("model").value || "운영 기본 모델" })로 실행합니다.`);
        const attachments = await collectAttachmentPayloads("large-log");
        const data = await api("/api/large-log/analyze", {
          method: "POST",
          headers: headers(),
          body: JSON.stringify(conversationPayload({
            model: $("model").value,
            customer: $("customer").value,
            site_code: $("nmsSite").value,
            question: $("question").value,
            prompt: $("prompt").value,
            attachments,
            source: "large-log-analysis",
          })),
        });
        applyConversationFromResponse(data);
        const stored = data.stored || {};
        const scan = data.scan || {};
        const header = [
          "[대용량 로그 처리]",
          `- 저장 위치: ${stored.directory || "-"}`,
          `- 파일: ${(stored.items || []).map((item) => `${item.name}(${formatBytes(item.size || 0)})`).join(", ") || "-"}`,
          `- 전체 조각: ${data.chunk_count_total || 0} / 분석 조각: ${data.chunk_count_analyzed || 0}`,
          `- 라인수: ${scan.line_count || 0} / 문자수: ${scan.chars || 0}`,
        ].join("\n");
        setResult(`${header}\n\n${renderResponseWithAttachmentWarnings(data)}`);
        $("savedAnalysisTitle").value ||= $("question").value.trim() || `${$("customer").value.trim() || "고객"} 대용량 로그 분석`;
        refresh();
      } catch (err) {
        setResult(String(err));
      }
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
    loadOperationsDashboard(true);
    loadSavedAnalyses(true);
    renderAttachmentSummary();
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
