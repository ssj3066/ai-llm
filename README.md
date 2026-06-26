# Metro LLM Ops for 118

118번 GPU 서버에서 SGLang 기반 로컬 LLM을 운영하기 위한 단일 Python 서비스입니다.

## 기능

- 웹 콘솔: `http://192.168.1.118:8090/`
- 상태 API: `GET /api/health`
- 모델 목록: `GET /api/models`
- 채팅 실행: `POST /api/chat`
- NMS/유지보수/고객이력 분석 템플릿: `POST /api/analyze`
- 저장 분석 목록/불러오기/삭제: `GET|POST|DELETE /api/saved-analyses`
- 33번 NMS 업체/현장 목록: `GET /api/nms/customers`
- 33번 NMS 최근 컨텍스트: `GET /api/nms/context`
- 33번 NMS 현장 테스트 분석 대상: `GET /api/nms/field-targets`
- 33번 NMS 현장 테스트 대상 evidence: `GET /api/nms/field-evidence`
- NETSCOUT Pulse/Ubuntu Collector 독립 분석: `POST /api/nms/field-analyze`
- 33번 NMS 상시 모니터링 상태: `GET /api/nms/monitor/status`
- 33번 NMS 선택 고객 분석: `POST /api/nms/analyze`
- 33번 NMS 자동분석 워커 상태: `GET /api/nms/autopilot/status`
- 33번 NMS 자동분석 즉시 실행: `POST /api/nms/autopilot/run`
- 모델 프리로드: `POST /api/preload`
- 모델 언로드: `POST /api/unload`
- GPU 상태: `nvidia-smi` 기반 표시

## 배포 위치

```bash
/home/metroai/llm-ops
```

## 소스 기준

```bash
/home/metro/work/llm-ops-118
```

- 이 폴더를 118번 LLM Ops의 로컬 소스 기준 경로로 사용합니다.
- 실서버에서 직접 긴급 수정한 경우에는 이 경로로 다시 동기화한 뒤 후속 작업을 진행합니다.

## systemd

```bash
sudo systemctl status metro-llm-ops.service
sudo systemctl restart metro-llm-ops.service
sudo journalctl -u metro-llm-ops.service -f
```

## SGLang 운영 기본값

- 기본 추론 경로는 SGLang입니다. `OLLAMA_ENABLED=false`, `SGLANG_ENABLED=true`를 기준으로 운영합니다.
- 고객/업무 화면에는 `LLM_OPS_PUBLIC_MODELS`에 정의한 3개 프로파일만 노출합니다: `metro-fast:sglang`, `metro-analysis:sglang`, `metro-report:sglang`.
- 3개 프로파일은 현재 운영 체크포인트 `Qwen/Qwen3-4B-Instruct-2507`로 연결합니다. 역할 구분은 고객 응대/상태 점검/보고서 워크플로우 기준으로 유지하고, 물리 모델 교체는 alias 매핑으로 처리합니다.
- SGLang 런타임은 `SGLANG_KV_CACHE_DTYPE=fp8_e5m2`와 `SGLANG_ENABLE_CACHE_REPORT=true`를 사용합니다.
- `SGLANG_QUANTIZATION`은 AWQ/GPTQ/FP8 등 양자화 체크포인트를 별도로 배치했을 때만 설정합니다. 일반 BF16/FP16 체크포인트에 임의로 켜지 않습니다.

## API 호출 예시

```bash
TOKEN="$(grep '^LLM_OPS_TOKEN=' /home/metroai/llm-ops/llm-ops.env | cut -d= -f2-)"
curl -sS -H "X-LLM-Ops-Token: $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"model":"metro-analysis:sglang","prompt":"돈우 NMS 로그 분석 기준을 한 문장으로 설명해줘."}' \
  http://192.168.1.118:8090/api/chat
```

## 운영 기준

- ERP/33번 NMS에서 직접 LLM을 돌리지 않고, 118번을 전용 추론 서버로 호출합니다.
- 33번 NMS는 관제 데이터 원본이고, 118번은 해당 데이터를 주기적으로 읽어서 분석/표시합니다.
- NMS Autopilot은 `NMS_AUTOPILOT_INTERVAL_SECONDS`마다 고객/현장 상태를 점검하고, 이벤트/장애 점수가 가장 높은 대상을 골라 분석을 생성해 대화 이력에 저장합니다.
- 같은 현장 반복 분석은 `NMS_AUTOPILOT_TARGET_COOLDOWN_SECONDS`로 제한합니다. 기본값은 30분입니다.
- 수동 `POST /api/nms/analyze`는 긴 심층분석, Autopilot은 짧고 반복적인 상시 분석입니다. 자동분석의 출력 길이는 `NMS_AUTOPILOT_NUM_PREDICT`로 제한합니다.
- 33번 NMS가 제공하는 `temporal.nas_ransomware_findings`는 규칙 기반 1차 경보로 취급하고, LLM 분석에서 severity/title/count/sample message를 우선 근거로 인용합니다.
- 33번 NMS가 `network-evidence-pack` API를 제공하면 NMS 심층분석은 기존 `nms-context` 대신 이 고객사별 evidence pack을 우선 사용합니다. 33번은 기간 전체 원천 데이터를 `compressed_evidence`로 먼저 집계하고, 118번은 이 집계와 대표 샘플을 LLM 입력으로 압축해 사용합니다.
- 휴대용 NETSCOUT Pulse와 현장 Ubuntu Collector는 `현장 테스트 분석 대상`에서 별도 선택합니다. 118번은 `field-analysis-evidence`를 받아 공인 접속 IP, 장비/collector 내부 보고 IP, PoE/포트/VLAN, syslog/trap, diagnostic 결과, 누락 데이터를 분리해 보고서를 생성합니다.
- 향후 고객사에 설치할 Ubuntu Collector는 33번 NMS의 `collectors`에 등록하고 heartbeat가 들어오면 같은 목록에 표시됩니다. Collector 분석에서는 heartbeat, diagnostic command, syslog/trap, polling sample을 evidence로 사용합니다.
- 과도한 원문 프롬프트가 들어오면 Ollama가 내부에서 임의 절단하기 전에 118번이 `LLM_OPS_MAX_PROMPT_CHARS`, `LLM_OPS_MAX_SINGLE_MESSAGE_CHARS`, `NMS_LLM_EVIDENCE_CHAR_LIMIT` 기준으로 명시적으로 압축하고 경고를 반환합니다.
- 웹 콘솔의 `업체/현장 분석 보관함`은 고객사/현장별 분석 결과를 별도 저장하고, 같은 범위로 다시 분석할 때 저장 이력을 자동 참고 문맥으로 넣습니다.
- 저장 분석은 `conversations.sqlite3` 내부 `saved_analyses` 테이블에 들어가며, 현재 UI에서는 저장/불러오기/삭제만 제공하고 수정은 새 저장으로 남깁니다.
- 일반 채팅은 별도 시스템 지시가 없으면 기본 응답 언어를 한국어로 유지합니다. 다른 언어를 명시적으로 요청하면 그 요청을 따릅니다.
- 일반 채팅, 분석, NMS 분석 시 `CSV`, `TXT`, `XLSX`, `DOCX` 첨부를 함께 보낼 수 있고, 서버에서 텍스트를 추출해 프롬프트에 같이 넣습니다.
- `XLS`, `DOC`는 LibreOffice가 있으면 텍스트 변환 후 분석하고, 없으면 `strings` 기반 보조 추출로 처리합니다.
- 기본 첨부 제한은 `최대 5개`, `파일당 5MB`, `전체 12MB`입니다.
- 웹 콘솔 상단 GPU/VRAM/실행모델 카드는 10초마다 자동 갱신됩니다.
- 기본 모델은 고객/현장 분석용 `metro-analysis:sglang`입니다.
- 빠른 단문/상태 점검은 `metro-fast:sglang`, 보고서형 응답은 `metro-report:sglang`를 사용합니다.
- 웹 콘솔에서 선택한 모델은 채팅, 일반 분석, NMS 분석, 프리로드, 언로드, 벤치마크 요청에 그대로 전달됩니다.
- 설치되지 않은 모델이나 timeout 실패는 `LLM_OPS_MODEL_FALLBACK_ENABLED=true`일 때만 기본 모델로 재시도합니다. SGLang 단일 운영에서는 기본값을 `false`로 두어 의도하지 않은 모델 전환을 막습니다.
- GPU 실행 모델 전환을 강제로 막아야 하는 운영 상황에서는 `LLM_OPS_MODEL_RUNNING_SWITCH_GUARD_ENABLED=true`를 설정합니다. 기본값은 `false`입니다.
- SGLang 단일 운영에서는 `LLM_OPS_MODEL_AUTO_UNLOAD_BEFORE_SWITCH_ENABLED=false`를 기본으로 둡니다. 모델 전환은 SGLang 체크포인트 교체 작업으로 별도 처리합니다.
- 웹 콘솔은 기본 업무 버튼을 우선 표시하고, 프리로드/언로드/벤치마크는 `고급 모델 작업` 영역에 접어 둡니다.
- 표준 사용 흐름은 `모델 선택 -> 업체/현장 선택 -> 분석 기능 선택 -> 분석 실행 -> 이어 묻기 -> 결과 저장`입니다.
- `상태카드 자동갱신`은 답변 생성이 아니라 NMS 상태 카드만 60초마다 새로 읽는 선택 기능입니다.
- `Codex 작업 지시 초안`은 현재 분석 결과와 고객/현장 범위를 Codex에게 넘길 작업 지시 형태로 정리합니다. 실제 Codex 실행 연동은 별도 bridge가 필요합니다.
- 한국어 요청에서 모델이 중국어/영어로 시작하면 서버가 1회 한국어 재작성 요청을 자동 수행합니다.
- 재작성 후에도 비한국어 안내문이 앞에 붙으면 화면에는 한국어 답변 부분만 표시합니다.
- API 토큰은 `llm-ops.env`에 저장하고, 코드나 문서에 직접 노출하지 않습니다.
