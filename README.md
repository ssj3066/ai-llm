# Metro LLM Ops for 118

118번 GPU 서버에서 Ollama 기반 로컬 LLM을 운영하기 위한 단일 Python 서비스입니다.

## 기능

- 웹 콘솔: `http://192.168.1.118:8090/`
- 상태 API: `GET /api/health`
- 모델 목록: `GET /api/models`
- 채팅 실행: `POST /api/chat`
- NMS/유지보수/고객이력 분석 템플릿: `POST /api/analyze`
- 저장 분석 목록/불러오기/삭제: `GET|POST|DELETE /api/saved-analyses`
- 33번 NMS 업체/현장 목록: `GET /api/nms/customers`
- 33번 NMS 최근 컨텍스트: `GET /api/nms/context`
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

## API 호출 예시

```bash
TOKEN="$(grep '^LLM_OPS_TOKEN=' /home/metroai/llm-ops/llm-ops.env | cut -d= -f2-)"
curl -sS -H "X-LLM-Ops-Token: $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"model":"metro-report:latest","prompt":"돈우 NMS 로그 분석 기준을 한 문장으로 설명해줘."}' \
  http://192.168.1.118:8090/api/chat
```

## 운영 기준

- ERP/33번 NMS에서 직접 LLM을 돌리지 않고, 118번을 전용 추론 서버로 호출합니다.
- 33번 NMS는 관제 데이터 원본이고, 118번은 해당 데이터를 주기적으로 읽어서 분석/표시합니다.
- NMS Autopilot은 `NMS_AUTOPILOT_INTERVAL_SECONDS`마다 고객/현장 상태를 점검하고, 이벤트/장애 점수가 가장 높은 대상을 골라 분석을 생성해 대화 이력에 저장합니다.
- 같은 현장 반복 분석은 `NMS_AUTOPILOT_TARGET_COOLDOWN_SECONDS`로 제한합니다. 기본값은 30분입니다.
- 수동 `POST /api/nms/analyze`는 긴 심층분석, Autopilot은 짧고 반복적인 상시 분석입니다. 자동분석의 출력 길이는 `NMS_AUTOPILOT_NUM_PREDICT`로 제한합니다.
- 웹 콘솔의 `업체/현장 분석 보관함`은 고객사/현장별 분석 결과를 별도 저장하고, 같은 범위로 다시 분석할 때 저장 이력을 자동 참고 문맥으로 넣습니다.
- 저장 분석은 `conversations.sqlite3` 내부 `saved_analyses` 테이블에 들어가며, 현재 UI에서는 저장/불러오기/삭제만 제공하고 수정은 새 저장으로 남깁니다.
- 일반 채팅은 별도 시스템 지시가 없으면 기본 응답 언어를 한국어로 유지합니다. 다른 언어를 명시적으로 요청하면 그 요청을 따릅니다.
- 일반 채팅, 분석, NMS 분석 시 `CSV`, `TXT`, `XLSX`, `DOCX` 첨부를 함께 보낼 수 있고, 서버에서 텍스트를 추출해 프롬프트에 같이 넣습니다.
- `XLS`, `DOC`는 LibreOffice가 있으면 텍스트 변환 후 분석하고, 없으면 `strings` 기반 보조 추출로 처리합니다.
- 기본 첨부 제한은 `최대 5개`, `파일당 5MB`, `전체 12MB`입니다.
- 웹 콘솔 상단 GPU/VRAM/실행모델 카드는 10초마다 자동 갱신됩니다.
- 기본 모델은 보고서/분석용 `metro-report:latest`입니다.
- 빠른 단문/상태 점검은 `metro-fast:latest`를 사용합니다.
- API 토큰은 `llm-ops.env`에 저장하고, 코드나 문서에 직접 노출하지 않습니다.
