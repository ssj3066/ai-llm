# 118 LLM Ops 현장 테스트 장비 분석 작업 메모

## 작업 개요

- 대상: `192.168.1.118` `~/llm-ops/app.py`
- 목적: NETSCOUT Pulse 2000과 향후 Ubuntu collector를 118번 LLM 서버에서 독립 분석 대상으로 선택하고 보고서를 생성

## 반영 내용

- 33번 NMS 연동 API 프록시 추가
  - `GET /api/nms/field-targets`
  - `GET /api/nms/field-evidence`
  - `POST /api/nms/field-analyze`
- 웹 콘솔 UI 추가
  - 분석 유형 `현장 테스트 장비 분석`
  - `현장 테스트 분석 대상` 카드
  - 대상 검색/선택
  - `대상 새로고침`
  - `선택 근거 보기`
  - `선택 대상 보고서`
- LLM 입력 압축 로직 추가
  - `field-analysis-evidence` evidence pack을 compact 처리
  - public IP와 private IP를 분리해서 전달
  - PoE/포트/VLAN/LLDP/ARP/diagnostic 값은 수집되지 않으면 생성하지 않도록 프롬프트 제한
- Evidence Store 저장
  - 현장 분석 evidence를 `source_type=field-analysis-evidence`로 저장
  - 보고서 생성 시 snapshot id를 응답에 포함

## 운영 반영 및 검증

- 118번 운영 백업:
  - `/home/metroai/.codex-deploy-backups/llm-field-analysis-20260626-051934`
  - `/home/metroai/.codex-deploy-backups/llm-field-analysis-meta-20260626-052400`
- 검증:
  - `python3 -m py_compile app.py`
  - `metro-llm-ops.service` active 확인
  - `/api/health` 정상
  - `/api/nms/field-targets` 대상 1건 확인
  - `/api/nms/field-evidence` `field-analysis-evidence-v1` 확인
  - `/api/nms/field-analyze` 실제 보고서 생성 확인

## 실제 보고서 검증값

- 대상: `NETSCOUT Pulse 2000 장애 현장 테스트 장치`
- 대상 ID: `40`
- 고객/현장: `메트로정보통신 / 장애 현장 테스트`
- Evidence snapshot id: `131`
- 모델 프로파일: deep / `metro-analysis:sglang`
- 생성 시간: 약 69초
- 보고서에 포함된 핵심 항목:
  - 공인 IP와 내부 IP 분리
  - Pulse observation/metric 커버리지
  - site collector 부재로 인한 ARP/LLDP/포트/VLAN 근거 부족
  - 다음 현장 확인 항목

## 운영 기준

- 118번 LLM은 Grafana 화면이 아니라 33번 NMS API evidence pack을 근거로 분석
- 대상 선택 목록에는 혼선을 줄이기 위해 등록 device IP를 노출하지 않음
- 주소가 필요할 때는 `public_ip`와 `private_ip` 의미를 분리해 표시
