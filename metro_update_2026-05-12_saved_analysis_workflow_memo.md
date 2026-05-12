# 118 LLM Ops 저장 분석 보관함 추가 메모

## 작업 개요

- 대상: `192.168.1.118` `~/llm-ops/app.py`
- 목적: 업체/현장별 분석 결과를 저장하고, 이후 같은 범위 분석 시 이전 결과를 참고 문맥으로 자동 주입

## 반영 내용

- SQLite `conversations.sqlite3`에 `saved_analyses` 테이블 추가
- API 추가
  - `GET /api/saved-analyses`
  - `GET /api/saved-analyses/{id}`
  - `POST /api/saved-analyses`
  - `DELETE /api/saved-analyses/{id}`
- `POST /api/analyze` 자동 참고 문맥 주입
- `POST /api/nms/analyze` 자동 참고 문맥 주입
- NMS Autopilot 분석에도 저장 이력 참고 적용
- 118 웹 콘솔에 `업체/현장 분석 보관함` UI 추가
  - 저장 제목
  - 목록 새로고침
  - 현재 결과 저장
  - 선택 불러오기
  - 선택 삭제

## 검증

- 로컬 문법 검증
  - `python3 -m py_compile /home/metro/work/llm-ops-118/app.py`
  - HTML 내장 스크립트 `node --check` 통과
- 118 실서버 반영 후 검증
  - `systemctl is-active metro-llm-ops.service` => `active`
  - 저장/목록/단건/삭제 API 동작 확인
  - 저장 분석의 고유문구가 새 분석 응답에 반영되는지 확인

## 비고

- 실서버 백업 파일: `~/llm-ops/app.py.backup-saved-analyses-20260512-111455`
- 이번 검증용 `테스트업체-118 / TEST-118-A` 저장 데이터는 마지막에 삭제함
