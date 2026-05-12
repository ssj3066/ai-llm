# 118 LLM Ops 첨부파일 분석 기능 추가 메모

## 목적

- 분석 요청 시 외부 파일 `CSV`, `TXT`, `XLS`, `XLSX`, `DOC`, `DOCX`를 함께 보내고
- 추출된 텍스트를 LLM 분석 문맥에 추가

## 구현 내용

- `app.py`
  - 첨부 base64 payload 처리 추가
  - TXT/CSV 직접 파싱
  - DOCX/XLSX 내부 XML 직접 파싱
  - DOC/XLS는 LibreOffice 변환 우선, 없으면 `strings` 보조 추출
  - `POST /api/chat`, `POST /api/analyze`, `POST /api/nms/analyze` 첨부 지원
  - 응답에 `attachment_summary`, `attachment_errors` 포함
- 웹 콘솔
  - 파일 선택 입력 추가
  - 선택 파일 요약 표시
  - 첨부 비우기 버튼 추가
  - 분석 결과에 첨부 처리 경고 표시
  - 일반 채팅 버튼도 동일 첨부 payload 사용
  - `XLS`는 LibreOffice로 `CSV` 변환 후 읽도록 보정

## 검증

- 로컬
  - `python3 -m py_compile /home/metro/work/llm-ops-118/app.py`
  - 내장 JS `node --check` 통과
  - 샘플 `TXT`, `CSV`, `DOCX`, `XLSX` 추출 확인
- 118 실서버
  - `metro-llm-ops.service` 재시작 후 `active`
  - 첨부 4건 업로드 분석 검증
  - `attachment_count=4`, `attachment_errors=[]`
  - 결과 응답에 `돈우`, `ping`, `코어스위치`, `온도 33` 반영 확인

## 운영 메모

- 2026-05-12 후속 작업으로 118 실서버에 `libreoffice-common`, `libreoffice-core`, `libreoffice-writer`, `libreoffice-calc` 설치 완료
- 설치 후 `DOC`는 텍스트 변환, `XLS`는 `CSV` 변환 기준으로 추출 검증 완료
- 실측 추출 예시
  - DOC: `돈우 레거시 DOC 테스트 / 코어스위치 정상`
  - XLS: `항목 | 값 / ping | 정상 / 온도 | 33`

## 실서버 백업

- `~/llm-ops/app.py.backup-attachment-analysis-20260512-113843`
- `~/llm-ops/app.py.backup-chat-attachments-20260512-120511`
- `~/llm-ops/app.py.backup-xls-csv-20260512-120717`
