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
  - `POST /api/analyze`, `POST /api/nms/analyze` 둘 다 첨부 지원
  - 응답에 `attachment_summary`, `attachment_errors` 포함
- 웹 콘솔
  - 파일 선택 입력 추가
  - 선택 파일 요약 표시
  - 첨부 비우기 버튼 추가
  - 분석 결과에 첨부 처리 경고 표시

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

- 118 실서버에는 현재 `libreoffice`가 없음
- 따라서 `DOC`, `XLS`는 현재 `strings` 기반 보조 추출로 동작
- 레거시 `DOC/XLS` 품질이 중요하면 118에 LibreOffice 설치 후 재검증 필요

## 실서버 백업

- `~/llm-ops/app.py.backup-attachment-analysis-20260512-113843`
