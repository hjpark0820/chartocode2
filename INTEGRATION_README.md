# Unified Chart Digitizer — 흑백 + 컬러 통합

하나의 웹 GUI에서 **컬러/흑백 모드를 선택**해 각 파이프라인을 돌립니다.
기본 GUI는 컬러 프로젝트의 세련된 편집기를 그대로 사용하고, 흑백 출력은
어댑터로 같은 스키마로 변환해 **편집기·실시간 재구성·CSV를 공유**합니다.

## 파일 (chartocode2/src/ 에 배치)
- `unified_server.py`   — 통합 FastAPI (mode=color|bw 라우팅)
- `bw_to_edit_data.py`  — 흑백 run_detection() 출력 → 공통 edit_data.json 어댑터
- `bw_pipeline.py`      — tkinter 스텁 shim (GUI 없이 run_detection 로드)
- `run_A4_auto_v39.py`  — 컬러 파이프라인 (컬러 프로젝트에서 복사)
- `index.html`          — 모드 토글이 추가된 통합 프론트엔드 (chartocode2/webapp/ 또는 src/ 옆)

## 배치 예
```
chartocode2/
├── src/
│   ├── run_gui_v2.py, chart_preprocessing.py, 5_correction_v2.py, ...  (기존 흑백)
│   ├── run_A4_auto_v39.py      (추가: 컬러 파이프라인)
│   ├── unified_server.py       (추가)
│   ├── bw_to_edit_data.py      (추가)
│   └── bw_pipeline.py          (추가)
└── webapp/
    └── index.html              (추가: 통합 프론트엔드)
```

## 실행
```
cd chartocode2/src
pip install fastapi "uvicorn[standard]" python-multipart      # 웹
pip install -r ../requirements.txt                            # 흑백(torch 등)
# 컬러 파이프라인 의존: opencv, numpy, scipy, matplotlib, openpyxl, pytesseract(+tesseract)
python unified_server.py            # http://localhost:8000
```
- 컬러 전용으로만 쓸 경우 torch/timm/ultralytics 불필요 (흑백 의존성은 mode=bw일 때만 lazy import).

## 모드별 입력
| | 컬러 | 흑백 |
|---|---|---|
| plot area | (선택, 없으면 자동) | **필수** |
| legend | 색 팔레트용 | 검출 영역에서 제외용 |
| 축 값(x/y min·max) | 선택(OCR 대체) | **필수** |
| 흑백 전용 | — | 마커 모양 선택, confidence, error bar 유무 |

## 동작
```
POST /digitize (multipart)
  mode=color → run_A4_auto_v39.py (subprocess) → edit_data.json
  mode=bw    → run_detection() (in-process) → 어댑터 → edit_data.json
반환: edit_data_url, input_url, overlay_url, (xlsx_url), log
```
두 모드 모두 같은 edit_data.json → 편집기/재구성/CSV 공통.

## 주의
- 흑백 ViT 마커 검출은 **torch 필요** (로컬 설치). 없으면 세그먼트/에러바 단계만 돌고 마커 0개.
- 프로덕션 배포 시 run_detection을 GUI-free 모듈로 추출하면 bw_pipeline 스텁 없이 깔끔.
