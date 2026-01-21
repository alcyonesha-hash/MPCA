# Project 3: Human-like Metrics User Survey

Human-like 메트릭의 효과를 검증하기 위한 사용자 설문 실험입니다.

## Quick Start

```bash
cd project3_survey

# 1. GIF/비디오 생성 (Python 필요)
pip install pillow imageio numpy
python scripts/generate_gifs.py

# 2. 로컬 서버 실행
python -m http.server 8000

# 3. 브라우저에서 열기
open http://localhost:8000
```

## 실험 설계

### 참가자
- 20명
- 이름/이메일 수집 (추적용)
- IT 경험 수준 조사

### 비교 조건 (10 세트)

| 세트 | 비교 | 형식 | 검증 내용 |
|------|------|------|----------|
| 1-4 | Full vs Baseline | 텍스트 | 전체 메트릭 효과 |
| 5-6 | Full vs No Timing | GIF | Timing 효과 |
| 7-8 | Full vs No Topical Fit | 텍스트 | 주제 관련성 효과 |
| 9-10 | Full vs No Chunking | GIF | 메시지 분할 효과 |

### 질문 (각 세트당)

1. **Q1**: Which is more human-like?
2. **Q2**: Which is less interruptive/annoying?
3. **Q3**: Which has more natural conversational flow?
4. **Q4**: Which would you keep in your group chat?

### 통제 변수

- **A/B 순서 랜덤화**: 각 세트에서 Full이 A인지 B인지 랜덤
- **세트 순서 랜덤화**: 10개 세트의 제시 순서 랜덤
- **Attention Check**: 5번째 세트 후 삽입

## 코드 구조

```
project3_survey/
├── index.html              # 메인 설문 페이지
├── src/
│   ├── styles.css          # 스타일
│   ├── app.js              # 설문 로직
│   └── data.js             # 비교 세트 데이터
├── data/
│   └── gifs/               # 생성된 비디오 파일
├── scripts/
│   ├── generate_gifs.py    # GIF 생성 스크립트
│   └── google_sheets_setup.md  # Google Sheets 연동 가이드
└── README.md
```

## 데이터 수집

### Google Sheets 연동

1. Google Sheet 생성
2. Apps Script로 Web App 배포
3. `index.html`에 URL 설정

자세한 설정: [scripts/google_sheets_setup.md](scripts/google_sheets_setup.md)

### 백업

Google Sheets 실패 시 자동으로 JSON 파일 다운로드

## 설문 흐름

```
① Consent (동의) → ② Info (참가자 정보) → ③ Instructions (안내)
    ↓
④ Main Survey (10 세트, 각 4문항)
    - 5번째 세트 후 Attention Check 삽입
    ↓
⑤ Complete (완료 + Debrief)
```

## 데이터 분석

수집된 데이터 컬럼:

| 컬럼 | 설명 |
|------|------|
| PrefersFullQ1-Q4 | Full 조건 선호 여부 (True/False) |
| ComparisonType | Baseline/NoTiming/NoTopicalFit/NoChunking |
| TimeSpentMs | 각 세트 응답 소요 시간 |

### 분석 예시

```python
import pandas as pd

df = pd.read_csv('responses.csv')

# 조건별 Full 선호율
df.groupby('ComparisonType')['PrefersFullQ1'].mean()

# Q1-Q4 평균 Full 선호율
df[['PrefersFullQ1', 'PrefersFullQ2', 'PrefersFullQ3', 'PrefersFullQ4']].mean()
```

## 배포

### GitHub Pages

```bash
git add .
git commit -m "Add survey"
git push

# Settings > Pages > Source: main branch
```

### Vercel

```bash
npm i -g vercel
vercel
```

## 대화 샘플 출처

Ubuntu IRC #ubuntu 채널 데이터 (2024.01)에서 추출 후 수정

## 상세 문서

- Project 1: [../project1_preprocessing/TECHNICAL_REPORT.md](../project1_preprocessing/TECHNICAL_REPORT.md)
- Project 2: [../project2_llm_test/TECHNICAL_REPORT.md](../project2_llm_test/TECHNICAL_REPORT.md)
