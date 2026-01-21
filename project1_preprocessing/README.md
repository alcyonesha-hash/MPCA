# Project 1: Ubuntu IRC Data Preprocessing

Ubuntu IRC #ubuntu 채널 로그에서 대화 에이전트의 행동 파라미터를 추출하는 전처리 파이프라인입니다.

## Quick Start

```bash
cd project1_preprocessing

# 의존성 설치
pip install -r requirements.txt

# 전처리 실행
python -c "
from src.data import IRCParser, ThreadDisentangler

# 1. 파싱
parser = IRCParser()
utterances = parser.parse_file('data/ubuntu_merged.txt')
print(f'Parsed: {len(utterances)} utterances')

# 2. Thread Disentanglement
disentangler = ThreadDisentangler()
threads = disentangler.disentangle(utterances)
print(f'Threads: {len(threads)}')
"
```

## 데이터셋 정보

| 항목 | 값 |
|------|-----|
| 데이터 소스 | Ubuntu IRC #ubuntu (Libera.chat) |
| 수집 기간 | 2024.01 (1개월) |
| 파싱 후 발화 수 | 8,951개 |
| 분리된 Thread 수 | 1,318개 |

## Thread Disentanglement 알고리즘

### Semantic 시간 감쇠 방식

시간 감쇠를 **의미 유사도에만** 적용하여, 참여자 연속성은 유지하면서 주제 관련성은 시간에 따라 약화됩니다.

```
base = participant_weight × I[speaker in thread]
     + semantic_weight × semantic_sim × exp(-Δt/τ)

score = mention_weight + base  (멘션 있음)
      = base                   (멘션 없음)
```

### 파라미터

| 파라미터 | 값 | 설명 |
|---------|-----|------|
| participant_weight | 1.0 | 참여자 연속성 (시간 감쇠 없음) |
| semantic_weight | 1.0 | 의미 유사도 (시간 감쇠 적용) |
| mention_weight | 2.0 | 멘션 (시간 무관, 항상 병합) |
| time_scale (τ) | 120초 | 2분 경과 시 37%로 감쇠 |
| threshold | 1.0 | 병합 기준 |
| max_gap_seconds | 1800초 | 30분 안전장치 |

### 각 요소별 동작

- **participant**: 같은 화자면 시간 무관하게 +1.0 → threshold 충족
- **semantic**: exp(-Δt/τ)로 시간에 따라 감쇠
- **mention**: 항상 +2.0 → 강제 병합

## 코드 구조

```
project1_preprocessing/
├── configs/
│   └── disentangle.yaml      # 파라미터 설정
├── data/
│   └── ubuntu_merged.txt     # 원본 IRC 로그
├── src/
│   ├── data/
│   │   ├── parser.py               # IRC 로그 파싱
│   │   ├── disentangle.py          # Thread Disentanglement
│   │   ├── opportunity_detector.py # 응답 기회 탐지
│   │   └── response_probability.py # 선택적 응답 확률
│   ├── agent/
│   │   └── timing_policy.py        # Timing 파라미터
│   └── policy/
│       ├── topic.py                # Topical Fit
│       └── conditions.py           # Chunking 정책
├── outputs/                   # 처리 결과
├── TECHNICAL_REPORT.md       # 상세 기술 문서
└── README.md
```

## 상세 문서

- [TECHNICAL_REPORT.md](TECHNICAL_REPORT.md) - 알고리즘 상세 설명 및 파라미터 도출 과정
