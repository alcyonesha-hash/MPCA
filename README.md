# MPCA: Multi-Party Conversation Agent

Ubuntu IRC 채널 대화 로그 분석을 통한 Human-like 대화 에이전트 파라미터 추출 및 검증 프로젝트입니다.

## 프로젝트 구조

```
MPCA/
├── project1_preprocessing/   # IRC 데이터 전처리 및 파라미터 추출
│   ├── src/data/            # 파싱, Thread Disentanglement, 필터링
│   ├── src/agent/           # Timing 정책
│   ├── src/policy/          # Topical Fit, Chunking
│   └── configs/             # 파라미터 설정
│
└── project2_llm_test/        # LLM 메트릭 ON/OFF 비교 실험
    ├── src/llm/             # LLM 클라이언트
    └── run_comparison.py    # 비교 실험 스크립트
```

## 핵심 기능

### 1. Thread Disentanglement (project1)

IRC 채널의 혼합 대화를 개별 스레드로 분리합니다.

**Semantic 시간 감쇠 알고리즘**:
- 참여자 연속성: 시간 무관하게 유지
- 의미 유사도: 시간에 따라 감쇠 (exp(-Δt/τ))
- 멘션: 항상 병합

| 파라미터 | 값 |
|---------|-----|
| participant_weight | 1.0 |
| semantic_weight | 1.0 |
| mention_weight | 2.0 |
| time_scale (τ) | 120초 |
| threshold | 1.0 |

**결과**: 8,951 발화 → 1,318 스레드

### 2. Human-like Timing (project1 → project2)

| 유형 | 즉답 확률 | 지연 공식 |
|------|----------|----------|
| Quick | 71% | 3-10초 |
| Normal | 69% | 3-10초 |
| Detailed | 62% | 10 + words×1.0 + tech×20 |

### 3. Chunking Policy (project1 → project2)

```
Human 분포: 67% 단일 / 21% 2개 / 12% 3+
청크당 최대 15단어 (Human Q75)
```

## Quick Start

```bash
# Project 1: 전처리
cd project1_preprocessing
python -c "
from src.data import IRCParser, ThreadDisentangler
parser = IRCParser()
utts = parser.parse_file('data/ubuntu_merged.txt')
threads = ThreadDisentangler().disentangle(utts)
print(f'{len(utts)} utterances → {len(threads)} threads')
"

# Project 2: LLM 비교 실험
cd ../project2_llm_test
python run_comparison.py
```

## 문서

- [Project 1 Technical Report](project1_preprocessing/TECHNICAL_REPORT.md)
- [Project 2 Technical Report](project2_llm_test/TECHNICAL_REPORT.md)

## 데이터셋

| 항목 | 값 |
|------|-----|
| 소스 | Ubuntu IRC #ubuntu (Libera.chat) |
| 기간 | 2024.01 (1개월) |
| 발화 수 | 8,951개 |
| 스레드 수 | 1,318개 |
