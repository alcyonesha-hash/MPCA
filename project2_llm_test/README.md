# Project 2: LLM Human-like Metrics ON/OFF 비교 실험

Ubuntu IRC 스타일 대화에서 Human-like 메트릭 적용 여부에 따른 에이전트 응답 차이를 비교합니다.

## Quick Start

```bash
cd project2_llm_test

# LLM 서버 실행 (localhost:1234)
# 비교 실험 실행
python run_comparison.py

# 결과 확인
cat outputs/comparison_stats.json
```

## 비교 조건

| 조건 | 설명 |
|------|------|
| **Baseline (OFF)** | LLM 응답 그대로, 즉시 전송 |
| **With Metrics (ON)** | Timing + Chunking 정책 적용 |

## Human-like 메트릭

### Timing 정책

| 유형 | 즉답 확률 | 즉답 범위 |
|------|----------|----------|
| Quick | 71% | 3-10초 |
| Normal | 69% | 3-10초 |
| Detailed | 62% | 3-10초 |

지연 공식: `delay = 10 + word_count × 1.0 + tech_score × 20`

### Chunking 정책

```
Human IRC 연속 발화 분포:
- 67%: 단일 메시지
- 21%: 2개 연속
- 12%: 3개 이상

청크당 최대 15단어 (Human Q75)
청크 간 지연: 1-3초
```

## 실험 결과

| 지표 | Baseline | With Metrics |
|------|----------|--------------|
| 응답 지연 | 0s | 5-15s |
| 유형 분류 | 없음 | Quick/Normal/Detailed |
| Chunking | 단일 | Human-like 분할 |

## 코드 구조

```
project2_llm_test/
├── run_comparison.py         # 메인 비교 스크립트
├── src/
│   ├── llm/
│   │   ├── client.py         # LLM API 클라이언트
│   │   └── generator.py      # 응답 생성기
│   ├── agent/
│   │   └── timing_policy.py  # Timing 정책
│   └── policy/
│       └── conditions.py     # Chunking 정책
├── outputs/                   # 실험 결과
└── TECHNICAL_REPORT.md       # 상세 기술 문서
```

## 상세 문서

- [TECHNICAL_REPORT.md](TECHNICAL_REPORT.md) - 실험 설계 및 결과 분석
