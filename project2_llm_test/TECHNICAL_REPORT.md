# Project 2: LLM 메트릭 ON/OFF 비교 실험

## 개요

Ubuntu IRC 스타일 대화에서 Human-like 메트릭 적용 여부에 따른 에이전트 응답 차이를 비교합니다.

---

## 1. 실험 설계

### 비교 조건

| 조건 | 설명 |
|------|------|
| **Baseline (OFF)** | LLM 응답 그대로 사용, 즉시 전송 |
| **With Metrics (ON)** | Timing + Chunking 정책 적용 |

### 평가 지표

1. **응답 길이 분포**: Human Q75=15단어 기준
2. **Chunk 수 분포**: Human 67%/21%/12% 기준
3. **응답 지연 분포**: Quick/Normal/Detailed 유형별

### LLM 설정

```yaml
endpoint: http://localhost:1234/v1/chat/completions
model: openai/gpt-oss-120b
target_words: 8  # Human median
max_words: 15    # Human Q75
```

---

## 2. Human 기준 파라미터

### 2.1 Timing 파라미터 (Project 1에서 추출)

| 유형 | 즉답 확률 | 즉답 범위 | 지연 공식 |
|------|----------|----------|----------|
| Quick | 71% | 3-10초 | - |
| Normal | 69% | 3-10초 | - |
| Detailed | 62% | 3-10초 | 10 + words × 1.0 + tech × 20 |

### 2.2 Chunk 분포 파라미터

```
Human IRC 연속 발화 분포:
- 67%: 단일 메시지
- 21%: 2개 연속
- 12%: 3개 이상

각 청크 최대 15단어 (Q75)
청크 간 지연: 1-3초
```

### 2.3 기술적 복잡도 점수

| 지표 | 점수 |
|------|------|
| URL | +0.3 |
| 코드 패턴 | +0.25 (max 0.5) |
| 경로 패턴 | +0.3 |
| 기술 키워드 | +0.1 (max 0.5) |

---

## 3. 실험 결과

### 3.1 테스트 프롬프트

```python
TEST_PROMPTS = [
    # Quick 응답 예상
    "thanks for the help",
    "ok I'll try that",
    "yes that works",

    # Normal 응답 예상
    "How do I install Python on Ubuntu?",
    "My wifi keeps disconnecting after suspend",
    "apt-get says package not found",

    # Detailed 응답 예상
    "I'm getting a dependency error when running apt-get install nodejs",
    "The system won't boot after grub update, shows error: file not found",
    "How do I configure nginx as a reverse proxy for my Node.js app?",
]
```

### 3.2 비교 결과

#### [1] 응답 지연 (Delay)

| 조건 | 평균 | 즉답률 (≤10s) |
|------|------|--------------|
| **Baseline** | 0.0s | 100% |
| **With Metrics** | 8.1s | 78% |
| **Human 기준** | ~5-15s | ~67% |

**분석**:
- Baseline은 지연 없이 즉시 응답 (비현실적)
- With Metrics는 Human 패턴에 가깝게 지연 적용

#### [2] 유형 분류 (Classification)

| 유형 | Baseline | With Metrics |
|------|----------|--------------|
| Quick | 0 | 3 |
| Normal | 0 | 1 |
| Detailed | 0 | 5 |
| Unknown | 9 | 0 |

**분석**:
- Baseline은 유형 분류 없음
- With Metrics는 응답 내용 기반 자동 분류

#### [3] 기술적 복잡도 점수 예시

| 응답 | Tech Score | 유형 |
|------|-----------|------|
| `"sudo dpkg --configure -a && sudo apt-get"` | 0.90 | Detailed |
| `"ufw allow ssh; ufw allow http; ufw enable"` | 0.30 | Detailed |
| `"Check inode usage with"` | 0.00 | Quick |
| `"Good luck!"` | 0.00 | Quick |

---

## 4. 샘플 출력 비교

### 샘플 1: Quick 응답

**Prompt**: "ok I'll try that"

| 조건 | 응답 | Delay |
|------|------|-------|
| Baseline | "Good luck!" | 0.0s |
| With Metrics | "Good luck!" | 7.7s |

→ **차이**: Metrics 적용 시 자연스러운 지연 추가

### 샘플 2: Detailed 응답

**Prompt**: "How do I fix apt-get broken packages?"

| 조건 | 응답 | Delay | Tech Score |
|------|------|-------|-----------|
| Baseline | `"Run sudo dpkg..."` | 0.0s | - |
| With Metrics | `"Run sudo dpkg..."` | 4.4s | 0.90 |

→ **차이**: 기술적 내용 인식, 적절한 지연 적용

### 샘플 3: Chunking 예시 (긴 응답)

**Prompt**: "How to configure nginx as reverse proxy?"

```
Baseline (단일 응답):
  [0.0s] "First install nginx, then edit /etc/nginx/nginx.conf..."

With Metrics (청킹 적용):
  [0.0s] "First install nginx with apt."
  [1.5s] "Then edit /etc/nginx/nginx.conf."
  [3.2s] "Set proxy_pass to your backend."
```

→ **차이**: Human-like 분할 및 타이핑 지연 시뮬레이션

---

## 5. 결론

### Metrics ON의 효과

| 지표 | Baseline | With Metrics | 개선 |
|------|----------|--------------|------|
| 응답 지연 | 0s (비현실적) | 5-15s (Human-like) | ✅ |
| 유형 분류 | 없음 | Quick/Normal/Detailed | ✅ |
| Chunking | 단일 응답 | 분할 가능 | ✅ |
| Tech Score | 미적용 | 0.0-1.0 | ✅ |

### Human-likeness 달성도

| 항목 | Human 기준 | With Metrics |
|------|-----------|--------------|
| 즉답률 | ~67% | 78% |
| 지연 범위 | 3-10s (즉답) | 3-10s |
| Chunk Q75 | 15단어 | 15단어 제한 |

---

## 6. 실행 방법

```bash
# 비교 실험 실행
python project2_llm_test/run_comparison.py

# 결과 확인
cat project2_llm_test/outputs/comparison_stats.json
```

---

## 7. 코드 구조

```
project2_llm_test/
├── run_comparison.py       # 메인 비교 실험 스크립트
├── outputs/
│   ├── baseline_results.json
│   ├── metrics_results.json
│   └── comparison_stats.json
└── TECHNICAL_REPORT.md     # 이 문서

의존성 코드:
├── src/llm/client.py           # LLM 클라이언트
├── src/agent/timing_policy.py  # Timing 정책
└── src/policy/conditions.py    # Chunking 정책
```

---

## 8. 코드 레퍼런스

### Timing Policy 적용
**파일**: `src/agent/timing_policy.py:388-441`

```python
def sample_delay(self, text: str) -> float:
    utt_type, analysis = classify_utterance(text)
    return self._sample_delay_internal(text, utt_type, analysis)

def _sample_delay_internal(self, text, utt_type, analysis):
    if random.random() < p.p_immediate:
        delay = random.uniform(*p.immediate_range)  # 3-10초
    else:
        delay = 10 + word_count * 1.0 + tech_score * 20
```

### Chunking Policy 적용
**파일**: `src/policy/conditions.py:324-377`

```python
def chunk_response(self, text: str) -> List[Tuple[str, float]]:
    # 15단어 이하: 분할 없음
    if len(words) <= max_len:
        return [(text, 0.0)]

    # Human-like 분포 샘플링 (67/21/12)
    target_chunks = self._sample_chunk_count()

    # 문장 경계에서 분할
    chunks = self._group_sentences_into_chunks(sentences, target_chunks, max_len)

    # 청크 간 1-3초 지연
    for i, chunk_text in enumerate(chunks):
        delay = 0.0 if i == 0 else random.uniform(1.0, 3.0)
```

### 기술적 복잡도 계산
**파일**: `src/agent/timing_policy.py:145-190`

```python
def analyze_technical_content(text: str) -> Dict:
    tech_score = 0.0

    if url_pattern.search(text): tech_score += 0.3
    if path_pattern.search(text): tech_score += 0.3
    code_matches = len(code_patterns_found)
    tech_score += min(code_matches * 0.25, 0.5)
    keyword_matches = len(keyword_found)
    tech_score += min(keyword_matches * 0.1, 0.5)

    return {'technical_score': min(tech_score, 1.0)}
```
