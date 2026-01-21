# Project 1: Ubuntu IRC 데이터 전처리 기술 문서

Ubuntu IRC #ubuntu 채널 로그 데이터에서 대화 에이전트의 행동 파라미터를 추출하는 전처리 파이프라인에 대한 기술 문서입니다.

---

## 1. 개요

### 1.1 목적

Multi-party 대화 환경에서 에이전트가 인간과 유사한 대화 행동을 보이도록 하기 위한 파라미터를 Ubuntu IRC 데이터에서 추출합니다.

### 1.2 데이터셋

| 항목 | 값 |
|------|-----|
| 데이터 소스 | Ubuntu IRC #ubuntu (Libera.chat) |
| 수집 기간 | 2024년 1월 (1개월) |
| 파싱 후 발화 수 | 8,951개 |
| 분리된 Thread 수 | 1,318개 |

### 1.3 파이프라인 구성

```
IRC 로그 → 파싱 → Thread Disentanglement → Opportunity Detection → Response Probability → Policy
```

---

## 2. IRC 로그 파싱

### 2.1 IRCParser 클래스

**파일**: [src/data/parser.py](src/data/parser.py)

IRC 로그를 구조화된 발화 데이터로 변환합니다.

#### 입력 형식

```
[2024-01-15 14:23:12] <alice> message text
```

#### 정규 표현식 패턴

```python
# parser.py:14-16
IRC_LOG_PATTERN = re.compile(
    r'^\[(?P<timestamp>[^\]]+)\]\s+<(?P<speaker>[^>]+)>\s+(?P<text>.*)$'
)
```

#### 출력 스키마

```python
{
    'utt_id': str,      # 발화 고유 ID
    'ts': str,          # ISO 형식 타임스탬프
    'speaker': str,     # 발화자
    'text': str,        # 발화 내용
    'channel': str,     # 채널명
    'mentions': List[str]  # 멘션된 사용자 목록
}
```

### 2.2 멘션 추출 알고리즘

**파일**: [src/data/parser.py:124-159](src/data/parser.py#L124-L159)

세 가지 패턴으로 멘션을 추출합니다:

1. **@username**: 항상 유효
2. **nick: / nick, / nick;**: 문장 시작에서 speaker_list와 대조 검증
3. **단어 경계 매칭**: speaker_list에 있는 닉네임 감지

```python
# parser.py:141
mentions.extend(re.findall(r'@(\w+)', text))

# parser.py:144-149
line_start_match = re.match(r'^(\w+)[:,;]\s*', text)
if line_start_match:
    nick = line_start_match.group(1)
    if not speaker_set or nick.lower() in speaker_set:
        mentions.append(nick)
```

---

## 3. Thread Disentanglement

### 3.1 알고리즘 개요

**파일**: [src/data/disentangle.py](src/data/disentangle.py)

Multi-party 대화에서 여러 개의 논리적 대화 스레드를 분리합니다. 가중치 기반 스코어링과 의미적 유사도를 결합한 하이브리드 접근법을 사용합니다.

### 3.2 스코어링 공식

**Semantic 시간 감쇠 방식**을 적용합니다:

```
base = participant_weight × I[speaker ∈ thread_participants]
     + semantic_weight × semantic_similarity × exp(-Δt/τ)

score = mention_weight + base  (멘션이 있는 경우)
      = base                   (멘션이 없는 경우)
```

### 3.3 파라미터 설정

**파일**: [configs/disentangle.yaml](configs/disentangle.yaml)

| 파라미터 | 값 | 설명 |
|---------|-----|------|
| participant_weight | 1.0 | 참여자 연속성 (시간 감쇠 없음) |
| semantic_weight | 1.0 | 의미 유사도 (시간 감쇠 적용) |
| mention_weight | 2.0 | 멘션 (시간 무관, 항상 병합) |
| time_scale (τ) | 120.0초 | 2분 경과 시 37%로 감쇠 |
| new_thread_threshold | 1.0 | 병합 기준 |
| max_gap_seconds | 1800.0초 | 30분 안전장치 |

### 3.4 각 요소별 동작

**파일**: [src/data/disentangle.py:358-414](src/data/disentangle.py#L358-L414)

```python
def _compute_thread_score(self, speaker, mentions, delta_seconds,
                          thread_participants, utterance_text, thread_embeddings):
    mention_score = self._compute_mention_score(mentions, thread_participants)
    participant_score = self._compute_participant_score(speaker, thread_participants)
    semantic_score = self._compute_semantic_score(utterance_text, thread_embeddings)
    time_decay = self._compute_time_score(delta_seconds)  # exp(-Δt/τ)

    # 시간 감쇠는 semantic에만 적용
    decayed_semantic = semantic_score * time_decay

    base_score = (
        self.participant_weight * participant_score +
        self.semantic_weight * decayed_semantic
    )

    if mention_score > 0:
        total_score = self.mention_weight * mention_score + base_score
    else:
        total_score = base_score

    return total_score
```

#### 시간 감쇠 함수

**파일**: [src/data/disentangle.py:242-264](src/data/disentangle.py#L242-L264)

```python
def _compute_time_score(self, delta_seconds):
    """
    time_score = exp(-delta / time_scale)

    - delta=0 → 1.0
    - delta=τ (120초) → 0.368
    - delta=2τ (240초) → 0.135
    """
    return math.exp(-delta_seconds / self.time_scale)
```

### 3.5 의미적 유사도

**파일**: [src/data/disentangle.py:319-356](src/data/disentangle.py#L319-L356)

Sentence-transformers `all-MiniLM-L6-v2` 모델을 사용하여 발화 간 의미적 유사도를 계산합니다.

```python
def _compute_semantic_score(self, utterance_text, thread_embeddings):
    if self.semantic_weight <= 0 or not thread_embeddings:
        return 0.0

    utt_embedding = self._get_embedding(utterance_text)

    # Thread vector = mean of recent embeddings
    thread_vector = np.mean(thread_embeddings[-self.max_thread_context:], axis=0)

    # L2 normalize
    thread_vector = thread_vector / np.linalg.norm(thread_vector)

    # Cosine similarity
    cosine = float(np.dot(thread_vector, utt_embedding))
    return max(0.0, cosine)
```

---

## 4. Opportunity Detection

### 4.1 개념

**파일**: [src/data/opportunity_detector.py](src/data/opportunity_detector.py)

"응답 기회(Opportunity)"는 에이전트가 응답해야 할 상황을 의미합니다. 모든 발화가 아닌, 응답이 기대되는 발화만을 opportunity로 분류합니다.

### 4.2 Opportunity 유형

**파일**: [src/data/opportunity_detector.py:23-29](src/data/opportunity_detector.py#L23-L29)

```python
class OpportunityType(Enum):
    NONE = "none"
    QUESTION = "question"
    HELP_REQUEST = "help_request"
    INITIATING_WINDOW = "initiating_window"
    DIRECT_MENTION = "direct_mention"
```

### 4.3 우선순위

```
DIRECT_MENTION > QUESTION > HELP_REQUEST > INITIATING_WINDOW > NONE
```

### 4.4 질문 패턴 탐지

**파일**: [src/data/opportunity_detector.py:52-58](src/data/opportunity_detector.py#L52-L58)

```python
QUESTION_PATTERNS = [
    r'\?$',                    # 물음표로 끝남
    r'\?[.!]*$',
    r'^(how|why|what|where|when|who|which|can\s+i|could\s+i|is\s+there|...)',
    r'(any\s+idea|anyone\s+know|somebody\s+help|can\s+someone|...)',
    r'(help\s+me|help\s+with|need\s+help)',
]
```

**점수 계산** ([src/data/opportunity_detector.py:269-288](src/data/opportunity_detector.py#L269-L288)):
```python
def _check_question(self, text):
    if '?' in text:
        score = 0.9
    else:
        score = min(0.8, 0.4 + 0.2 * matches)
    return score
```

### 4.5 도움 요청 탐지

**파일**: [src/data/opportunity_detector.py:61-78](src/data/opportunity_detector.py#L61-L78)

```python
HELP_PATTERNS = [
    r'(i\s+have\s+a\s+problem|i\'m\s+having\s+(an?\s+)?issue|...)',
    r'(doesn\'t\s+work|not\s+working|won\'t\s+work|failed|failing)',
    r'(can\'t|cannot|unable\s+to)\s+\w+',
    r'(error|issue|problem|bug|broken|stuck)',
    ...
]

# 기술 문제 패턴 (Ubuntu 도메인 특화)
TECH_PROBLEM_PATTERNS = [
    r'(apt|apt-get|dpkg|snap)\s+.*(error|fail|broken)',
    r'(permission\s+denied|access\s+denied|sudo)',
    ...
]
```

**점수 계산** ([src/data/opportunity_detector.py:290-309](src/data/opportunity_detector.py#L290-L309)):
```python
def _check_help_request(self, text):
    if help_matches > 0 and tech_matches > 0:
        return 0.85  # 기술 문제 + 도움 요청
    elif help_matches > 0:
        return 0.65
    elif tech_matches > 0:
        return 0.55
    return 0.0
```

### 4.6 임계값

```python
min_question_confidence = 0.5  # opportunity로 인정하는 최소 점수
```

- 질문(`?` 포함): 0.9 → opportunity
- 기술 문제만 언급: 0.55 → opportunity (0.5 초과)
- 일반 대화: 0.0 → not opportunity

---

## 5. Response Probability

### 5.1 개념

**파일**: [src/data/response_probability.py](src/data/response_probability.py)

모든 opportunity에 응답하지 않고, Human helper 패턴을 모방하여 선택적으로 응답합니다.

### 5.2 Human 응답률 데이터

Ubuntu IRC 데이터 분석 결과:

| 상황 | 응답률 |
|------|--------|
| direct_mention | 100% |
| question (첫 응답) | 65.4% |
| question (이미 답변 있음) | 15% |
| help_request (첫 응답) | 55.8% |
| help_request (이미 답변 있음) | 10% |
| init_window | 40% / 10% |

### 5.3 확률 테이블

**파일**: [src/data/response_probability.py:54-66](src/data/response_probability.py#L54-L66)

```python
DEFAULT_PROBABILITIES = {
    # (opportunity_type, has_prior_response) -> probability
    ('direct_mention', False): 1.0,   # 직접 호출 → 무조건
    ('direct_mention', True): 1.0,
    ('question', False): 0.65,        # 첫 응답: 65.4%
    ('question', True): 0.15,         # 이미 답변 있음: 낮춤
    ('help_request', False): 0.56,    # 첫 응답: 55.8%
    ('help_request', True): 0.10,     # 이미 답변 있음: 낮춤
    ('init_window', False): 0.40,
    ('init_window', True): 0.10,
    ('none', False): 0.0,
    ('none', True): 0.0,
}
```

### 5.4 결정 프로세스

**파일**: [src/data/response_probability.py:83-134](src/data/response_probability.py#L83-L134)

```python
def calculate(self, opportunity, has_prior_response):
    opp_type = opportunity.opp_type
    key = (opp_type, has_prior_response)
    probability = self.probabilities.get(key, 0.0)

    # 확률적 결정
    roll = random.random()
    should_respond = roll < probability

    return ResponseProbabilityResult(
        should_respond=should_respond,
        probability=probability,
        ...
    )
```

---

## 6. Timing Policy

### 6.1 개념

**파일**: [src/agent/timing_policy.py](src/agent/timing_policy.py)

응답 지연 시간을 Bimodal 분포로 모델링합니다. Ubuntu IRC Intra-Thread 데이터(2,397개 샘플)에서 추출한 파라미터를 사용합니다.

### 6.2 발화 유형 분류

**파일**: [src/agent/timing_policy.py:253-300](src/agent/timing_policy.py#L253-L300)

```python
def classify_utterance(text):
    """
    분류 기준:
    - Quick: acknowledgment 패턴 OR ≤5 words (비기술적)
    - Normal: 6-20 words, 비기술적
    - Detailed: 기술적 내용 포함 OR >20 words
    """
    if is_acknowledgment(text):
        return 'quick', {'reason': 'acknowledgment'}

    tech_analysis = analyze_technical_content(text)
    if tech_analysis['is_technical']:
        return 'detailed', {'reason': 'technical_content', ...}

    word_count = len(text.split())
    if word_count <= 5:
        return 'quick', {'reason': 'short_non_technical'}
    if word_count > 20:
        return 'detailed', {'reason': 'long_text'}

    return 'normal', {'reason': 'medium_non_technical'}
```

### 6.3 Acknowledgment 패턴

**파일**: [src/agent/timing_policy.py:32-34](src/agent/timing_policy.py#L32-L34)

```python
ACKNOWLEDGMENT_PATTERNS = [
    r'^(yes|no|yep|nope|yeah|nah|ok|okay|k|kk|sure|np|thx|thanks|ty|yw|
       wb|hi|hello|hey|bye|lol|haha|hmm|ah|oh|wow|nice|cool|great|good|
       right|exactly|indeed|correct|true|false|maybe|probably|possibly|
       nvm|nevermind|sorry|oops|please|pls|plz|welp|yup|nah|mhm|uh|um)[\.!\?,\s]*$',
]
```

### 6.4 기술적 내용 판별

**파일**: [src/agent/timing_policy.py:37-119](src/agent/timing_policy.py#L37-L119)

Ubuntu/Linux 도메인 특화 키워드와 패턴:

```python
TECHNICAL_KEYWORDS = [
    'sudo', 'apt', 'apt-get', 'dpkg', ...  # 패키지 관리
    'systemctl', 'service', 'daemon', ...   # 시스템 관리
    'chmod', 'chown', 'mount', ...          # 파일 시스템
    'ssh', 'scp', 'wget', 'curl', ...       # 네트워크
    ...
]

CODE_PATTERNS = [
    r'`[^`]+`',           # backtick 코드
    r'\$\{?\w+\}?',       # 변수: $VAR
    r'--\w+[-\w]*',       # CLI 옵션: --option
    r'-[a-zA-Z]+',        # 짧은 옵션: -rf
    r'\|\s*\w+',          # 파이프: | grep
    ...
]

PATH_PATTERNS = [
    r'/etc/\w+',          # /etc/fstab
    r'/var/\w+',          # /var/log
    r'/usr/\w+',          # /usr/bin
    r'~/\w+',             # ~/Documents
    ...
]
```

### 6.5 Bimodal 지연 파라미터

**파일**: [src/agent/timing_policy.py:315-358](src/agent/timing_policy.py#L315-L358)

```python
DATA_DRIVEN_PARAMS = {
    'quick': TypeTimingParams(
        p_immediate=0.71,           # 71.0% 즉답
        immediate_range=(3, 10),    # 즉답: 3~10초
        ...
    ),
    'normal': TypeTimingParams(
        p_immediate=0.69,           # 69.1% 즉답
        immediate_range=(3, 10),
        ...
    ),
    'detailed': TypeTimingParams(
        p_immediate=0.62,           # 62.4% 즉답
        immediate_range=(3, 10),
        ...
    ),
}

# 지연 응답 계수
DELAY_BASE = 10.0           # 기본 지연 (초)
DELAY_PER_WORD = 1.0        # 단어당 추가 지연
DELAY_PER_TECH = 20.0       # 기술점수당 추가 지연
```

### 6.6 지연 계산 공식

**파일**: [src/agent/timing_policy.py:430-456](src/agent/timing_policy.py#L430-L456)

```python
def _sample_delay_internal(self, text, utt_type, analysis):
    p = self.params[utt_type]

    if random.random() < p.p_immediate:
        # 즉답: uniform(3, 10)
        delay = random.uniform(*p.immediate_range)
    else:
        # 지연: 공식 기반
        word_count = len(text.split())
        tech_score = analysis.get('technical', {}).get('technical_score', 0.0)
        delay = DELAY_BASE + (word_count * DELAY_PER_WORD) + (tech_score * DELAY_PER_TECH)

    return max(self.min_delay, min(delay, self.max_delay))
```

---

## 7. Topical Fit

### 7.1 개념

**파일**: [src/policy/topic.py](src/policy/topic.py)

Embedding 기반으로 발화가 현재 주제와 관련 있는지 측정합니다.

### 7.2 Topic Vector 계산

**파일**: [src/policy/topic.py:250-269](src/policy/topic.py#L250-L269)

```python
def _update_topic_vector(self):
    """
    Topic vector = mean of window embeddings (L2 normalized)
    """
    embeddings = np.array([emb for _, emb in self.window])
    mean_vec = np.mean(embeddings, axis=0)

    norm = np.linalg.norm(mean_vec)
    if norm > 0:
        mean_vec = mean_vec / norm

    self.current_topic_vec = mean_vec
```

### 7.3 Relevance 계산

**파일**: [src/policy/topic.py:280-315](src/policy/topic.py#L280-L315)

```python
def compute_cosine(self, utterance_text):
    """
    Cosine similarity between utterance and topic vector.
    Returns 1.0 if no topic yet (initial state).
    """
    if self.current_topic_vec is None:
        return 1.0

    utt_embedding = self._get_embedding(utterance_text)
    cosine = float(np.dot(self.current_topic_vec, utt_embedding))

    return max(0.0, cosine)
```

### 7.4 파라미터

| 파라미터 | 값 | 설명 |
|---------|-----|------|
| window_size | 7 | 최근 발화 수 (topic vector 계산용) |
| drift_threshold | 0.30 | 이 값 미만이면 topic drift |

### 7.5 Thread-based Topical Fit

**파일**: [src/policy/topic.py:386-418](src/policy/topic.py#L386-L418)

LLM 응답이 올바른 thread에 속하는지 검증:

```python
def compute_thread_fit(self, response_text):
    """
    LLM 응답이 현재 thread에 맞는지 평가.

    Returns:
        (cosine_similarity, is_correct_thread)
    """
    response_emb = self._get_embedding(response_text)
    cosine = float(np.dot(self._thread_vector, response_emb))
    is_correct = cosine >= self.drift_threshold

    return cosine, is_correct
```

---

## 8. Policy Conditions

### 8.1 세 가지 조건

**파일**: [src/policy/conditions.py](src/policy/conditions.py)

| 조건 | 설명 | 적용 모듈 |
|------|------|----------|
| C0 (Baseline) | Opportunity에서만 응답 | OpportunityDetector, ResponseProbability |
| C1 (+M1) | 참여 확률 + relevance gate + timing | + TopicTracker, TimingPolicy |
| C2 (+M1+M2) | C1 + form control + chunking + drift stop | + Chunking, DriftStop |

### 8.2 C0: Baseline

**파일**: [src/policy/conditions.py:87-163](src/policy/conditions.py#L87-L163)

```python
def should_respond(self, utterance, context, opp_result=None, ...):
    if opp_result is None:
        opp_result = self.opportunity_detector.detect(utterance, context)

    if not opp_result.is_opportunity:
        return False, 0.0, "Not an opportunity", {}

    # ResponseProbabilityCalculator로 Human-like 확률 적용
    prob_result = self.response_calculator.calculate(
        opportunity=opp_result,
        has_prior_response=prior_response
    )

    if prob_result.should_respond:
        return True, self.config.default_delay, f"C0: {prob_result.reason}", {...}
    else:
        return False, 0.0, f"C0: {prob_result.reason}", {}
```

### 8.3 C1: +M1 (Participation)

**파일**: [src/policy/conditions.py:166-355](src/policy/conditions.py#L166-L355)

C0에 추가:
- **Relevance gate**: 주제 관련성 검사
- **TimingPolicy**: Bimodal 지연
- **Turn constraint**: burst suppression

```python
def should_respond(self, utterance, context, ...):
    # 1. Opportunity 체크
    if not opp_result.is_opportunity:
        return False, 0.0, "Not an opportunity", {}

    # 2. Turn constraint (burst suppression)
    if self.consecutive_turns >= self.config.max_consecutive_turns:
        # Cooldown 체크
        ...

    # 3. Relevance gate
    relevance = self._compute_relevance(utterance, context)
    if relevance < self.config.relevance_threshold:
        return False, 0.0, f"Low relevance: {relevance:.3f}", {}

    # 4. Human-like 확률 결정
    prob_result = self.response_calculator.calculate(...)

    # 5. TimingPolicy 기반 지연 계산
    delay, utt_type, analysis = self._sample_delay_with_type(text_for_timing)

    return True, delay, reason, metadata
```

### 8.4 C2: +M1+M2 (Full)

**파일**: [src/policy/conditions.py:357-588](src/policy/conditions.py#L357-L588)

C1에 추가:
- **Chunking**: 긴 응답을 여러 메시지로 분할
- **Drift stop**: topic drift 시 남은 chunk 중단

#### Human-like Chunking 분포

**파일**: [src/policy/conditions.py:73-77](src/policy/conditions.py#L73-L77)

```python
human_consecutive_dist = {
    1: 0.67,  # 67% 단일 메시지
    2: 0.21,  # 21% 2개 메시지
    3: 0.12,  # 12% 3개 이상
}
```

#### Chunking 알고리즘

**파일**: [src/policy/conditions.py:393-446](src/policy/conditions.py#L393-L446)

```python
def chunk_response(self, text):
    """
    응답을 Human-like 청크로 분할.

    - 각 청크 최대 15단어 (human Q75)
    - 청크 간 지연: 1-3초
    """
    max_len = self.config.max_words  # 15 words

    if len(words) <= max_len:
        return [(text, 0.0)]

    # Human-like 분포에서 목표 청크 수 샘플링
    target_chunks = self._sample_chunk_count()

    # 최소 청크 수 보장 (정보 손실 방지)
    min_chunks_needed = (len(words) + max_len - 1) // max_len
    target_chunks = max(target_chunks, min_chunks_needed)

    # 문장 경계 우선 분할
    chunks = self._group_sentences_into_chunks(sentences, target_chunks, max_len)

    # 지연 시간 추가
    result = []
    for i, chunk_text in enumerate(chunks):
        delay = 0.0 if i == 0 else random.uniform(1.0, 3.0)
        result.append((chunk_text, delay))

    return result
```

---

## 9. 코드 구조

```
project1_preprocessing/
├── configs/
│   └── disentangle.yaml          # Thread Disentanglement 파라미터
├── data/
│   └── ubuntu_merged.txt         # 원본 IRC 로그
├── src/
│   ├── data/
│   │   ├── __init__.py
│   │   ├── parser.py             # IRC 로그 파싱
│   │   ├── disentangle.py        # Thread Disentanglement
│   │   ├── opportunity_detector.py  # 응답 기회 탐지
│   │   └── response_probability.py  # 선택적 응답 확률
│   ├── agent/
│   │   └── timing_policy.py      # Bimodal 지연 정책
│   └── policy/
│       ├── __init__.py
│       ├── topic.py              # Topical Fit
│       └── conditions.py         # 정책 조건 (C0/C1/C2)
├── outputs/                       # 처리 결과
└── README.md
```

---

## 10. 사용 예시

### 10.1 기본 파이프라인

```python
from src.data import IRCParser, ThreadDisentangler

# 1. 파싱
parser = IRCParser()
utterances = parser.parse_file('data/ubuntu_merged.txt')
print(f'Parsed: {len(utterances)} utterances')

# 2. Thread Disentanglement
disentangler = ThreadDisentangler()
threads = disentangler.disentangle(utterances)
print(f'Threads: {len(threads)}')
```

### 10.2 Opportunity Detection + Response Probability

```python
from src.data import OpportunityDetector, ResponseProbabilityCalculator

detector = OpportunityDetector(agent_name="helper_bot")
calculator = ResponseProbabilityCalculator()

for thread in threads:
    opp_results = detector.detect_thread(thread)
    prob_results = calculator.calculate_thread(thread, opp_results, agent_name="helper_bot")

    for utt, opp, prob in prob_results:
        if prob.should_respond:
            print(f"Should respond to: {utt['text'][:50]}...")
```

### 10.3 Policy 적용

```python
from src.policy import create_policy

# C1 정책 생성
config, policy = create_policy("C1")

# 응답 결정
should_respond, delay, reason, metadata = policy.should_respond(
    utterance={"text": "How do I install numpy?", "speaker": "user1"},
    context=previous_utterances
)

if should_respond:
    print(f"Respond after {delay:.1f}s, type: {metadata['utt_type']}")
```

---

## 11. 파라미터 요약

### Thread Disentanglement

| 파라미터 | 값 | 설명 |
|---------|-----|------|
| participant_weight | 1.0 | 참여자 연속성 |
| semantic_weight | 1.0 | 의미 유사도 |
| mention_weight | 2.0 | 멘션 |
| time_scale | 120초 | 시간 감쇠 상수 |
| new_thread_threshold | 1.0 | 병합 기준 |
| max_gap_seconds | 1800초 | 안전장치 |

### Opportunity Detection

| 파라미터 | 값 |
|---------|-----|
| min_question_confidence | 0.5 |
| initiating_window_seconds | 120초 |

### Response Probability

| 상황 | 확률 |
|------|------|
| direct_mention | 100% |
| question (첫 응답) | 65% |
| question (이미 답변) | 15% |
| help_request (첫 응답) | 56% |
| help_request (이미 답변) | 10% |

### Timing Policy

| 유형 | 즉답 확률 | 즉답 범위 |
|------|----------|----------|
| quick | 71% | 3-10초 |
| normal | 69% | 3-10초 |
| detailed | 62% | 3-10초 |

지연 공식: `delay = 10 + word_count × 1.0 + tech_score × 20`

### Topical Fit

| 파라미터 | 값 |
|---------|-----|
| window_size | 7 |
| drift_threshold | 0.30 |

---

## 12. 임베딩 모델

### 12.1 사용 모델

- **Primary**: `sentence-transformers/all-MiniLM-L6-v2`
- **Fallback**: HuggingFace transformers mean-pooling

### 12.2 임베딩 차원

- 384차원 (all-MiniLM-L6-v2)

### 12.3 정규화

모든 임베딩은 L2 정규화되어 cosine similarity = dot product
