"""
Policy Conditions Module (Spec 6)

3가지 조건:
- C0 (Baseline): is_opportunity에서만 응답, 확률 1.0 or config
- C1 (+M1): 참여 확률 + relevance gate + timing (TimingPolicy 사용)
- C2 (+M1+M2): C1 + form control + chunking + drift stop

중요: 동일한 base LLM 고정! 정책만 변경
TimingPolicy: 응답 유형(quick/normal/detailed) 기반 Bimodal 지연
"""

import random
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from datetime import datetime

from src.data.opportunity_detector import OpportunityDetector, OpportunityResult, OpportunityType
from src.data.response_probability import ResponseProbabilityCalculator, ResponseProbabilityResult
from src.policy.topic import TopicTracker
from src.agent.timing_policy import (
    TimingPolicy, classify_utterance, classify_utterance_simple,
    analyze_technical_content
)

logger = logging.getLogger(__name__)


@dataclass
class PolicyConfig:
    """정책 설정 (Spec 6.1~6.3)"""
    name: str = "baseline"
    condition: str = "C0"  # C0, C1, C2

    # C0 baseline config
    baseline_respond_prob: float = 1.0  # opportunity에서 응답 확률

    # C1 참여 확률 (from data)
    p_mention: float = 0.9       # P(respond|mentioned)
    p_spont: float = 0.5         # P(respond|opportunity, no mention)

    # C1 relevance gate
    relevance_k: int = 5         # context window
    relevance_threshold: float = 0.3  # min cosine similarity

    # C1 timing - TimingPolicy 사용 여부
    use_timing_policy: bool = True  # True: Bimodal TimingPolicy 사용
    timing_scale_factor: float = 1.0  # delay 스케일 (1.0=원본, 0.5=절반)

    # C1 timing (legacy - use_timing_policy=False 시 사용)
    delay_mention_params: Dict = field(default_factory=dict)
    delay_opp_params: Dict = field(default_factory=dict)
    default_delay: float = 30.0

    # C1 turn constraint
    max_consecutive_turns: int = 3
    burst_cooldown_sec: float = 60.0

    # C2 form control
    target_words: int = 8
    max_words: int = 15

    # C2 chunking (Human-like distribution)
    chunking_enabled: bool = False
    chunk_delay_sec: float = 2.0  # 청크 간 지연 (1-3초 범위)
    chunk_delay_min: float = 1.0
    chunk_delay_max: float = 3.0

    # Human-like consecutive distribution (from Ubuntu IRC data)
    # 67% single message, 21% 2 messages, 12% 3+ messages
    human_consecutive_dist: Dict = field(default_factory=lambda: {
        1: 0.67,  # 67% single message
        2: 0.21,  # 21% 2 messages
        3: 0.12,  # 12% 3+ messages
    })

    # C2 drift stop
    drift_stop_enabled: bool = False
    drift_threshold: float = 0.30  # Data-derived optimal (F1=85.3)

    # Agent name
    agent_name: str = "agent"


class ConditionC0:
    """
    Baseline: Opportunity에서만 응답 (Human-like 확률 적용)

    Spec 6.3 C0:
    - is_opportunity인 경우에만 응답 후보
    - ResponseProbabilityCalculator로 Human-like 확률 적용
    - 고정 delay 사용 (TimingPolicy 미적용)

    Human 응답 패턴 (IRC 데이터 분석):
    - direct_mention: 100%
    - question (첫 응답): 65%
    - question (이미 답변 있음): 15%
    - help_request (첫 응답): 56%
    - help_request (이미 답변 있음): 10%
    """

    def __init__(self, config: PolicyConfig):
        self.config = config
        self.opportunity_detector = OpportunityDetector(agent_name=config.agent_name)
        self.response_calculator = ResponseProbabilityCalculator()
        self.has_responded_in_thread = False  # 스레드 내 응답 여부 추적

    def should_respond(
        self,
        utterance: Dict,
        context: List[Dict],
        opp_result: OpportunityResult = None,
        response_text: str = None,
        has_prior_response: bool = None
    ) -> Tuple[bool, float, str, Dict]:
        """
        응답 여부 결정

        Args:
            utterance: 현재 발화
            context: 이전 발화들
            opp_result: opportunity 결과 (없으면 계산)
            response_text: 응답 텍스트 (미사용)
            has_prior_response: 이미 다른 사람이 응답했는지 (None이면 self.has_responded_in_thread 사용)

        Returns:
            (should_respond, delay, reason, metadata)
        """
        if opp_result is None:
            opp_result = self.opportunity_detector.detect(utterance, context)

        if not opp_result.is_opportunity:
            return False, 0.0, "Not an opportunity", {}

        # has_prior_response 결정
        prior_response = has_prior_response if has_prior_response is not None else self.has_responded_in_thread

        # ResponseProbabilityCalculator로 확률적 결정
        prob_result = self.response_calculator.calculate(
            opportunity=opp_result,
            has_prior_response=prior_response
        )

        if prob_result.should_respond:
            self.has_responded_in_thread = True
            return (
                True,
                self.config.default_delay,
                f"C0: {prob_result.reason}",
                {
                    'utt_type': 'baseline',
                    'probability': prob_result.probability,
                    'opportunity_type': opp_result.opp_type
                }
            )
        else:
            return False, 0.0, f"C0: {prob_result.reason}", {}

    def reset_thread_state(self):
        """새 스레드 시작 시 상태 리셋"""
        self.has_responded_in_thread = False


class ConditionC1:
    """
    +M1: Human-like 응답 확률 + relevance gate + timing

    Spec 6.3 C1:
    - ResponseProbabilityCalculator로 Human-like 확률 적용
    - relevance gate: 최근 k발화 TF-IDF/embedding cosine >= rho
    - timing: TimingPolicy 기반 Bimodal 지연 (quick/normal/detailed)
    - turn constraint: burst suppression

    Human 응답 패턴 (IRC 데이터 분석):
    - direct_mention: 100%
    - question (첫 응답): 65%
    - question (이미 답변 있음): 15%
    - help_request (첫 응답): 56%
    - help_request (이미 답변 있음): 10%
    """

    def __init__(self, config: PolicyConfig):
        self.config = config
        self.opportunity_detector = OpportunityDetector(agent_name=config.agent_name)
        self.response_calculator = ResponseProbabilityCalculator()
        self.topic_tracker = TopicTracker()

        # TimingPolicy 초기화 (Bimodal 지연)
        self.timing_policy = TimingPolicy(
            scale_factor=config.timing_scale_factor,
            max_delay=900.0,  # 15분
            min_delay=3.0,
        )

        # Turn tracking
        self.consecutive_turns = 0
        self.last_response_time: Optional[datetime] = None
        self.has_responded_in_thread = False  # 스레드 내 응답 여부 추적

        logger.info(f"C1 initialized with TimingPolicy (scale={config.timing_scale_factor})")

    def should_respond(
        self,
        utterance: Dict,
        context: List[Dict],
        opp_result: OpportunityResult = None,
        response_text: str = None,
        has_prior_response: bool = None
    ) -> Tuple[bool, float, str, Dict]:
        """
        응답 여부 결정 (M1 정책 적용)

        Args:
            utterance: 현재 발화
            context: 컨텍스트
            opp_result: opportunity 결과
            response_text: 생성될 응답 텍스트 (delay 계산용, 없으면 utterance 기반)
            has_prior_response: 이미 다른 사람이 응답했는지 (None이면 self.has_responded_in_thread 사용)

        Returns:
            (should_respond, delay, reason, metadata)
            metadata: {'utt_type': str, 'technical_analysis': Dict, ...}
        """
        if opp_result is None:
            opp_result = self.opportunity_detector.detect(utterance, context)

        # 1. Opportunity 체크
        if not opp_result.is_opportunity:
            self.consecutive_turns = 0  # 리셋
            return False, 0.0, "Not an opportunity", {}

        # 2. Turn constraint (burst suppression)
        if self.consecutive_turns >= self.config.max_consecutive_turns:
            # Cooldown 체크
            current_time = self._parse_time(utterance.get('ts') or utterance.get('timestamp'))
            if current_time and self.last_response_time:
                elapsed = (current_time - self.last_response_time).total_seconds()
                if elapsed < self.config.burst_cooldown_sec:
                    return False, 0.0, f"Burst suppression: {self.consecutive_turns} consecutive turns", {}

            self.consecutive_turns = 0

        # 3. Relevance gate
        relevance = self._compute_relevance(utterance, context)
        if relevance < self.config.relevance_threshold:
            return False, 0.0, f"Low relevance: {relevance:.3f} < {self.config.relevance_threshold}", {}

        # 4. Human-like 확률 결정 (ResponseProbabilityCalculator)
        prior_response = has_prior_response if has_prior_response is not None else self.has_responded_in_thread
        prob_result = self.response_calculator.calculate(
            opportunity=opp_result,
            has_prior_response=prior_response
        )

        # 5. 확률 기반 응답
        if prob_result.should_respond:
            self.consecutive_turns += 1
            self.has_responded_in_thread = True
            self.last_response_time = self._parse_time(
                utterance.get('ts') or utterance.get('timestamp')
            )

            # 6. TimingPolicy 기반 지연 계산
            text_for_timing = response_text or utterance.get('text', '')
            delay, utt_type, analysis = self._sample_delay_with_type(text_for_timing)

            metadata = {
                'utt_type': utt_type,
                'classification_reason': analysis.get('reason', ''),
                'technical_analysis': analysis.get('technical', {}),
                'probability': prob_result.probability,
                'opportunity_type': opp_result.opp_type,
            }

            reason = f"C1: {prob_result.reason}, rel={relevance:.2f}, type={utt_type}"
            return True, delay, reason, metadata
        else:
            return False, 0.0, f"C1: {prob_result.reason}", {}

    def _compute_relevance(self, utterance: Dict, context: List[Dict]) -> float:
        """relevance gate: 최근 k 발화와의 cosine similarity"""
        text = utterance.get('text', '')
        if not context:
            return 1.0  # 컨텍스트 없으면 통과

        # Update topic tracker
        self.topic_tracker.update(utterance)

        # Compute relevance
        return self.topic_tracker.compute_relevance(text)

    def _sample_delay_with_type(self, text: str) -> Tuple[float, str, Dict]:
        """
        TimingPolicy 기반 delay 샘플링

        Args:
            text: 응답 텍스트 (유형 분류에 사용)

        Returns:
            (delay, utt_type, analysis)
        """
        if self.config.use_timing_policy:
            # TimingPolicy 사용 (Bimodal: quick/normal/detailed)
            return self.timing_policy.sample_delay_with_analysis(text)
        else:
            # Legacy 방식
            delay = self._sample_delay_legacy()
            utt_type = classify_utterance_simple(text)
            return delay, utt_type, {'reason': 'legacy'}

    def _sample_delay_legacy(self, delay_type: str = 'opp') -> float:
        """Legacy delay 샘플링 (use_timing_policy=False 시)"""
        if delay_type == 'mention':
            params = self.config.delay_mention_params
        else:
            params = self.config.delay_opp_params

        if params and 'lognormal_fit' in params and params['lognormal_fit']:
            # Lognormal 분포에서 샘플링
            fit = params['lognormal_fit']
            mu, sigma = fit.get('mu', 3.0), fit.get('sigma', 1.0)
            delay = np.random.lognormal(mu, sigma)
            return min(delay, 300.0)  # 5분 상한

        elif params and 'samples' in params and params['samples']:
            # Empirical 샘플에서 랜덤 선택
            return random.choice(params['samples'])

        else:
            # Default
            return self.config.default_delay

    def _parse_time(self, ts) -> Optional[datetime]:
        if ts is None:
            return None
        if isinstance(ts, datetime):
            return ts
        if isinstance(ts, str):
            try:
                return datetime.fromisoformat(ts)
            except:
                return None
        return None

    def reset_consecutive_counter(self):
        """다른 사람이 응답하면 연속 카운터 리셋"""
        self.consecutive_turns = 0

    def reset_thread_state(self):
        """새 스레드 시작 시 상태 리셋"""
        self.consecutive_turns = 0
        self.has_responded_in_thread = False


class ConditionC2(ConditionC1):
    """
    +M1+M2: C1 + form control + chunking + drift stop

    Spec 6.3 C2:
    - C1의 모든 기능 포함 (TimingPolicy 기반 지연)
    - response form: target_words 유지
    - chunking: 긴 응답을 분절하여 여러 메시지로
    - drift stop: topic drift 시 남은 chunk 중단
    """

    def __init__(self, config: PolicyConfig):
        super().__init__(config)
        self.pending_chunks: List[str] = []
        self.current_topic_embedding = None

    def should_respond(
        self,
        utterance: Dict,
        context: List[Dict],
        opp_result: OpportunityResult = None,
        response_text: str = None
    ) -> Tuple[bool, float, str, Dict]:
        """
        C2 응답 결정 (C1 + form control)
        """
        # Drift stop: 새 발화가 topic drift면 pending chunks 중단
        if self.pending_chunks and self.config.drift_stop_enabled:
            if self._check_drift(utterance):
                dropped = len(self.pending_chunks)
                self.pending_chunks = []
                logger.debug(f"Drift stop: dropped {dropped} pending chunks")

        # C1 로직 호출 (TimingPolicy 기반 지연 포함)
        return super().should_respond(utterance, context, opp_result, response_text)

    def chunk_response(self, text: str) -> List[Tuple[str, float]]:
        """
        응답을 Human-like 청크로 분할 (M2)

        Human IRC 패턴 기반:
        - 67% 단일 메시지 (청킹 안함, 단 max_words 이하일 때만)
        - 21% 2개 메시지로 분할
        - 12% 3개 이상 메시지로 분할
        - 각 청크 최대 15단어 (human Q75)
        - 청크 간 지연: 1-3초

        핵심: 긴 응답은 반드시 분할 (정보 손실 방지)

        Returns:
            [(chunk_text, delay), ...]
        """
        if not self.config.chunking_enabled:
            return [(text, 0.0)]

        words = text.split()
        max_len = self.config.max_words  # 15 words (human Q75)

        # 짧은 응답은 분할 불필요
        if len(words) <= max_len:
            return [(text, 0.0)]

        # 문장 단위로 먼저 분리
        sentences = self._split_sentences(text)

        # 필요한 최소 청크 수 계산 (정보 손실 방지)
        min_chunks_needed = (len(words) + max_len - 1) // max_len  # ceil division

        # Human-like 분포에 따라 목표 청크 수 결정
        target_chunks = self._sample_chunk_count()

        # 최소 청크 수보다 작으면 최소값 사용 (정보 손실 방지)
        target_chunks = max(target_chunks, min_chunks_needed)

        # 목표 청크 수에 맞게 문장 그룹화
        chunks = self._group_sentences_into_chunks(sentences, target_chunks, max_len)

        # 지연 시간 추가 (첫 번째는 0, 나머지는 1-3초)
        result = []
        for i, chunk_text in enumerate(chunks):
            if i == 0:
                delay = 0.0
            else:
                delay = random.uniform(
                    self.config.chunk_delay_min,
                    self.config.chunk_delay_max
                )
            result.append((chunk_text, delay))

        return result

    def _sample_chunk_count(self) -> int:
        """Human-like 분포에서 목표 청크 수 샘플링"""
        dist = self.config.human_consecutive_dist
        r = random.random()

        cumulative = 0.0
        for count, prob in sorted(dist.items()):
            cumulative += prob
            if r < cumulative:
                return count

        return 3  # fallback

    def _split_sentences(self, text: str) -> List[str]:
        """문장 단위로 분리"""
        import re
        # 문장 끝 구분자로 분리 (마침표, 물음표, 느낌표)
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]

    def _group_sentences_into_chunks(
        self,
        sentences: List[str],
        target_chunks: int,
        max_words: int
    ) -> List[str]:
        """
        문장들을 목표 청크 수에 맞게 그룹화

        규칙:
        1. 각 청크는 max_words 이하
        2. 문장 경계 우선
        3. 긴 문장은 단어 단위로 분할
        """
        if not sentences:
            return []

        # 단일 메시지인 경우
        if target_chunks == 1:
            combined = ' '.join(sentences)
            # 길이 초과 시 강제 truncate
            words = combined.split()
            if len(words) > max_words:
                return [' '.join(words[:max_words])]
            return [combined]

        chunks = []
        current_chunk_words = []

        for sentence in sentences:
            sent_words = sentence.split()

            # 현재 청크에 추가 가능한지 확인
            if len(current_chunk_words) + len(sent_words) <= max_words:
                current_chunk_words.extend(sent_words)
            else:
                # 현재 청크 저장하고 새 청크 시작
                if current_chunk_words:
                    chunks.append(' '.join(current_chunk_words))
                    current_chunk_words = []

                # 문장이 max_words보다 긴 경우 자연스러운 분할점에서 분할
                if len(sent_words) > max_words:
                    split_chunks = self._split_long_sentence(sent_words, max_words)
                    chunks.extend(split_chunks)
                else:
                    current_chunk_words = sent_words

        # 남은 단어 처리
        if current_chunk_words:
            chunks.append(' '.join(current_chunk_words))

        # 목표 청크 수보다 적으면 그대로 반환
        # 목표 청크 수보다 많으면 병합 (rare case)
        if len(chunks) > target_chunks and target_chunks > 1:
            # 마지막 청크들 병합
            while len(chunks) > target_chunks:
                last = chunks.pop()
                chunks[-1] = chunks[-1] + ' ' + last

        return chunks if chunks else [' '.join(sentences)]

    def _split_long_sentence(self, words: List[str], max_words: int) -> List[str]:
        """
        긴 문장을 자연스러운 분할점에서 나누기

        분할 우선순위:
        1. 접속사 (and, or, but, then, so) 앞
        2. 전치사구 시작 (to, for, with, from, in, on, at) 앞
        3. 쉼표 뒤
        4. 불가피 시 max_words 경계
        """
        if len(words) <= max_words:
            return [' '.join(words)]

        # 자연스러운 분할점 찾기
        split_words = {'and', 'or', 'but', 'then', 'so', 'because', 'if', 'when'}
        prep_words = {'to', 'for', 'with', 'from', 'in', 'on', 'at', 'by', 'after', 'before'}

        chunks = []
        current = []

        for i, word in enumerate(words):
            word_lower = word.lower().rstrip('.,;:')

            # 분할점 체크 (현재 청크가 충분히 길 때만)
            should_split = False
            if len(current) >= max_words * 0.5:  # 최소 절반 이상 채워졌을 때
                if word_lower in split_words:
                    should_split = True
                elif word_lower in prep_words and len(current) >= max_words * 0.6:
                    should_split = True
                elif i > 0 and words[i-1].endswith(',') and len(current) >= max_words * 0.5:
                    should_split = True

            # 강제 분할 (max_words 도달)
            if len(current) >= max_words:
                should_split = True

            if should_split and current:
                chunks.append(' '.join(current))
                current = []

            current.append(word)

        # 남은 단어 처리
        if current:
            chunks.append(' '.join(current))

        return chunks

    def _check_drift(self, utterance: Dict) -> bool:
        """새 발화가 topic drift인지 체크"""
        if self.current_topic_embedding is None:
            return False

        text = utterance.get('text', '')
        new_relevance = self.topic_tracker.compute_relevance(text)

        return new_relevance < self.config.drift_threshold


def create_policy(
    condition: str,
    stats: Dict = None,
    config_overrides: Dict = None
) -> Tuple[PolicyConfig, object]:
    """
    조건에 맞는 정책 생성

    Args:
        condition: "C0", "C1", "C2"
        stats: 추출된 통계 (p_mention, p_spont, delays 등)
        config_overrides: 설정 오버라이드

    Returns:
        (PolicyConfig, policy_instance)
    """
    config = PolicyConfig(condition=condition)

    # 통계에서 파라미터 로드
    if stats:
        config.p_mention = stats.get('mention_response_prob', 0.9)
        config.p_spont = stats.get('opportunity_response_prob', 0.5)

        # Delay 분포
        if 'delay_mention' in stats:
            config.delay_mention_params = stats['delay_mention']
        if 'delay_opp' in stats:
            config.delay_opp_params = stats['delay_opp']

        # 길이 통계
        if 'human_length' in stats:
            length_stats = stats['human_length']
            config.target_words = int(length_stats.get('q75', 8))
            config.max_words = int(length_stats.get('q90', 15))

    # Config 오버라이드 적용
    if config_overrides:
        for key, value in config_overrides.items():
            if hasattr(config, key):
                setattr(config, key, value)

    # 조건별 정책 인스턴스 생성
    if condition == "C0":
        config.name = "Baseline"
        policy = ConditionC0(config)
    elif condition == "C1":
        config.name = "+M1 (Participation)"
        policy = ConditionC1(config)
    elif condition == "C2":
        config.name = "+M1+M2 (Full)"
        config.chunking_enabled = True
        config.drift_stop_enabled = True
        policy = ConditionC2(config)
    else:
        raise ValueError(f"Unknown condition: {condition}")

    logger.info(f"Created policy: {config.name} ({condition})")
    logger.info(f"  p_mention={config.p_mention:.3f}, p_spont={config.p_spont:.3f}")
    logger.info(f"  target_words={config.target_words}, chunking={config.chunking_enabled}")

    return config, policy
