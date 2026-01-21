"""
Opportunity Detector Module

핵심: P(respond|no mention)의 분모를 '모든 발화'가 아니라
'응답이 기대되는 상황(opportunity)'으로 제한합니다.

Opportunity 유형:
1. QUESTION: 질문으로 분류된 발화 (?, how, why, what, can I, help 등)
2. HELP_REQUEST: 도움 요청/문제 보고 패턴
3. INITIATING_WINDOW: thread 시작 질문 이후 일정 시간 내 발화
"""

import re
import logging
from typing import Dict, List, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class OpportunityType(Enum):
    NONE = "none"
    QUESTION = "question"
    HELP_REQUEST = "help_request"
    INITIATING_WINDOW = "initiating_window"
    DIRECT_MENTION = "direct_mention"


@dataclass
class OpportunityResult:
    """Spec 4.2: is_opportunity 라벨링 결과"""
    is_opportunity: bool
    opportunity_type: OpportunityType
    confidence: float
    reason: str
    is_question: bool = False
    is_help_request: bool = False
    opp_type: str = "none"  # "question"|"help_request"|"init_window"|"direct_mention"|"none"


class OpportunityDetector:
    """
    각 발화에 대해 '응답 기회(opportunity)' 여부를 판단합니다.

    중요: 멘션이 없더라도 '응답이 기대되는 상황'에서만 opportunity를 인정합니다.
    이렇게 해야 P(respond|no mention AND opportunity)의 분모가 적절해집니다.
    """

    # 질문 패턴
    QUESTION_PATTERNS = [
        r'\?$',  # 물음표로 끝남
        r'\?[.!]*$',
        r'^(how|why|what|where|when|who|which|can\s+i|could\s+i|is\s+there|are\s+there|does|do\s+i|should\s+i|would)',
        r'(any\s+idea|anyone\s+know|somebody\s+help|can\s+someone|could\s+someone)',
        r'(help\s+me|help\s+with|need\s+help)',
    ]

    # 도움 요청 / 문제 보고 패턴
    HELP_PATTERNS = [
        r'(i\s+have\s+a\s+problem|i\'m\s+having\s+(an?\s+)?issue|i\'m\s+getting\s+(an?\s+)?error)',
        r'(doesn\'t\s+work|not\s+working|won\'t\s+work|failed|failing)',
        r'(can\'t|cannot|unable\s+to)\s+\w+',
        r'(error|issue|problem|bug|broken|stuck)',
        r'(please\s+help|need\s+assistance|looking\s+for\s+help)',
        r'(any\s+suggestions?|any\s+advice|recommendations?)',
    ]

    # Ubuntu/기술 관련 키워드 (문제 상황에서 자주 등장)
    TECH_PROBLEM_PATTERNS = [
        r'(apt|apt-get|dpkg|snap)\s+.*(error|fail|broken)',
        r'(permission\s+denied|access\s+denied|sudo)',
        r'(segfault|segmentation\s+fault|core\s+dump)',
        r'(dependency|unmet\s+dependencies)',
        r'(boot|grub|kernel).*(fail|error|panic)',
        r'(install|upgrade|update).*(fail|error|stuck)',
    ]

    def __init__(
        self,
        initiating_window_seconds: float = 120.0,  # 시작 질문 후 2분 내
        min_question_confidence: float = 0.5,
        agent_name: str = "agent"
    ):
        self.initiating_window_seconds = initiating_window_seconds
        self.min_question_confidence = min_question_confidence
        self.agent_name = agent_name

        # 컴파일된 패턴
        self._question_patterns = [re.compile(p, re.IGNORECASE) for p in self.QUESTION_PATTERNS]
        self._help_patterns = [re.compile(p, re.IGNORECASE) for p in self.HELP_PATTERNS]
        self._tech_patterns = [re.compile(p, re.IGNORECASE) for p in self.TECH_PROBLEM_PATTERNS]

    def detect(
        self,
        utterance: Dict,
        thread_context: List[Dict],
        thread_start_time: datetime = None
    ) -> OpportunityResult:
        """
        단일 발화에 대해 opportunity 여부 판단

        Args:
            utterance: 현재 발화 {"speaker", "text", "timestamp", "mentions"}
            thread_context: 이전 발화들
            thread_start_time: 스레드 시작 시간 (initiating window 계산용)

        Returns:
            OpportunityResult
        """
        text = utterance.get('text', '')
        mentions = utterance.get('mentions', [])
        speaker = utterance.get('speaker', '')

        # 질문/도움 요청 점수 미리 계산
        question_score = self._check_question(text)
        help_score = self._check_help_request(text)
        is_question = question_score >= self.min_question_confidence
        is_help_request = help_score >= self.min_question_confidence

        # 0. 에이전트 자신의 발화는 opportunity 아님
        if speaker == self.agent_name:
            return OpportunityResult(
                is_opportunity=False,
                opportunity_type=OpportunityType.NONE,
                confidence=1.0,
                reason="Agent's own utterance",
                is_question=is_question,
                is_help_request=is_help_request,
                opp_type="none"
            )

        # 1. 직접 멘션된 경우 - 가장 확실한 opportunity
        if self.agent_name in mentions:
            return OpportunityResult(
                is_opportunity=True,
                opportunity_type=OpportunityType.DIRECT_MENTION,
                confidence=1.0,
                reason=f"Directly mentioned: @{self.agent_name}",
                is_question=is_question,
                is_help_request=is_help_request,
                opp_type="direct_mention"
            )

        # 2. 질문 패턴 검사
        if is_question:
            return OpportunityResult(
                is_opportunity=True,
                opportunity_type=OpportunityType.QUESTION,
                confidence=question_score,
                reason="Question pattern detected",
                is_question=True,
                is_help_request=is_help_request,
                opp_type="question"
            )

        # 3. 도움 요청 / 문제 보고 패턴
        if is_help_request:
            return OpportunityResult(
                is_opportunity=True,
                opportunity_type=OpportunityType.HELP_REQUEST,
                confidence=help_score,
                reason="Help request or problem report detected",
                is_question=is_question,
                is_help_request=True,
                opp_type="help_request"
            )

        # 4. Initiating window 체크 (스레드 시작 질문 이후 일정 시간 내)
        if thread_start_time and thread_context:
            in_window, window_reason = self._check_initiating_window(
                utterance, thread_context, thread_start_time
            )
            if in_window:
                return OpportunityResult(
                    is_opportunity=True,
                    opportunity_type=OpportunityType.INITIATING_WINDOW,
                    confidence=0.6,
                    reason=window_reason,
                    is_question=is_question,
                    is_help_request=is_help_request,
                    opp_type="init_window"
                )

        # 5. Opportunity 아님
        return OpportunityResult(
            is_opportunity=False,
            opportunity_type=OpportunityType.NONE,
            confidence=1.0,
            reason="No opportunity pattern matched",
            is_question=is_question,
            is_help_request=is_help_request,
            opp_type="none"
        )

    def detect_thread(self, thread: Dict) -> List[Tuple[Dict, OpportunityResult]]:
        """
        전체 스레드에 대해 opportunity 라벨링

        Returns:
            [(utterance, OpportunityResult), ...]
        """
        utterances = thread.get('utterances', [])
        if not utterances:
            return []

        # 스레드 시작 시간
        first_ts = utterances[0].get('timestamp') or utterances[0].get('ts')
        if isinstance(first_ts, str):
            try:
                thread_start_time = datetime.fromisoformat(first_ts)
            except:
                thread_start_time = None
        else:
            thread_start_time = first_ts

        # 스레드 내 모든 speaker 목록 추출 (mention 검증용)
        speaker_list = list(set(u.get('speaker', '') for u in utterances if u.get('speaker')))

        results = []
        for i, utt in enumerate(utterances):
            context = utterances[:i]  # 이전 발화들

            # mentions 필드가 없거나 비어있으면 재추출
            if not utt.get('mentions'):
                text = utt.get('text', '')
                speaker = utt.get('speaker', '')
                utt = dict(utt)  # 복사
                utt['mentions'] = self._extract_mentions_with_context(text, speaker, speaker_list)

            result = self.detect(utt, context, thread_start_time)
            results.append((utt, result))

        return results

    def _extract_mentions_with_context(self, text: str, speaker: str, speaker_list: List[str]) -> List[str]:
        """
        텍스트에서 mentions 추출 (speaker_list 기반 검증, 자기 멘션 제외)

        Patterns:
        1. @nick - always captured
        2. nick: or nick, or nick; at line start - validated against speaker_list
        3. \\bnick\\b word boundary match - validated against speaker_list
        """
        mentions = []
        speaker_set = set(s.lower() for s in speaker_list if s)

        # Pattern 1: @username - always valid
        at_mentions = re.findall(r'@(\w+)', text)
        mentions.extend(at_mentions)

        # Pattern 2: nick: or nick, or nick; at line start
        line_start_match = re.match(r'^(\w+)[:,;]\s*', text)
        if line_start_match:
            nick = line_start_match.group(1)
            if nick.lower() in speaker_set:
                mentions.append(nick)

        # Pattern 3: word boundary match for known speakers
        words = re.findall(r'\b(\w+)\b', text)
        for word in words:
            if word.lower() in speaker_set and word not in mentions:
                mentions.append(word)

        # 자기 멘션 제외, 중복 제거
        return list(set(m for m in mentions if m.lower() != speaker.lower()))

    def _check_question(self, text: str) -> float:
        """질문 패턴 매칭 점수 (0~1)"""
        if not text:
            return 0.0

        score = 0.0
        matches = 0

        for pattern in self._question_patterns:
            if pattern.search(text):
                matches += 1

        if matches > 0:
            # 물음표가 있으면 높은 점수
            if '?' in text:
                score = 0.9
            else:
                score = min(0.8, 0.4 + 0.2 * matches)

        return score

    def _check_help_request(self, text: str) -> float:
        """도움 요청/문제 보고 패턴 점수"""
        if not text:
            return 0.0

        help_matches = sum(1 for p in self._help_patterns if p.search(text))
        tech_matches = sum(1 for p in self._tech_patterns if p.search(text))

        if help_matches > 0 or tech_matches > 0:
            # 기술 문제 + 도움 요청이 동시에 있으면 높은 점수
            if help_matches > 0 and tech_matches > 0:
                return 0.85
            elif tech_matches >= 2:
                return 0.7
            elif help_matches > 0:
                return 0.65
            else:
                return 0.55

        return 0.0

    def _check_initiating_window(
        self,
        utterance: Dict,
        context: List[Dict],
        thread_start_time: datetime
    ) -> Tuple[bool, str]:
        """
        스레드 시작 질문 이후 일정 시간 내에 있는지 확인
        """
        # 첫 번째 발화가 질문인지 확인
        if not context:
            return False, ""

        first_utt = context[0]
        first_question_score = self._check_question(first_utt.get('text', ''))
        first_help_score = self._check_help_request(first_utt.get('text', ''))

        is_initiating_question = (first_question_score >= 0.5 or first_help_score >= 0.5)

        if not is_initiating_question:
            return False, ""

        # 현재 발화 시간 확인
        current_ts = utterance.get('timestamp')
        if isinstance(current_ts, str):
            try:
                current_time = datetime.fromisoformat(current_ts)
            except:
                return False, ""
        else:
            current_time = current_ts

        if current_time is None or thread_start_time is None:
            return False, ""

        # 시간 차이 계산
        time_diff = (current_time - thread_start_time).total_seconds()

        if time_diff <= self.initiating_window_seconds:
            return True, f"Within {self.initiating_window_seconds}s of initiating question"

        return False, ""


def compute_opportunity_stats(
    threads: List[Dict],
    agent_name: str = "agent"
) -> Dict:
    """
    스레드들에 대해 opportunity 통계 계산

    Returns:
        {
            'total_utterances': int,
            'opportunities': int,
            'opportunity_rate': float,
            'by_type': {type: count},
            'mention_opportunities': int,
            'non_mention_opportunities': int,
        }
    """
    detector = OpportunityDetector(agent_name=agent_name)

    stats = {
        'total_utterances': 0,
        'opportunities': 0,
        'by_type': {t.value: 0 for t in OpportunityType},
        'mention_opportunities': 0,
        'non_mention_opportunities': 0,
    }

    for thread in threads:
        results = detector.detect_thread(thread)

        for utt, opp_result in results:
            # 에이전트 발화 제외
            if utt.get('speaker') == agent_name:
                continue

            stats['total_utterances'] += 1

            if opp_result.is_opportunity:
                stats['opportunities'] += 1
                stats['by_type'][opp_result.opportunity_type.value] += 1

                if opp_result.opportunity_type == OpportunityType.DIRECT_MENTION:
                    stats['mention_opportunities'] += 1
                else:
                    stats['non_mention_opportunities'] += 1

    # 비율 계산
    if stats['total_utterances'] > 0:
        stats['opportunity_rate'] = stats['opportunities'] / stats['total_utterances']
    else:
        stats['opportunity_rate'] = 0.0

    logger.info(f"Opportunity stats: {stats['opportunities']}/{stats['total_utterances']} "
                f"= {stats['opportunity_rate']:.2%}")
    logger.info(f"  Mention opportunities: {stats['mention_opportunities']}")
    logger.info(f"  Non-mention opportunities: {stats['non_mention_opportunities']}")

    return stats
