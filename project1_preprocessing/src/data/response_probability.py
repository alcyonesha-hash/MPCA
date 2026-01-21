"""
Response Probability Module

Human helper 응답 패턴 기반 선택적 응답 확률 계산.

IRC 데이터 분석 결과 (Ubuntu #ubuntu 2024.01):
- direct_mention: 100% (직접 호출 시 무조건 응답)
- question (첫 응답): 65.4%
- question (이미 응답 있음): 92.0% → 추가 응답 불필요하므로 낮춤
- help_request (첫 응답): 55.8%
- help_request (이미 응답 있음): 89.0% → 추가 응답 불필요하므로 낮춤

설계 원칙:
1. 직접 멘션 → 무조건 응답
2. 아무도 안 답했으면 → human 수준으로 응답 (65%, 56%)
3. 이미 누가 답했으면 → 크게 낮춤 (필요시에만 추가 응답)
"""

import random
import logging
from typing import Dict, List, Optional
from dataclasses import dataclass
from enum import Enum

from .opportunity_detector import OpportunityResult, OpportunityType

logger = logging.getLogger(__name__)


class ResponseDecision(Enum):
    RESPOND = "respond"
    SKIP = "skip"


@dataclass
class ResponseProbabilityResult:
    """응답 확률 계산 결과"""
    should_respond: bool
    decision: ResponseDecision
    probability: float
    reason: str
    opportunity_type: str


class ResponseProbabilityCalculator:
    """
    Opportunity에 대해 실제로 응답할지 확률적으로 결정.

    Human helper 패턴을 모방하여 모든 opportunity에 응답하지 않고
    선택적으로 응답합니다.
    """

    # Human helper 응답률 기반 기본 확률 (IRC 데이터 분석 결과)
    DEFAULT_PROBABILITIES = {
        # (opportunity_type, has_prior_response) -> probability
        ('direct_mention', False): 1.0,   # 직접 호출 → 무조건
        ('direct_mention', True): 1.0,    # 직접 호출 → 무조건
        ('question', False): 0.65,        # 첫 응답: 65.4%
        ('question', True): 0.15,         # 이미 답변 있음: 낮춤
        ('help_request', False): 0.56,    # 첫 응답: 55.8%
        ('help_request', True): 0.10,     # 이미 답변 있음: 낮춤
        ('init_window', False): 0.40,     # initiating window: 중간
        ('init_window', True): 0.10,      # 이미 답변 있음: 낮춤
        ('none', False): 0.0,             # opportunity 아님
        ('none', True): 0.0,
    }

    def __init__(
        self,
        probabilities: Optional[Dict] = None,
        seed: Optional[int] = None
    ):
        """
        Args:
            probabilities: 커스텀 확률 설정 (DEFAULT_PROBABILITIES 형식)
            seed: 랜덤 시드 (재현성용)
        """
        self.probabilities = probabilities or self.DEFAULT_PROBABILITIES.copy()

        if seed is not None:
            random.seed(seed)

    def calculate(
        self,
        opportunity: OpportunityResult,
        has_prior_response: bool = False,
        context: Optional[Dict] = None
    ) -> ResponseProbabilityResult:
        """
        응답 확률 계산 및 결정

        Args:
            opportunity: OpportunityDetector의 결과
            has_prior_response: 이미 다른 사람이 응답했는지
            context: 추가 컨텍스트 (확장용)

        Returns:
            ResponseProbabilityResult
        """
        opp_type = opportunity.opp_type

        # 확률 조회
        key = (opp_type, has_prior_response)
        probability = self.probabilities.get(key, 0.0)

        # 확률적 결정
        roll = random.random()
        should_respond = roll < probability

        # 이유 생성
        if opp_type == 'direct_mention':
            reason = "Directly mentioned - must respond"
        elif opp_type == 'none':
            reason = "Not an opportunity"
        elif has_prior_response:
            if should_respond:
                reason = f"Adding to existing discussion (p={probability:.0%}, roll={roll:.2f})"
            else:
                reason = f"Already answered by others (p={probability:.0%}, roll={roll:.2f})"
        else:
            if should_respond:
                reason = f"First responder (p={probability:.0%}, roll={roll:.2f})"
            else:
                reason = f"Skipping opportunity (p={probability:.0%}, roll={roll:.2f})"

        decision = ResponseDecision.RESPOND if should_respond else ResponseDecision.SKIP

        return ResponseProbabilityResult(
            should_respond=should_respond,
            decision=decision,
            probability=probability,
            reason=reason,
            opportunity_type=opp_type
        )

    def calculate_thread(
        self,
        thread: Dict,
        opportunity_results: List[tuple],  # [(utterance, OpportunityResult), ...]
        agent_name: str = "agent"
    ) -> List[tuple]:
        """
        전체 스레드에 대해 응답 확률 계산

        Args:
            thread: thread dict
            opportunity_results: OpportunityDetector.detect_thread()의 결과
            agent_name: 에이전트 이름 (자기 발화 제외용)

        Returns:
            [(utterance, OpportunityResult, ResponseProbabilityResult), ...]
        """
        results = []
        has_any_response = False  # 스레드 내 응답 여부 추적

        for utt, opp_result in opportunity_results:
            speaker = utt.get('speaker', '')

            # 에이전트 자신의 발화는 응답 대상 아님
            if speaker == agent_name:
                prob_result = ResponseProbabilityResult(
                    should_respond=False,
                    decision=ResponseDecision.SKIP,
                    probability=0.0,
                    reason="Agent's own utterance",
                    opportunity_type=opp_result.opp_type
                )
            else:
                # has_prior_response: 이 opportunity 이전에 다른 사람이 응답했는지
                prob_result = self.calculate(
                    opportunity=opp_result,
                    has_prior_response=has_any_response
                )

                # 응답 여부 업데이트 (opportunity에 대해 다른 사람이 발화하면 응답으로 간주)
                if opp_result.is_opportunity:
                    # 다음 발화가 다른 사람이면 응답으로 기록
                    has_any_response = True

            results.append((utt, opp_result, prob_result))

        return results


def compute_response_stats(
    threads: List[Dict],
    agent_name: str = "agent",
    seed: Optional[int] = 42
) -> Dict:
    """
    스레드들에 대해 응답 확률 통계 계산

    Returns:
        {
            'total_opportunities': int,
            'would_respond': int,
            'would_skip': int,
            'response_rate': float,
            'by_type': {type: {'respond': int, 'skip': int}},
        }
    """
    from .opportunity_detector import OpportunityDetector

    detector = OpportunityDetector(agent_name=agent_name)
    calculator = ResponseProbabilityCalculator(seed=seed)

    stats = {
        'total_opportunities': 0,
        'would_respond': 0,
        'would_skip': 0,
        'by_type': {},
    }

    for thread in threads:
        opp_results = detector.detect_thread(thread)
        prob_results = calculator.calculate_thread(thread, opp_results, agent_name)

        for utt, opp_result, prob_result in prob_results:
            if utt.get('speaker') == agent_name:
                continue

            if opp_result.is_opportunity:
                stats['total_opportunities'] += 1
                opp_type = opp_result.opp_type

                if opp_type not in stats['by_type']:
                    stats['by_type'][opp_type] = {'respond': 0, 'skip': 0}

                if prob_result.should_respond:
                    stats['would_respond'] += 1
                    stats['by_type'][opp_type]['respond'] += 1
                else:
                    stats['would_skip'] += 1
                    stats['by_type'][opp_type]['skip'] += 1

    # 비율 계산
    if stats['total_opportunities'] > 0:
        stats['response_rate'] = stats['would_respond'] / stats['total_opportunities']
    else:
        stats['response_rate'] = 0.0

    logger.info(f"Response stats: {stats['would_respond']}/{stats['total_opportunities']} "
                f"= {stats['response_rate']:.1%} would respond")

    return stats
