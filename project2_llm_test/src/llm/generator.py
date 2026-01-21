"""
Response Generator (LLM + Chunking 통합)

핵심:
- LLM 응답을 human 길이에 맞춤
- 청킹 정책과 통합
- 길이 통제 검증 포함
- 유형별(quick/normal/detailed) 응답 스타일 지원
"""

import logging
import random
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

from .client import LLMClient, LLMConfig
from src.agent.timing_policy import classify_utterance_simple, analyze_technical_content

logger = logging.getLogger(__name__)


@dataclass
class GenerationResult:
    """응답 생성 결과"""
    chunks: List[str]  # 청크 리스트
    total_words: int
    chunk_word_counts: List[int]
    used_llm: bool
    was_truncated: bool
    raw_response: str


class ResponseGenerator:
    """
    응답 생성기

    역할:
    1. LLM으로 응답 생성 (길이 통제)
    2. 청킹 정책 적용
    3. 결과 구조화
    """

    def __init__(
        self,
        llm_config: LLMConfig = None,
        length_stats: Dict = None,
        use_llm: bool = True
    ):
        """
        Args:
            llm_config: LLM 설정
            length_stats: human 길이 통계 (target 설정에 사용)
            use_llm: LLM 사용 여부
        """
        self.use_llm = use_llm

        # Human 길이 통계에서 target 설정
        if length_stats:
            utterance_lengths = length_stats.get('utterance_lengths', {})
            target_words = int(utterance_lengths.get('q75', 8))
            max_words = int(utterance_lengths.get('q90', 12))
        else:
            target_words = 8
            max_words = 12

        # LLM 설정 업데이트
        if llm_config:
            llm_config.target_words = target_words
            llm_config.max_words = max_words
            # max_tokens도 조정 (단어당 약 1.5 토큰)
            llm_config.max_tokens = min(30, max_words * 2)
        else:
            llm_config = LLMConfig(
                target_words=target_words,
                max_words=max_words,
                max_tokens=min(30, max_words * 2)
            )

        self.llm_config = llm_config
        self.llm_client = LLMClient(llm_config) if use_llm else None

        logger.info(f"ResponseGenerator initialized:")
        logger.info(f"  use_llm={use_llm}")
        logger.info(f"  target_words={target_words}")
        logger.info(f"  max_words={max_words}")

    def generate(
        self,
        utterance: Dict,
        thread_context: List[Dict],
        chunking_enabled: bool = False,
        max_chunk_len: int = None
    ) -> GenerationResult:
        """
        응답 생성

        Args:
            utterance: 응답할 발화
            thread_context: 스레드 컨텍스트
            chunking_enabled: 청킹 활성화 여부
            max_chunk_len: 청크당 최대 단어 수

        Returns:
            GenerationResult
        """
        prompt = utterance.get('text', '')
        speaker = utterance.get('speaker', 'user')

        # 컨텍스트 준비
        context = []
        for ctx in thread_context[-5:]:
            context.append({
                'text': ctx.get('text', ''),
                'is_agent': ctx.get('speaker') == 'agent'
            })

        # LLM 응답 생성
        if self.use_llm and self.llm_client:
            result = self.llm_client.generate(
                prompt=f"{speaker}: {prompt}",
                context=context
            )
        else:
            result = self._dummy_response(prompt)

        raw_response = result['raw_response']
        response_text = result['text']
        used_llm = result['used_llm']
        was_truncated = result['was_truncated']

        # 청킹 적용
        if chunking_enabled and max_chunk_len:
            chunks = self._chunk_response(response_text, max_chunk_len)
        else:
            chunks = [response_text]

        # 결과 구성
        chunk_word_counts = [len(c.split()) for c in chunks]
        total_words = sum(chunk_word_counts)

        return GenerationResult(
            chunks=chunks,
            total_words=total_words,
            chunk_word_counts=chunk_word_counts,
            used_llm=used_llm,
            was_truncated=was_truncated,
            raw_response=raw_response
        )

    def _chunk_response(self, text: str, max_chunk_len: int) -> List[str]:
        """
        응답을 청크로 분할

        규칙:
        1. 문장 경계에서 분할 우선
        2. max_chunk_len 초과 시 강제 분할
        """
        if not text:
            return []

        words = text.split()
        if len(words) <= max_chunk_len:
            return [text]

        chunks = []
        current_chunk = []
        current_len = 0

        for word in words:
            current_chunk.append(word)
            current_len += 1

            # 문장 끝이거나 길이 초과
            is_sentence_end = word.endswith(('.', '!', '?'))
            is_over_limit = current_len >= max_chunk_len

            if is_sentence_end or is_over_limit:
                chunk_text = ' '.join(current_chunk)
                chunks.append(chunk_text)
                current_chunk = []
                current_len = 0

        # 남은 단어
        if current_chunk:
            chunks.append(' '.join(current_chunk))

        return chunks

    def _dummy_response(self, prompt: str) -> Dict:
        """
        Dummy 응답 (LLM 미사용 시) - 유형별 응답 스타일 적용

        유형 분류:
        - Quick: 단답 (acknowledgment)
        - Normal: 일반적인 짧은 응답
        - Detailed: 기술적 내용 포함
        """
        # 프롬프트의 유형 분류
        utt_type = classify_utterance_simple(prompt)
        prompt_lower = prompt.lower()

        # 유형별 응답 템플릿
        if utt_type == 'quick':
            # Quick: 단답형 응답
            responses = [
                "Yes.",
                "No.",
                "Okay.",
                "Got it.",
                "Thanks.",
                "Sure.",
                "Right.",
                "I see.",
            ]
        elif utt_type == 'detailed':
            # Detailed: 기술적 응답 (더 긴 설명)
            if any(w in prompt_lower for w in ['apt', 'install', 'package']):
                responses = [
                    "Try running sudo apt update first, then retry.",
                    "Check if the package exists with apt search.",
                    "You might need to add the repository first.",
                    "Run apt update && apt upgrade to refresh.",
                ]
            elif any(w in prompt_lower for w in ['error', 'fail', 'denied', 'permission']):
                responses = [
                    "Check the logs in /var/log for details.",
                    "Try running with sudo for permissions.",
                    "Look at journalctl -xe for the error.",
                    "Check file permissions with ls -la.",
                ]
            elif any(w in prompt_lower for w in ['network', 'connect', 'ssh', 'ip']):
                responses = [
                    "Check your network with ip addr or ifconfig.",
                    "Try ping to test connectivity.",
                    "Verify firewall rules with ufw status.",
                    "Check /etc/resolv.conf for DNS settings.",
                ]
            else:
                responses = [
                    "Check the system logs for more info.",
                    "Try the command with verbose flag.",
                    "Look at the documentation for details.",
                    "Restart the service and check status.",
                ]
        else:
            # Normal: 일반적인 짧은 응답
            if '?' in prompt or any(w in prompt_lower for w in ['how', 'what', 'why', 'help']):
                responses = [
                    "Try restarting first.",
                    "Check the settings.",
                    "That should work.",
                    "Give that a try.",
                    "What error do you see?",
                ]
            else:
                responses = [
                    "That makes sense.",
                    "Good idea.",
                    "That should help.",
                    "Sounds good.",
                    "Let me know how it goes.",
                ]

        text = random.choice(responses)

        return {
            'text': text,
            'word_count': len(text.split()),
            'was_truncated': False,
            'used_llm': False,
            'raw_response': text,
            'utt_type': utt_type,
        }

    def get_llm_stats(self) -> Dict:
        """LLM 사용 통계"""
        if self.llm_client:
            return self.llm_client.get_stats()
        return {
            'total_calls': 0,
            'fallback_count': 0,
            'truncation_count': 0,
            'llm_success_rate': 0.0
        }

    def validate_length_control(self, human_q75: float) -> Dict:
        """
        길이 통제가 적용되는지 검증

        Returns:
            {
                'is_valid': bool,
                'target_words': int,
                'human_q75': float,
                'max_words': int,
                'message': str
            }
        """
        target = self.llm_config.target_words
        max_w = self.llm_config.max_words

        # target이 human q75 근처인지 (±50%)
        is_near_human = 0.5 * human_q75 <= target <= 1.5 * human_q75

        # max_words가 합리적인지
        is_max_reasonable = max_w <= 2 * human_q75

        is_valid = is_near_human and is_max_reasonable

        if is_valid:
            message = f"PASS: Length control configured (target={target}, human_q75={human_q75:.1f})"
        else:
            message = f"FAIL: Length control mismatch (target={target}, human_q75={human_q75:.1f})"

        return {
            'is_valid': is_valid,
            'target_words': target,
            'human_q75': human_q75,
            'max_words': max_w,
            'message': message
        }
