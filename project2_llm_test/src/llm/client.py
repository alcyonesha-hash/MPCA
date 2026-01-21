"""
LLM Client with Strict Length Control

핵심 수정:
- 응답 길이를 human 분포 (q75) 근처로 강제 통제
- max_tokens 제한
- system prompt에서 길이 명시
- 길이 초과 시 자동 절단
"""

import requests
import logging
from typing import Dict, List, Optional
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class LLMConfig:
    """LLM 설정 (길이 통제 포함)"""
    endpoint: str = "http://localhost:1234/v1/chat/completions"
    model: str = "openai/gpt-oss-120b"

    # 길이 통제 (핵심!)
    target_words: int = 8  # human q75 근처
    max_words: int = 15    # 절대 상한
    max_tokens: int = 30   # API 토큰 제한

    # 스타일
    style: str = "irc"  # irc, formal, casual

    # 타임아웃
    timeout: int = 30

    # 재시도
    max_retries: int = 2


class LLMClient:
    """
    LLM API 클라이언트 (길이 통제 포함)

    핵심: topical fit이 길이로 부풀려지지 않도록
    응답 길이를 human 분포에 맞춤
    """

    # 스타일별 시스템 프롬프트
    STYLE_PROMPTS = {
        'irc': (
            "You are a helpful user in an Ubuntu IRC channel. "
            "Reply in IRC style: brief, direct, no greetings. "
            "CRITICAL: Reply in EXACTLY {target_words} words or less. "
            "One short sentence only. No explanations, just the answer."
        ),
        'formal': (
            "You are a technical support agent. "
            "Be concise and professional. "
            "Reply in {target_words} words or less."
        ),
        'casual': (
            "Reply briefly and helpfully. "
            "{target_words} words max."
        )
    }

    def __init__(self, config: LLMConfig = None):
        self.config = config or LLMConfig()
        self._call_count = 0
        self._fallback_count = 0
        self._truncation_count = 0

    def generate(
        self,
        prompt: str,
        context: List[Dict] = None,
        target_words: int = None
    ) -> Dict:
        """
        응답 생성 (길이 통제 적용)

        Args:
            prompt: 사용자 입력
            context: 대화 컨텍스트
            target_words: 목표 단어 수 (없으면 config 사용)

        Returns:
            {
                'text': str,
                'word_count': int,
                'was_truncated': bool,
                'used_llm': bool,
                'raw_response': str (truncation 전)
            }
        """
        target = target_words or self.config.target_words
        self._call_count += 1

        # 시스템 프롬프트 구성
        system_prompt = self.STYLE_PROMPTS.get(
            self.config.style, self.STYLE_PROMPTS['irc']
        ).format(target_words=target)

        # 메시지 구성
        messages = [{"role": "system", "content": system_prompt}]

        if context:
            for ctx in context[-3:]:  # 최근 3개만
                role = "assistant" if ctx.get('is_agent') else "user"
                messages.append({
                    "role": role,
                    "content": ctx.get('text', '')[:200]  # 컨텍스트도 제한
                })

        messages.append({"role": "user", "content": prompt[:300]})

        # API 호출
        try:
            response = requests.post(
                self.config.endpoint,
                json={
                    "model": self.config.model,
                    "messages": messages,
                    "max_tokens": self.config.max_tokens,
                    "temperature": 0.7,
                    "stream": False
                },
                timeout=self.config.timeout
            )

            if response.status_code == 200:
                result = response.json()
                message = result['choices'][0]['message']
                # content가 비어있으면 reasoning 필드 사용 (일부 모델의 특성)
                raw_text = message.get('content', '') or message.get('reasoning', '')
                raw_text = raw_text.strip()

                # Reasoning 텍스트 제거 (모델이 content에 reasoning을 섞어 출력하는 경우)
                raw_text = self._clean_reasoning(raw_text)

                # 길이 통제 적용
                processed = self._enforce_length(raw_text, target)

                return {
                    'text': processed['text'],
                    'word_count': processed['word_count'],
                    'was_truncated': processed['was_truncated'],
                    'used_llm': True,
                    'raw_response': raw_text
                }
            else:
                logger.warning(f"LLM API error: {response.status_code}")
                self._fallback_count += 1
                return self._fallback_response(prompt)

        except Exception as e:
            logger.warning(f"LLM request failed: {e}")
            self._fallback_count += 1
            return self._fallback_response(prompt)

    def _clean_reasoning(self, text: str) -> str:
        """
        Reasoning 텍스트 제거

        일부 모델은 content에 reasoning을 섞어서 출력함:
        - "Need 8 words max. 'sudo apt install python3'."
        - "Maybe 'run apt update first'."

        따옴표 안의 실제 응답만 추출
        """
        import re

        # 패턴 1: 따옴표로 감싸진 응답 추출 ("..." 또는 '...')
        quoted = re.findall(r'["\']([^"\']{5,})["\']', text)
        if quoted:
            # 가장 긴 따옴표 내용 선택 (실제 응답일 가능성 높음)
            return max(quoted, key=len)

        # 패턴 2: "Need X words" 같은 reasoning prefix 제거
        reasoning_patterns = [
            r'^Need \d+ words[^.]*\.\s*',
            r'^Maybe\s+',
            r'^I\'ll\s+',
            r'^Let me\s+',
            r'^Here\'s\s+',
        ]
        cleaned = text
        for pattern in reasoning_patterns:
            cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE)

        return cleaned.strip()

    def _enforce_length(self, text: str, target_words: int) -> Dict:
        """
        길이 강제 통제

        Returns:
            {'text': str, 'word_count': int, 'was_truncated': bool}
        """
        # 정리
        text = text.strip()
        # 마크다운 제거
        text = text.replace('**', '').replace('*', '')
        # 줄바꿈을 공백으로
        text = ' '.join(text.split())

        words = text.split()
        original_count = len(words)

        # 최대 길이 초과 시 절단
        max_words = self.config.max_words
        if len(words) > max_words:
            # 문장 경계에서 자르기 시도
            truncated = self._truncate_at_sentence(words, max_words)
            self._truncation_count += 1
            was_truncated = True
        else:
            truncated = ' '.join(words)
            was_truncated = False

        final_words = truncated.split()

        return {
            'text': truncated,
            'word_count': len(final_words),
            'was_truncated': was_truncated
        }

    def _truncate_at_sentence(self, words: List[str], max_words: int) -> str:
        """문장 경계에서 절단"""
        text = ' '.join(words[:max_words])

        # 마지막 문장 종결 부호 찾기
        for i in range(len(text) - 1, max(0, len(text) - 50), -1):
            if text[i] in '.!?':
                return text[:i+1]

        # 못 찾으면 그냥 자르고 마침표 추가
        truncated = ' '.join(words[:max_words])
        if not truncated.endswith(('.', '!', '?')):
            truncated = truncated.rstrip('.,;:') + '.'

        return truncated

    def _fallback_response(self, prompt: str) -> Dict:
        """Fallback 응답 (LLM 실패 시)"""
        # 간단한 템플릿 기반 응답
        prompt_lower = prompt.lower()

        if '?' in prompt or 'how' in prompt_lower or 'what' in prompt_lower:
            responses = [
                "Try checking the documentation.",
                "Run apt update first.",
                "Check your system logs.",
                "That depends on your setup.",
            ]
        else:
            responses = [
                "I see.",
                "Got it.",
                "Makes sense.",
                "Thanks for the info.",
            ]

        import random
        text = random.choice(responses)

        return {
            'text': text,
            'word_count': len(text.split()),
            'was_truncated': False,
            'used_llm': False,
            'raw_response': text
        }

    def get_stats(self) -> Dict:
        """사용 통계"""
        return {
            'total_calls': self._call_count,
            'fallback_count': self._fallback_count,
            'truncation_count': self._truncation_count,
            'llm_success_rate': (
                (self._call_count - self._fallback_count) / self._call_count
                if self._call_count > 0 else 0.0
            )
        }

    def reset_stats(self):
        """통계 리셋"""
        self._call_count = 0
        self._fallback_count = 0
        self._truncation_count = 0
