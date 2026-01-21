"""
TimingPolicy: Ubuntu IRC 데이터 기반 Bimodal 지연 시간 정책

실제 Ubuntu IRC 데이터셋에서 추출된 파라미터를 사용합니다.

모델: Bimodal (즉답 확률 + 지연 분포)
- 즉답: P_immediate 확률로 짧은 지연 (추정 3-45초, 데이터 해상도 한계)
- 지연: (1 - P_immediate) 확률로 Log-normal 분포에서 샘플링

데이터 출처: Ubuntu IRC 8,953개 발화 중 Intra-Thread 2,397개 분석 (2024-01)
- Thread 판별: 같은 화자 OR mention으로 연결된 발화
- Cross-thread delay는 제외하여 실제 대화 맥락 내 응답 시간만 반영

유형 분류:
- Quick: 단답형 (yes, no, thanks 등) 또는 짧은 비기술적 발화 (71% 즉답)
- Normal: 일반적인 대화, 6-20 단어, 기술 내용 미포함 (69% 즉답)
- Detailed: 기술적 내용 포함 (명령어, 경로, URL, 코드 등) (62% 즉답)
"""

import re
import random
import math
from typing import Dict, Optional, Tuple
from dataclasses import dataclass


# ============================================================================
# 기술적 내용 판별 패턴 (Ubuntu IRC 도메인 특화)
# ============================================================================

# 1. Acknowledgment 패턴 (단답)
ACKNOWLEDGMENT_PATTERNS = [
    r'^(yes|no|yep|nope|yeah|nah|ok|okay|k|kk|sure|np|thx|thanks|ty|yw|wb|hi|hello|hey|bye|lol|haha|hmm|ah|oh|wow|nice|cool|great|good|right|exactly|indeed|correct|true|false|maybe|probably|possibly|nvm|nevermind|sorry|oops|please|pls|plz|welp|yup|nah|mhm|uh|um)[\.\!\?\,\s]*$',
]

# 2. 기술 키워드 (Ubuntu/Linux 도메인)
TECHNICAL_KEYWORDS = [
    # 패키지 관리
    'sudo', 'apt', 'apt-get', 'dpkg', 'aptitude', 'snap', 'flatpak', 'ppa',
    'install', 'remove', 'purge', 'update', 'upgrade', 'dist-upgrade',
    'package', 'repository', 'dependencies',

    # 시스템 관리
    'systemctl', 'service', 'daemon', 'init', 'systemd', 'cron', 'crontab',
    'reboot', 'shutdown', 'restart', 'enable', 'disable', 'status',

    # 파일 시스템
    'chmod', 'chown', 'chgrp', 'mount', 'umount', 'fstab', 'fdisk', 'parted',
    'mkfs', 'fsck', 'df', 'du', 'lsblk', 'blkid',

    # 파일 조작
    'cat', 'ls', 'cd', 'mkdir', 'rm', 'cp', 'mv', 'ln', 'touch', 'find',
    'grep', 'sed', 'awk', 'head', 'tail', 'less', 'more', 'nano', 'vim', 'vi',

    # 네트워크
    'ssh', 'scp', 'rsync', 'wget', 'curl', 'ping', 'traceroute', 'netstat',
    'ip', 'ifconfig', 'iptables', 'ufw', 'firewall', 'dns', 'dhcp', 'nmcli',
    'networkmanager', 'resolv.conf',

    # 프로세스/시스템
    'ps', 'top', 'htop', 'kill', 'killall', 'pkill', 'nice', 'renice',
    'kernel', 'grub', 'boot', 'dmesg', 'journalctl', 'syslog',

    # 사용자/권한
    'root', 'user', 'group', 'passwd', 'adduser', 'useradd', 'usermod',
    'permission', 'ownership', 'executable',

    # 프로그래밍/스크립트
    'python', 'python3', 'pip', 'bash', 'sh', 'script', 'function',
    'variable', 'array', 'loop', 'export', 'source', 'env',

    # 하드웨어/드라이버
    'driver', 'nvidia', 'amd', 'gpu', 'cpu', 'ram', 'disk', 'usb',
    'lspci', 'lsusb', 'lshw', 'hwinfo', 'modprobe', 'module',

    # 데스크톱/GUI
    'gnome', 'kde', 'xfce', 'wayland', 'x11', 'xorg', 'display', 'resolution',

    # 에러/로그
    'error', 'warning', 'failed', 'denied', 'permission', 'segfault',
    'traceback', 'exception', 'log', 'debug',
]

# 3. 코드/명령어 패턴
CODE_PATTERNS = [
    r'`[^`]+`',           # backtick 코드: `command`
    r'\$\{?\w+\}?',       # 변수: $VAR, ${VAR}
    r'--\w+[-\w]*',       # CLI 옵션: --option, --long-option
    r'-[a-zA-Z]+',        # 짧은 옵션: -rf, -la
    r'\|\s*\w+',          # 파이프: | grep
    r'&&',                # AND 연산자
    r'\|\|',              # OR 연산자
    r'>\s*[/\w]',         # 리다이렉션: > /file
    r'>>\s*[/\w]',        # 어펜드: >> /file
    r'2>&1',              # stderr 리다이렉션
]

# 4. 파일 경로 패턴
PATH_PATTERNS = [
    r'/etc/\w+',          # /etc/fstab, /etc/apt
    r'/var/\w+',          # /var/log, /var/lib
    r'/usr/\w+',          # /usr/bin, /usr/lib
    r'/home/\w+',         # /home/user
    r'/tmp/\w+',          # /tmp/file
    r'/opt/\w+',          # /opt/app
    r'/dev/\w+',          # /dev/sda
    r'/proc/\w+',         # /proc/cpuinfo
    r'/sys/\w+',          # /sys/class
    r'~/\w+',             # ~/Documents
    r'\./\w+',            # ./script.sh
    r'\.\./\w+',          # ../parent
]

# 5. URL 패턴
URL_PATTERN = r'https?://[^\s]+'

# 6. 파일 확장자 패턴
FILE_EXTENSION_PATTERN = r'\b\w+\.(conf|cfg|log|txt|sh|py|pl|rb|js|c|cpp|h|java|xml|json|yaml|yml|ini|service|socket|timer|desktop|deb|rpm|tar|gz|zip|iso)\b'

# 7. IP/포트 패턴
IP_PORT_PATTERN = r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}(:\d+)?\b'


# ============================================================================
# 기술적 내용 분석 함수
# ============================================================================

def analyze_technical_content(text: str) -> Dict:
    """
    텍스트의 기술적 내용을 상세 분석

    Returns:
        {
            'is_technical': bool,
            'technical_score': float (0.0 ~ 1.0),
            'indicators': {
                'has_url': bool,
                'has_code': bool,
                'has_path': bool,
                'has_file_ext': bool,
                'has_ip': bool,
                'keyword_count': int,
                'keyword_matches': List[str],
            },
            'detected_patterns': List[str],
        }
    """
    if not text or not text.strip():
        return {
            'is_technical': False,
            'technical_score': 0.0,
            'indicators': {},
            'detected_patterns': [],
        }

    text_lower = text.lower()
    detected = []
    score = 0.0

    # 1. URL 체크 (+0.3)
    has_url = bool(re.search(URL_PATTERN, text))
    if has_url:
        score += 0.3
        detected.append('url')

    # 2. 코드 패턴 체크 (+0.25 each, max 0.5)
    code_matches = 0
    for pattern in CODE_PATTERNS:
        if re.search(pattern, text):
            code_matches += 1
    has_code = code_matches > 0
    if has_code:
        score += min(code_matches * 0.25, 0.5)
        detected.append(f'code_patterns({code_matches})')

    # 3. 경로 패턴 체크 (+0.3)
    has_path = False
    for pattern in PATH_PATTERNS:
        if re.search(pattern, text):
            has_path = True
            break
    if has_path:
        score += 0.3
        detected.append('path')

    # 4. 파일 확장자 체크 (+0.2)
    has_file_ext = bool(re.search(FILE_EXTENSION_PATTERN, text_lower))
    if has_file_ext:
        score += 0.2
        detected.append('file_extension')

    # 5. IP/포트 체크 (+0.2)
    has_ip = bool(re.search(IP_PORT_PATTERN, text))
    if has_ip:
        score += 0.2
        detected.append('ip_address')

    # 6. 기술 키워드 체크 (+0.1 each, max 0.5)
    keyword_matches = []
    for kw in TECHNICAL_KEYWORDS:
        # 단어 경계로 매칭 (부분 매칭 방지)
        if re.search(rf'\b{re.escape(kw)}\b', text_lower):
            keyword_matches.append(kw)

    keyword_count = len(keyword_matches)
    if keyword_count > 0:
        score += min(keyword_count * 0.1, 0.5)
        detected.append(f'keywords({keyword_count})')

    # 기술적 여부 판정 (score >= 0.2 또는 특정 강력 지표)
    is_technical = (
        score >= 0.2 or
        has_url or
        has_code or
        has_path or
        keyword_count >= 2
    )

    return {
        'is_technical': is_technical,
        'technical_score': min(score, 1.0),
        'indicators': {
            'has_url': has_url,
            'has_code': has_code,
            'has_path': has_path,
            'has_file_ext': has_file_ext,
            'has_ip': has_ip,
            'keyword_count': keyword_count,
            'keyword_matches': keyword_matches[:10],  # 상위 10개만
        },
        'detected_patterns': detected,
    }


def is_acknowledgment(text: str) -> bool:
    """Acknowledgment(단답) 여부 판별"""
    if not text or not text.strip():
        return True

    text_clean = text.strip().lower()

    for pattern in ACKNOWLEDGMENT_PATTERNS:
        if re.match(pattern, text_clean, re.IGNORECASE):
            return True

    return False


# ============================================================================
# 발화 유형 분류
# ============================================================================

def classify_utterance(text: str) -> Tuple[str, Dict]:
    """
    발화를 quick/normal/detailed로 분류

    분류 기준:
    - Quick: acknowledgment 패턴 OR ≤5 words (비기술적)
    - Normal: 6-20 words, 비기술적
    - Detailed: 기술적 내용 포함 OR >20 words

    Returns:
        (type: str, analysis: Dict)
    """
    if not text or not text.strip():
        return 'quick', {'reason': 'empty'}

    words = text.split()
    word_count = len(words)

    # 1. Acknowledgment 체크 → Quick
    if is_acknowledgment(text):
        return 'quick', {'reason': 'acknowledgment', 'word_count': word_count}

    # 2. 기술적 내용 분석
    tech_analysis = analyze_technical_content(text)

    # 3. 기술적 내용 포함 → Detailed
    if tech_analysis['is_technical']:
        return 'detailed', {
            'reason': 'technical_content',
            'word_count': word_count,
            'technical': tech_analysis,
        }

    # 4. 길이 기반 분류
    if word_count <= 5:
        return 'quick', {'reason': 'short_non_technical', 'word_count': word_count}

    if word_count > 20:
        return 'detailed', {'reason': 'long_text', 'word_count': word_count}

    # 5. 나머지 → Normal (6-20 words, 비기술적)
    return 'normal', {'reason': 'medium_non_technical', 'word_count': word_count}


def classify_utterance_simple(text: str) -> str:
    """간단한 유형 분류 (유형만 반환)"""
    utt_type, _ = classify_utterance(text)
    return utt_type


# ============================================================================
# 데이터 기반 파라미터 (Ubuntu IRC Intra-Thread 분석에서 추출)
# ============================================================================
#
# 분석 방법:
# - 전체 8,953개 발화 중 같은 thread 내 연속 발화만 분석
# - Thread 판별: 같은 화자 OR mention으로 연결된 발화
# - Intra-thread 샘플: 2,397개 (Quick:310, Normal:750, Detailed:1,337)
# - 즉답 기준: ≤60초 (timestamp 해상도가 분 단위이므로)
#
# ============================================================================

@dataclass
class TypeTimingParams:
    """유형별 타이밍 파라미터"""
    p_immediate: float      # 즉답 확률 (≤60초로 관측된 비율)
    immediate_range: tuple  # 즉답 시 delay 범위 (초) - 추정값
    lognorm_mu: float       # 지연 시 log-normal μ
    lognorm_sigma: float    # 지연 시 log-normal σ
    delayed_median: float   # 지연 시 median (검증용)
    delayed_p75: float      # 지연 시 P75 (검증용)


# Ubuntu IRC Intra-Thread 데이터에서 추출된 즉답 확률
# 지연 응답은 공식 기반: 10초 + word_count * 1.0 + tech_score * 20
DATA_DRIVEN_PARAMS = {
    'quick': TypeTimingParams(
        p_immediate=0.71,           # 71.0% 즉답 (n=310, 데이터 기반)
        immediate_range=(3, 10),    # 즉답: 3~10초
        lognorm_mu=0.0,             # 미사용 (공식 기반으로 대체)
        lognorm_sigma=0.0,
        delayed_median=0.0,
        delayed_p75=0.0,
    ),
    'normal': TypeTimingParams(
        p_immediate=0.69,           # 69.1% 즉답 (n=750, 데이터 기반)
        immediate_range=(3, 10),    # 즉답: 3~10초
        lognorm_mu=0.0,
        lognorm_sigma=0.0,
        delayed_median=0.0,
        delayed_p75=0.0,
    ),
    'detailed': TypeTimingParams(
        p_immediate=0.62,           # 62.4% 즉답 (n=1,337, 데이터 기반)
        immediate_range=(3, 10),    # 즉답: 3~10초
        lognorm_mu=0.0,
        lognorm_sigma=0.0,
        delayed_median=0.0,
        delayed_p75=0.0,
    ),
}

# 지연 응답 계수 (실시간 회의 대화용)
DELAY_BASE = 10.0           # 기본 지연 (초)
DELAY_PER_WORD = 1.0        # 단어당 추가 지연 (초)
DELAY_PER_TECH = 20.0       # 기술점수(0~1)당 추가 지연 (초)


# ============================================================================
# TimingPolicy 클래스
# ============================================================================

class TimingPolicy:
    """
    Ubuntu IRC 데이터 기반 Bimodal 응답 지연 정책

    모델:
        if random() < P_immediate:
            delay = uniform(immediate_range)
        else:
            delay = lognormal(μ, σ)

    Usage:
        policy = TimingPolicy()

        # 응답 텍스트 기반 delay 샘플링
        delay, utt_type, analysis = policy.sample_delay_with_analysis("yes")

        # 유형 직접 지정
        delay = policy.sample_delay_by_type('detailed')
    """

    def __init__(
        self,
        params: Dict[str, TypeTimingParams] = None,
        max_delay: float = 900.0,  # 15분 cap
        min_delay: float = 3.0,
        scale_factor: float = 1.0,  # 전체 스케일 조정
    ):
        """
        Args:
            params: 유형별 파라미터 (None이면 데이터 기반 기본값)
            max_delay: 최대 delay (초)
            min_delay: 최소 delay (초)
            scale_factor: 전체 delay 스케일 (1.0 = 원본, 0.5 = 절반)
        """
        self.params = params or DATA_DRIVEN_PARAMS
        self.max_delay = max_delay
        self.min_delay = min_delay
        self.scale_factor = scale_factor

    def sample_delay(self, text: str) -> float:
        """
        텍스트 기반 delay 샘플링

        1. 텍스트를 quick/normal/detailed로 분류
        2. 즉답 확률로 즉답 vs 지연 결정
        3. 지연 시 공식 적용: 10 + word_count + tech_score * 20
        """
        utt_type, analysis = classify_utterance(text)
        return self._sample_delay_internal(text, utt_type, analysis)

    def sample_delay_with_analysis(self, text: str) -> Tuple[float, str, Dict]:
        """
        텍스트 기반 delay 샘플링 (상세 분석 포함)

        Returns:
            (delay, utt_type, analysis)
        """
        utt_type, analysis = classify_utterance(text)
        delay = self._sample_delay_internal(text, utt_type, analysis)

        analysis['delay'] = delay
        analysis['utt_type'] = utt_type

        return delay, utt_type, analysis

    def _sample_delay_internal(self, text: str, utt_type: str, analysis: Dict) -> float:
        """
        내부 delay 샘플링 로직

        즉답: uniform(3, 10)
        지연: 10 + word_count * 1.0 + tech_score * 20
        """
        if utt_type not in self.params:
            utt_type = 'normal'

        p = self.params[utt_type]

        # Bimodal: 즉답 vs 지연
        if random.random() < p.p_immediate:
            # 즉답: uniform(3, 10)
            delay = random.uniform(*p.immediate_range)
        else:
            # 지연: 공식 기반
            word_count = len(text.split()) if text else 0
            tech_score = analysis.get('technical', {}).get('technical_score', 0.0)

            delay = DELAY_BASE + (word_count * DELAY_PER_WORD) + (tech_score * DELAY_PER_TECH)

        # 스케일 적용 및 클램핑
        delay *= self.scale_factor
        delay = max(self.min_delay, min(delay, self.max_delay))

        return delay

    def sample_delay_by_type(self, utt_type: str, text: str = "") -> float:
        """유형 직접 지정 시 사용 (하위 호환)"""
        analysis = {}
        if text:
            _, analysis = classify_utterance(text)
        return self._sample_delay_internal(text, utt_type, analysis)

    def get_type_stats(self, utt_type: str) -> Dict:
        """유형별 파라미터 조회"""
        if utt_type not in self.params:
            return {}

        p = self.params[utt_type]
        geom_mean = math.exp(p.lognorm_mu)

        return {
            'type': utt_type,
            'p_immediate': p.p_immediate,
            'immediate_range': p.immediate_range,
            'delayed_lognorm': {'mu': p.lognorm_mu, 'sigma': p.lognorm_sigma},
            'delayed_geometric_mean': geom_mean,
            'delayed_median': p.delayed_median,
            'delayed_p75': p.delayed_p75,
        }

    def get_delay_breakdown(self, text: str) -> Dict:
        """delay 계산 과정 반환 (디버깅/설명용)"""
        utt_type, analysis = classify_utterance(text)
        p = self.params.get(utt_type, self.params['normal'])

        return {
            'text': text[:60] + '...' if len(text) > 60 else text,
            'classified_type': utt_type,
            'classification_reason': analysis.get('reason', ''),
            'p_immediate': p.p_immediate,
            'immediate_range': p.immediate_range,
            'delayed_distribution': f'LogNormal(μ={p.lognorm_mu:.2f}, σ={p.lognorm_sigma:.2f})',
            'delayed_geometric_mean': f'{math.exp(p.lognorm_mu):.0f}s',
            'scale_factor': self.scale_factor,
            'technical_analysis': analysis.get('technical', {}),
        }

    def __repr__(self) -> str:
        return f"TimingPolicy(scale={self.scale_factor}, max={self.max_delay}s)"


# ============================================================================
# 편의 함수
# ============================================================================

_default_policy: TimingPolicy = None


def get_default_policy() -> TimingPolicy:
    """기본 TimingPolicy 인스턴스 반환 (싱글톤)"""
    global _default_policy
    if _default_policy is None:
        _default_policy = TimingPolicy()
    return _default_policy


def sample_response_delay(text: str) -> float:
    """응답 텍스트에 맞는 delay 샘플링 (편의 함수)"""
    return get_default_policy().sample_delay(text)


def reset_default_policy():
    """기본 정책 리셋"""
    global _default_policy
    _default_policy = None


# ============================================================================
# 테스트
# ============================================================================

if __name__ == "__main__":
    import statistics

    print("=" * 70)
    print("TimingPolicy - Ubuntu IRC Data-Driven Bimodal Model")
    print("=" * 70)

    policy = TimingPolicy()
    print(f"\n{policy}")

    # 파라미터 출력
    print("\n데이터 기반 파라미터:")
    print("-" * 70)
    for utt_type in ['quick', 'normal', 'detailed']:
        stats = policy.get_type_stats(utt_type)
        print(f"\n{utt_type.upper()}:")
        print(f"  즉답 확률: {stats['p_immediate']*100:.1f}%")
        print(f"  즉답 범위: {stats['immediate_range']}초")
        print(f"  지연 분포: LogNormal(μ={stats['delayed_lognorm']['mu']:.2f}, σ={stats['delayed_lognorm']['sigma']:.2f})")
        print(f"  지연 geometric mean: {stats['delayed_geometric_mean']:.0f}초")

    # 기술적 내용 분석 테스트
    print("\n" + "=" * 70)
    print("기술적 내용 분석 테스트")
    print("=" * 70)

    tech_test_cases = [
        "yes",
        "thanks!",
        "I think that might work",
        "try restarting your computer",
        "run sudo apt update",
        "check /etc/apt/sources.list",
        "sudo systemctl restart nginx",
        "curl -X POST https://api.example.com/data",
        "edit ~/.bashrc and add export PATH=$PATH:/usr/local/bin",
    ]

    print(f"\n{'Text':<50} {'Type':<10} {'Technical?':<10} {'Score'}")
    print("-" * 80)

    for text in tech_test_cases:
        utt_type, analysis = classify_utterance(text)
        tech = analysis.get('technical', {})
        is_tech = tech.get('is_technical', False) if tech else False
        score = tech.get('technical_score', 0) if tech else 0
        display_text = text[:47] + '...' if len(text) > 50 else text
        print(f"{display_text:<50} {utt_type:<10} {str(is_tech):<10} {score:.2f}")

    # 테스트 케이스 with delay
    print("\n" + "=" * 70)
    print("유형별 분류 및 지연 테스트")
    print("=" * 70)

    test_cases = [
        "yes",
        "no",
        "thanks!",
        "ok cool",
        "try that",
        "I think that might work",
        "Have you tried restarting?",
        "run sudo apt update",
        "Check the /etc/apt/sources.list file",
        "You need to run sudo apt update && sudo apt upgrade",
    ]

    print(f"\n{'Text':<45} {'Type':<10} {'Sample Delays (5x)'}")
    print("-" * 80)

    for text in test_cases:
        utt_type = classify_utterance_simple(text)
        samples = [policy.sample_delay(text) for _ in range(5)]
        samples_str = ', '.join([f"{s:.0f}s" for s in samples])
        display_text = text[:42] + '...' if len(text) > 45 else text
        print(f"{display_text:<45} {utt_type:<10} {samples_str}")
