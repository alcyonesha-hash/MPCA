#!/usr/bin/env python3
"""
Project 2: LLM 메트릭 ON/OFF 비교 실험

두 가지 조건을 비교:
1. Baseline (메트릭 OFF): LLM 응답 그대로 사용
2. With Metrics (메트릭 ON): Timing + Chunking 적용

평가 기준:
- 응답 길이 분포 (Human Q75=15단어 기준)
- Chunk 수 분포 (Human: 67%/21%/12%)
- 응답 지연 분포 (Quick/Normal/Detailed)
"""

import sys
import os
import json
import random
from datetime import datetime
from typing import Dict, List, Tuple

# 프로젝트 루트 경로 추가
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.llm.client import LLMClient, LLMConfig
from src.agent.timing_policy import TimingPolicy, classify_utterance
from src.policy.conditions import PolicyConfig, ConditionC2


# Human 기준 파라미터 (Project 1에서 추출)
HUMAN_PARAMS = {
    'chunk_length_q75': 15,
    'consecutive_dist': {1: 0.67, 2: 0.21, 3: 0.12},
    'timing': {
        'quick': {'p_immediate': 0.71, 'range': (3, 10)},
        'normal': {'p_immediate': 0.69, 'range': (3, 10)},
        'detailed': {'p_immediate': 0.62, 'range': (3, 10)},
    }
}


# 테스트용 Ubuntu IRC 스타일 프롬프트
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


def run_baseline_test(client: LLMClient, prompts: List[str]) -> List[Dict]:
    """Baseline: LLM 응답 그대로 사용 (메트릭 미적용)"""
    results = []

    for prompt in prompts:
        result = client.generate(prompt=prompt, context=[])

        results.append({
            'prompt': prompt,
            'response': result['text'],
            'word_count': result['word_count'],
            'chunks': [result['text']],  # 단일 청크
            'chunk_count': 1,
            'chunk_lengths': [result['word_count']],
            'delay': 0.0,  # 즉시 응답
            'utt_type': 'unknown',
            'mode': 'baseline',
        })

    return results


def run_metrics_test(
    client: LLMClient,
    timing_policy: TimingPolicy,
    chunking_policy: ConditionC2,
    prompts: List[str]
) -> List[Dict]:
    """With Metrics: Timing + Chunking 적용"""
    results = []

    for prompt in prompts:
        # 1. LLM 응답 생성
        result = client.generate(prompt=prompt, context=[])
        response_text = result['text']

        # 2. Timing Policy 적용
        delay, utt_type, analysis = timing_policy.sample_delay_with_analysis(response_text)

        # 3. Chunking Policy 적용
        chunks_with_delays = chunking_policy.chunk_response(response_text)

        chunk_texts = [c[0] for c in chunks_with_delays]
        chunk_lengths = [len(c[0].split()) for c in chunks_with_delays]
        chunk_delays = [c[1] for c in chunks_with_delays]

        results.append({
            'prompt': prompt,
            'response': response_text,
            'word_count': result['word_count'],
            'chunks': chunk_texts,
            'chunk_count': len(chunk_texts),
            'chunk_lengths': chunk_lengths,
            'chunk_delays': chunk_delays,
            'delay': delay,
            'utt_type': utt_type,
            'tech_score': analysis.get('technical', {}).get('technical_score', 0.0),
            'mode': 'with_metrics',
        })

    return results


def compute_statistics(results: List[Dict], mode: str) -> Dict:
    """결과 통계 계산"""
    stats = {
        'mode': mode,
        'total_responses': len(results),
        'word_counts': [],
        'chunk_counts': [],
        'delays': [],
        'utt_types': {'quick': 0, 'normal': 0, 'detailed': 0, 'unknown': 0},
    }

    for r in results:
        stats['word_counts'].append(r['word_count'])
        stats['chunk_counts'].append(r['chunk_count'])
        stats['delays'].append(r['delay'])
        utt_type = r.get('utt_type', 'unknown')
        if utt_type in stats['utt_types']:
            stats['utt_types'][utt_type] += 1

    # 요약 통계
    import numpy as np

    wc = np.array(stats['word_counts'])
    stats['word_count_summary'] = {
        'mean': float(np.mean(wc)),
        'median': float(np.median(wc)),
        'q75': float(np.percentile(wc, 75)),
        'max': int(np.max(wc)),
    }

    cc = np.array(stats['chunk_counts'])
    stats['chunk_count_summary'] = {
        'mean': float(np.mean(cc)),
        'single_pct': float(np.sum(cc == 1) / len(cc) * 100),
        'double_pct': float(np.sum(cc == 2) / len(cc) * 100),
        'triple_plus_pct': float(np.sum(cc >= 3) / len(cc) * 100),
    }

    delays = np.array(stats['delays'])
    stats['delay_summary'] = {
        'mean': float(np.mean(delays)),
        'median': float(np.median(delays)),
        'immediate_pct': float(np.sum(delays <= 10) / len(delays) * 100),
    }

    return stats


def print_comparison(baseline_stats: Dict, metrics_stats: Dict):
    """비교 결과 출력"""
    print("\n" + "=" * 70)
    print("COMPARISON: Baseline vs With Metrics")
    print("=" * 70)

    print("\n[1] Word Count Distribution")
    print("-" * 50)
    print(f"  Human Q75: 15 words")
    print(f"  {'Baseline':<20} Mean: {baseline_stats['word_count_summary']['mean']:.1f}, "
          f"Q75: {baseline_stats['word_count_summary']['q75']:.1f}")
    print(f"  {'With Metrics':<20} Mean: {metrics_stats['word_count_summary']['mean']:.1f}, "
          f"Q75: {metrics_stats['word_count_summary']['q75']:.1f}")

    print("\n[2] Chunk Count Distribution (Human: 67%/21%/12%)")
    print("-" * 50)
    print(f"  {'Baseline':<20} Single: {baseline_stats['chunk_count_summary']['single_pct']:.0f}%, "
          f"Double: {baseline_stats['chunk_count_summary']['double_pct']:.0f}%, "
          f"3+: {baseline_stats['chunk_count_summary']['triple_plus_pct']:.0f}%")
    print(f"  {'With Metrics':<20} Single: {metrics_stats['chunk_count_summary']['single_pct']:.0f}%, "
          f"Double: {metrics_stats['chunk_count_summary']['double_pct']:.0f}%, "
          f"3+: {metrics_stats['chunk_count_summary']['triple_plus_pct']:.0f}%")

    print("\n[3] Response Delay")
    print("-" * 50)
    print(f"  Human: ~67% immediate (≤10s)")
    print(f"  {'Baseline':<20} Mean: {baseline_stats['delay_summary']['mean']:.1f}s, "
          f"Immediate: {baseline_stats['delay_summary']['immediate_pct']:.0f}%")
    print(f"  {'With Metrics':<20} Mean: {metrics_stats['delay_summary']['mean']:.1f}s, "
          f"Immediate: {metrics_stats['delay_summary']['immediate_pct']:.0f}%")

    print("\n[4] Utterance Type Classification")
    print("-" * 50)
    print(f"  {'Baseline':<20} {baseline_stats['utt_types']}")
    print(f"  {'With Metrics':<20} {metrics_stats['utt_types']}")

    print("\n" + "=" * 70)


def save_results(baseline_results: List[Dict], metrics_results: List[Dict],
                 baseline_stats: Dict, metrics_stats: Dict, output_dir: str):
    """결과 저장"""
    os.makedirs(output_dir, exist_ok=True)

    # 상세 결과
    with open(os.path.join(output_dir, 'baseline_results.json'), 'w') as f:
        json.dump(baseline_results, f, indent=2, ensure_ascii=False)

    with open(os.path.join(output_dir, 'metrics_results.json'), 'w') as f:
        json.dump(metrics_results, f, indent=2, ensure_ascii=False)

    # 통계 요약
    comparison = {
        'timestamp': datetime.now().isoformat(),
        'baseline': baseline_stats,
        'with_metrics': metrics_stats,
        'human_reference': HUMAN_PARAMS,
    }

    with open(os.path.join(output_dir, 'comparison_stats.json'), 'w') as f:
        json.dump(comparison, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to: {output_dir}/")


def print_sample_outputs(baseline_results: List[Dict], metrics_results: List[Dict], n: int = 3):
    """샘플 출력 비교"""
    print("\n" + "=" * 70)
    print("SAMPLE OUTPUTS")
    print("=" * 70)

    for i in range(min(n, len(baseline_results))):
        b = baseline_results[i]
        m = metrics_results[i]

        print(f"\n[Sample {i+1}]")
        print(f"  Prompt: \"{b['prompt'][:60]}...\"" if len(b['prompt']) > 60 else f"  Prompt: \"{b['prompt']}\"")

        print(f"\n  Baseline:")
        print(f"    Response: \"{b['response']}\"")
        print(f"    Words: {b['word_count']}, Chunks: {b['chunk_count']}, Delay: {b['delay']:.1f}s")

        print(f"\n  With Metrics:")
        print(f"    Response: \"{m['response']}\"")
        print(f"    Words: {m['word_count']}, Chunks: {m['chunk_count']}, Delay: {m['delay']:.1f}s")
        print(f"    Type: {m['utt_type']}, Tech Score: {m.get('tech_score', 0):.2f}")
        if m['chunk_count'] > 1:
            print(f"    Chunked output:")
            for j, (chunk, delay) in enumerate(zip(m['chunks'], m.get('chunk_delays', [0]*len(m['chunks'])))):
                print(f"      [{j+1}] (+{delay:.1f}s) \"{chunk}\"")
        print("-" * 70)


def main():
    print("=" * 70)
    print("Project 2: LLM Metric Comparison Test")
    print("=" * 70)

    # LLM 클라이언트 초기화
    print("\n[Setup] Initializing LLM client...")
    llm_config = LLMConfig(
        endpoint='http://localhost:1234/v1/chat/completions',
        target_words=8,
        max_words=15,
        timeout=30
    )
    client = LLMClient(llm_config)

    # Timing Policy 초기화
    print("[Setup] Initializing Timing Policy...")
    timing_policy = TimingPolicy()

    # Chunking Policy 초기화 (C2 조건)
    print("[Setup] Initializing Chunking Policy...")
    policy_config = PolicyConfig(
        condition='C2',
        chunking_enabled=True,
        max_words=15,
        target_words=8,
        chunk_delay_min=1.0,
        chunk_delay_max=3.0,
    )
    # 임베딩 모델 로드 없이 청킹만 테스트하기 위해 간단히 생성
    # ConditionC2는 TopicTracker 필요하므로 직접 chunk_response만 사용

    class SimpleChunker:
        """간단한 청킹 테스터 (TopicTracker 없이)"""
        def __init__(self, config):
            self.config = config

        def chunk_response(self, text: str) -> List[Tuple[str, float]]:
            import re

            words = text.split()
            max_len = self.config.max_words

            if len(words) <= max_len:
                return [(text, 0.0)]

            # 문장 분리
            sentences = re.split(r'(?<=[.!?])\s+', text)
            sentences = [s.strip() for s in sentences if s.strip()]

            # Human-like 분포 샘플링
            r = random.random()
            if r < 0.67:
                target = 1
            elif r < 0.88:
                target = 2
            else:
                target = 3

            # 최소 청크 수 보장
            min_needed = (len(words) + max_len - 1) // max_len
            target = max(target, min_needed)

            # 청킹
            chunks = self._group_sentences(sentences, target, max_len)

            # 지연 추가
            result = []
            for i, chunk in enumerate(chunks):
                delay = 0.0 if i == 0 else random.uniform(
                    self.config.chunk_delay_min, self.config.chunk_delay_max
                )
                result.append((chunk, delay))

            return result

        def _group_sentences(self, sentences, target, max_words):
            if not sentences:
                return []

            if target == 1:
                combined = ' '.join(sentences)
                words = combined.split()
                if len(words) > max_words:
                    return [' '.join(words[:max_words])]
                return [combined]

            chunks = []
            current = []

            for sent in sentences:
                sw = sent.split()
                if len(current) + len(sw) <= max_words:
                    current.extend(sw)
                else:
                    if current:
                        chunks.append(' '.join(current))
                    if len(sw) > max_words:
                        # 긴 문장 분할
                        chunks.append(' '.join(sw[:max_words]))
                        current = sw[max_words:]
                    else:
                        current = sw

            if current:
                chunks.append(' '.join(current))

            return chunks if chunks else [' '.join(sentences)]

    chunking_policy = SimpleChunker(policy_config)

    # 테스트 실행
    print(f"\n[Test] Running {len(TEST_PROMPTS)} test prompts...")

    print("\n  Running Baseline test...")
    baseline_results = run_baseline_test(client, TEST_PROMPTS)

    print("  Running With Metrics test...")
    metrics_results = run_metrics_test(client, timing_policy, chunking_policy, TEST_PROMPTS)

    # 통계 계산
    baseline_stats = compute_statistics(baseline_results, 'baseline')
    metrics_stats = compute_statistics(metrics_results, 'with_metrics')

    # 결과 출력
    print_comparison(baseline_stats, metrics_stats)
    print_sample_outputs(baseline_results, metrics_results, n=3)

    # 결과 저장
    output_dir = os.path.join(os.path.dirname(__file__), 'outputs')
    save_results(baseline_results, metrics_results, baseline_stats, metrics_stats, output_dir)

    # LLM 통계
    print("\n[LLM Stats]")
    stats = client.get_stats()
    print(f"  Total calls: {stats['total_calls']}")
    print(f"  Success rate: {stats['llm_success_rate']*100:.0f}%")
    print(f"  Truncations: {stats['truncation_count']}")


if __name__ == "__main__":
    main()
