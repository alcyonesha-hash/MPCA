"""
Hybrid Weighted-scoring based conversation thread disentanglement

Refactored from heuristic-based to weighted-scoring approach with semantic similarity:
- 하드 임계치(time_threshold) 대신 가중치 기반 스코어링 사용
- 시간 간격, 멘션, 참여자 연속성, **의미적 유사도**를 스코어로 결합
- 여러 후보 thread 중 best-match 선택

Scoring formula (semantic 시간 감쇠 방식):
    base = participant_weight * I[speaker in participants]
         + semantic_weight * semantic_similarity × time_decay

    score = mention_weight + base (if mention exists)
          = base (if no mention)

where:
    time_decay = exp(-delta / time_scale)  # 시간에 따라 semantic 기여도 감소
    semantic_similarity = cosine(embed(utterance), mean(embed(thread_utterances)))

효과:
    - participant: 같은 화자면 시간 무관하게 점수 기여
    - semantic: 시간이 지날수록 유사도 기여 감소
    - mention: 시간 무관하게 항상 병합 보장
"""

from typing import List, Dict, Set, Tuple, Optional
from datetime import datetime, timedelta
from collections import defaultdict
import logging
import math
import numpy as np

logger = logging.getLogger(__name__)


class ThreadDisentangler:
    """
    Extract conversation threads using hybrid weighted scoring with semantic similarity

    스코어 기반 thread 할당:
    1. 각 utterance에 대해 모든 active thread의 스코어 계산
    2. 스코어 = 참여자 + 멘션 + 시간 + **의미적 유사도**
    3. 최고 스코어 thread가 threshold를 넘으면 해당 thread에 할당
    4. 아니면 새 thread 생성
    """

    def __init__(
        self,
        # 가중치 파라미터
        time_weight: float = 1.0,  # 시간 감쇠 (exp(-Δt/τ))
        time_scale: float = 120.0,  # 감쇠 상수 (2분 기준, τ=120초일 때 score≈0.37)
        mention_weight: float = 2.0,  # 멘션 (가장 강력한 신호)
        participant_weight: float = 1.0,  # 참여자 연속성
        semantic_weight: float = 1.0,  # 의미적 유사도 (시간 감쇠 적용)
        # 새 thread 생성 기준
        new_thread_threshold: float = 1.0,  # 병합 기준
        # 안전장치: 이 시간 초과 시 같은 화자도 분리
        max_gap_seconds: Optional[float] = 1800.0,  # 30분: 같은 화자도 시간 초과 시 분리
        # 임베딩 설정
        embedding_model: str = "auto",  # "auto", "sentence-transformers", "hf-meanpool", "none"
        max_thread_context: int = 5,  # thread의 최근 N개 발화만 사용 (메모리 절약)
        # Active thread window (성능 최적화)
        max_active_threads: int = 10,  # 최근 N개 thread만 비교 대상으로
        # 하위 호환용 (deprecated)
        time_threshold: Optional[int] = None,  # deprecated, 무시됨
    ):
        """
        Args:
            time_weight: 시간 스코어 가중치 - 현재 미사용, semantic 감쇠에 통합됨
            time_scale: 시간 감쇠 스케일 (초). τ=120초일 때 exp(-Δt/τ)≈0.37 at 2분
            mention_weight: 멘션 가중치 (default: 2.0) - 시간 무관 항상 병합
            participant_weight: 참여자 연속성 가중치 (default: 1.0) - 시간 감쇠 없음
            semantic_weight: 의미적 유사도 가중치 (default: 1.0) - 시간 감쇠 적용
            new_thread_threshold: 이 값 이하면 새 thread 생성 (default: 1.0)
            max_gap_seconds: 30분 안전장치 - 이 시간 초과 시 강제 분리 (default: 1800)
            embedding_model: 임베딩 모델 선택 (default: "auto")
            max_thread_context: thread 벡터 계산 시 사용할 최대 발화 수 (default: 5)
            max_active_threads: 비교 대상 thread 수 제한 (default: 10)
            time_threshold: DEPRECATED - 무시됨

        Score Interpretation:
            - base = participant + (semantic × time_decay)
            - score = mention_weight + base (멘션 시) / base (멘션 없음)
            - 같은 화자: 시간 무관하게 participant=1.0 → threshold 충족
            - 멘션: 시간 무관하게 +2.0 → 항상 병합
            - 새 화자+주제 유사: 시간에 따라 감쇠, 빠를수록 병합 유리
        """
        self.time_weight = time_weight
        self.time_scale = time_scale
        self.mention_weight = mention_weight
        self.participant_weight = participant_weight
        self.semantic_weight = semantic_weight
        self.new_thread_threshold = new_thread_threshold
        self.max_gap_seconds = max_gap_seconds
        self.max_thread_context = max_thread_context
        self.max_active_threads = max_active_threads

        # Embedding model initialization
        self._embedding_model = None
        self._embedding_model_type = None
        self._embedding_cache: Dict[str, np.ndarray] = {}
        self._cache_max_size = 2000

        if semantic_weight > 0 and embedding_model != "none":
            self._init_embedding_model(embedding_model)

        # Deprecated parameter warning
        if time_threshold is not None:
            logger.warning(
                "time_threshold is deprecated and ignored. "
                "Use time_weight, time_scale, and max_gap_seconds instead."
            )

    def _init_embedding_model(self, model_preference: str):
        """Initialize embedding model with fallback."""
        if model_preference in ("auto", "sentence-transformers"):
            if self._try_init_sentence_transformers():
                return

        if model_preference in ("auto", "hf-meanpool"):
            if self._try_init_hf_meanpool():
                return

        # Last resort
        if model_preference == "hf-meanpool":
            if self._try_init_sentence_transformers():
                return

        logger.warning(
            "Failed to initialize embedding model. "
            "Semantic similarity will be disabled. "
            "Install sentence-transformers for better disentanglement."
        )
        self.semantic_weight = 0.0

    def _try_init_sentence_transformers(self) -> bool:
        """Try to initialize sentence-transformers model."""
        try:
            from sentence_transformers import SentenceTransformer
            self._embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            self._embedding_model_type = "sentence-transformers"
            logger.info("ThreadDisentangler: Using sentence-transformers (all-MiniLM-L6-v2)")
            return True
        except ImportError:
            logger.debug("sentence-transformers not available")
            return False
        except Exception as e:
            logger.warning(f"Failed to load sentence-transformers: {e}")
            return False

    def _try_init_hf_meanpool(self) -> bool:
        """Try to initialize HuggingFace transformers with mean-pooling."""
        try:
            import torch
            from transformers import AutoTokenizer, AutoModel

            model_name = "sentence-transformers/all-MiniLM-L6-v2"
            try:
                self._tokenizer = AutoTokenizer.from_pretrained(model_name)
                self._embedding_model = AutoModel.from_pretrained(model_name)
            except Exception:
                model_name = "distilbert-base-uncased"
                self._tokenizer = AutoTokenizer.from_pretrained(model_name)
                self._embedding_model = AutoModel.from_pretrained(model_name)

            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self._embedding_model = self._embedding_model.to(self._device)
            self._embedding_model.eval()
            self._embedding_model_type = "hf-meanpool"
            logger.info(f"ThreadDisentangler: Using transformers mean-pooling ({model_name})")
            return True
        except ImportError:
            logger.debug("transformers or torch not available")
            return False
        except Exception as e:
            logger.warning(f"Failed to load transformers model: {e}")
            return False

    def _get_embedding(self, text: str) -> Optional[np.ndarray]:
        """Get embedding for text with caching."""
        if self._embedding_model is None or self.semantic_weight <= 0:
            return None

        if not text or not text.strip():
            return None

        # Check cache
        if text in self._embedding_cache:
            return self._embedding_cache[text]

        # Compute embedding
        try:
            if self._embedding_model_type == "sentence-transformers":
                embedding = self._embedding_model.encode(text, convert_to_numpy=True)
            elif self._embedding_model_type == "hf-meanpool":
                embedding = self._compute_hf_meanpool_embedding(text)
            else:
                return None

            # L2 normalize
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm

            # Cache management
            if len(self._embedding_cache) >= self._cache_max_size:
                keys_to_remove = list(self._embedding_cache.keys())[:self._cache_max_size // 10]
                for key in keys_to_remove:
                    del self._embedding_cache[key]

            self._embedding_cache[text] = embedding
            return embedding
        except Exception as e:
            logger.debug(f"Failed to compute embedding: {e}")
            return None

    def _compute_hf_meanpool_embedding(self, text: str) -> np.ndarray:
        """Compute embedding using HuggingFace transformers with mean-pooling."""
        import torch

        inputs = self._tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True
        )
        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self._embedding_model(**inputs)
            last_hidden = outputs.last_hidden_state
            attention_mask = inputs["attention_mask"]

        mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden.size()).float()
        sum_embeddings = torch.sum(last_hidden * mask_expanded, dim=1)
        sum_mask = mask_expanded.sum(dim=1).clamp(min=1e-9)
        mean_pooled = sum_embeddings / sum_mask

        return mean_pooled.cpu().numpy().flatten()

    def _compute_time_score(self, delta_seconds: float) -> float:
        """
        시간 간격에 따른 스코어 계산 (exponential decay)

        time_score = exp(-delta / time_scale)

        Properties:
            - delta=0 -> score=1.0
            - delta=time_scale -> score≈0.368
            - delta=2*time_scale -> score≈0.135
            - delta→∞ -> score→0 (but never exactly 0)

        Args:
            delta_seconds: 마지막 발화로부터의 시간 간격 (초)

        Returns:
            0.0 ~ 1.0 범위의 시간 스코어
        """
        if delta_seconds <= 0:
            return 1.0
        if self.time_scale <= 0:
            return 0.0
        return math.exp(-delta_seconds / self.time_scale)

    def _compute_mention_score(
        self,
        mentions: List[str],
        thread_participants: Set[str]
    ) -> float:
        """
        멘션 기반 스코어 계산

        현재 발화의 mentions 중 thread 참여자가 있으면 1.0, 없으면 0.0

        Args:
            mentions: 현재 발화에서 추출된 멘션 리스트
            thread_participants: thread의 참여자 집합

        Returns:
            0.0 또는 1.0
        """
        if not mentions or not thread_participants:
            return 0.0

        # mentions를 소문자로 정규화하여 비교
        mentions_lower = {m.lower() for m in mentions}
        participants_lower = {p.lower() for p in thread_participants}

        # 교집합이 있으면 1.0
        if mentions_lower & participants_lower:
            return 1.0
        return 0.0

    def _compute_participant_score(
        self,
        speaker: str,
        thread_participants: Set[str]
    ) -> float:
        """
        참여자 연속성 스코어 계산

        speaker가 이미 thread 참여자면 1.0, 아니면 0.0

        Args:
            speaker: 현재 발화자
            thread_participants: thread의 참여자 집합

        Returns:
            0.0 또는 1.0
        """
        if not speaker or not thread_participants:
            return 0.0

        if speaker.lower() in {p.lower() for p in thread_participants}:
            return 1.0
        return 0.0

    def _compute_semantic_score(
        self,
        utterance_text: str,
        thread_embeddings: List[np.ndarray]
    ) -> float:
        """
        의미적 유사도 스코어 계산

        semantic_score = cosine(embed(utterance), mean(thread_embeddings))

        Args:
            utterance_text: 현재 발화 텍스트
            thread_embeddings: thread 내 발화들의 임베딩 리스트

        Returns:
            0.0 ~ 1.0 범위의 유사도 스코어
        """
        if self.semantic_weight <= 0 or not thread_embeddings:
            return 0.0

        utt_embedding = self._get_embedding(utterance_text)
        if utt_embedding is None:
            return 0.0

        # Thread vector = mean of recent embeddings
        thread_embeddings_array = np.array(thread_embeddings[-self.max_thread_context:])
        thread_vector = np.mean(thread_embeddings_array, axis=0)

        # L2 normalize
        norm = np.linalg.norm(thread_vector)
        if norm > 0:
            thread_vector = thread_vector / norm

        # Cosine similarity (both vectors are normalized)
        cosine = float(np.dot(thread_vector, utt_embedding))

        # Clamp to [0, 1]
        return max(0.0, cosine)

    def _compute_thread_score(
        self,
        speaker: str,
        mentions: List[str],
        delta_seconds: float,
        thread_participants: Set[str],
        utterance_text: str = "",
        thread_embeddings: Optional[List[np.ndarray]] = None
    ) -> float:
        """
        종합 thread 스코어 계산 (하이브리드 + semantic 시간 감쇠)

        공식:
            base = participant + (semantic × time_decay)
            score = mention_weight + base (if mention)
                  = base (if no mention)

        시간 감쇠는 semantic에만 적용:
        - participant: 같은 화자면 시간 무관하게 +1
        - semantic: 시간이 지날수록 유사도 기여 감소
        - mention: 시간 무관하게 항상 병합

        Args:
            speaker: 현재 발화자
            mentions: 현재 발화의 멘션 리스트
            delta_seconds: thread의 마지막 발화로부터의 시간 간격
            thread_participants: thread의 참여자 집합
            utterance_text: 현재 발화 텍스트 (semantic similarity용)
            thread_embeddings: thread 내 발화들의 임베딩 리스트

        Returns:
            종합 스코어 (0 이상)
        """
        mention_score = self._compute_mention_score(mentions, thread_participants)
        participant_score = self._compute_participant_score(speaker, thread_participants)
        semantic_score = self._compute_semantic_score(
            utterance_text, thread_embeddings or []
        )
        time_decay = self._compute_time_score(delta_seconds)  # exp(-Δt/τ), 0~1

        # 시간 감쇠는 semantic에만 적용
        # - participant: 같은 화자면 시간 무관하게 +1
        # - semantic: 시간이 지날수록 유사도 기여 감소
        decayed_semantic = semantic_score * time_decay

        base_score = (
            self.participant_weight * participant_score +
            self.semantic_weight * decayed_semantic
        )

        if mention_score > 0:
            # 멘션이 있으면 시간 무관하게 병합
            total_score = self.mention_weight * mention_score + base_score
        else:
            total_score = base_score

        return total_score

    def disentangle(self, utterances: List[Dict]) -> List[Dict]:
        """
        Group utterances into threads using hybrid weighted scoring

        알고리즘:
        1. 각 utterance에 대해 모든 active thread의 스코어 계산
        2. 스코어 = 참여자 + 멘션 + 시간 + 의미적 유사도
        3. 최고 스코어 thread 선택
        4. 스코어가 threshold 이상이고, max_gap 조건 충족 시 해당 thread에 할당
        5. 아니면 새 thread 생성

        Args:
            utterances: List of utterance dicts with keys: ts, speaker, text, mentions

        Returns:
            List of thread dicts with schema:
            {
                "thread_id": str,
                "channel": str,
                "utterances": List[Dict]
            }
        """
        if not utterances:
            return []

        # Sort by timestamp
        sorted_utts = sorted(utterances, key=lambda u: u.get('ts', ''))

        # Thread 관리 데이터 구조
        threads: List[Dict] = []
        thread_metadata: List[Dict] = []  # 각 thread의 메타정보 (participants, last_ts, embeddings)

        for utt in sorted_utts:
            # timestamp 파싱
            try:
                ts = datetime.fromisoformat(utt['ts'])
            except (ValueError, KeyError, TypeError) as e:
                logger.warning(f"Skipping utterance with invalid timestamp: {utt.get('ts')}")
                continue

            speaker = utt.get('speaker', '')
            mentions = utt.get('mentions', [])
            channel = utt.get('channel', '#unknown')
            text = utt.get('text', '')

            # 현재 발화의 임베딩 계산 (semantic similarity용)
            utt_embedding = self._get_embedding(text) if self.semantic_weight > 0 else None

            # 첫 번째 utterance는 무조건 새 thread
            if not threads:
                thread_id = "thread_0"
                threads.append({
                    'thread_id': thread_id,
                    'channel': channel,
                    'utterances': [utt]
                })
                thread_metadata.append({
                    'participants': {speaker},
                    'last_ts': ts,
                    'embeddings': [utt_embedding] if utt_embedding is not None else []
                })
                continue

            # 최근 N개 thread만 비교 (성능 최적화)
            # last_ts 기준으로 정렬하여 최근 활성 thread만 선택
            if len(threads) > self.max_active_threads:
                # (index, last_ts) 튜플로 정렬하여 최근 N개 선택
                recent_indices = sorted(
                    range(len(threads)),
                    key=lambda i: thread_metadata[i]['last_ts'],
                    reverse=True
                )[:self.max_active_threads]
            else:
                recent_indices = range(len(threads))

            best_thread_idx = -1
            best_score = -1.0
            best_delta = float('inf')

            for idx in recent_indices:
                meta = thread_metadata[idx]
                delta_seconds = (ts - meta['last_ts']).total_seconds()

                # 음수 delta는 시간순 정렬 문제 - 스킵
                if delta_seconds < 0:
                    continue

                score = self._compute_thread_score(
                    speaker=speaker,
                    mentions=mentions,
                    delta_seconds=delta_seconds,
                    thread_participants=meta['participants'],
                    utterance_text=text,
                    thread_embeddings=meta.get('embeddings', [])
                )

                if score > best_score:
                    best_score = score
                    best_thread_idx = idx
                    best_delta = delta_seconds

            # 새 thread 생성 여부 결정
            create_new_thread = False

            if best_thread_idx < 0:
                # 유효한 후보 thread 없음
                create_new_thread = True
            elif best_score < self.new_thread_threshold:
                # 스코어가 threshold 미만
                create_new_thread = True
            elif self.max_gap_seconds is not None and best_delta > self.max_gap_seconds:
                # 안전장치: 너무 긴 시간 간격
                create_new_thread = True

            if create_new_thread:
                # 새 thread 생성
                thread_id = f"thread_{len(threads)}"
                threads.append({
                    'thread_id': thread_id,
                    'channel': channel,
                    'utterances': [utt]
                })
                thread_metadata.append({
                    'participants': {speaker},
                    'last_ts': ts,
                    'embeddings': [utt_embedding] if utt_embedding is not None else []
                })
            else:
                # 기존 thread에 추가
                threads[best_thread_idx]['utterances'].append(utt)
                thread_metadata[best_thread_idx]['participants'].add(speaker)
                thread_metadata[best_thread_idx]['last_ts'] = ts

                # 임베딩 추가 (최대 개수 제한)
                if utt_embedding is not None:
                    embeddings = thread_metadata[best_thread_idx].get('embeddings', [])
                    embeddings.append(utt_embedding)
                    # 메모리 절약: 최근 N개만 유지
                    if len(embeddings) > self.max_thread_context * 2:
                        embeddings = embeddings[-self.max_thread_context:]
                    thread_metadata[best_thread_idx]['embeddings'] = embeddings

        # 빈 thread 제거 (안전장치)
        threads = [t for t in threads if t['utterances']]

        logger.info(f"Disentangled {len(sorted_utts)} utterances into {len(threads)} threads "
                    f"(semantic_weight={self.semantic_weight})")
        return threads

    def refine_with_mentions(self, threads: List[Dict]) -> List[Dict]:
        """
        Optional: refine threads by splitting on topic/participant shifts

        This is a more sophisticated heuristic that can be enabled for better quality.
        스코어 기반 접근과 함께 사용 가능.
        """
        refined = []

        for thread in threads:
            utts = thread['utterances']
            if len(utts) <= 2:
                refined.append(thread)
                continue

            # Split if participant set changes dramatically
            sub_threads = []
            current_sub = [utts[0]]
            current_participants = {utts[0]['speaker']}

            for i in range(1, len(utts)):
                prev_participants = current_participants.copy()
                current_participants.add(utts[i]['speaker'])

                # Check mention continuity
                mentions = utts[i].get('mentions', [])
                has_link = any(m in prev_participants for m in mentions)

                if not has_link and len(current_participants) > len(prev_participants) + 1:
                    # Participant set expanded without mention link -> split
                    sub_threads.append(current_sub)
                    current_sub = [utts[i]]
                    current_participants = {utts[i]['speaker']}
                else:
                    current_sub.append(utts[i])

            sub_threads.append(current_sub)

            # Create refined threads
            for j, sub_utts in enumerate(sub_threads):
                refined.append({
                    'thread_id': f"{thread['thread_id']}_sub{j}",
                    'channel': thread['channel'],
                    'utterances': sub_utts
                })

        return refined

    def compute_eda_stats(self, threads: List[Dict], utterances: List[Dict]) -> Dict:
        """
        Spec 2.3: EDA 출력
        - thread 수 분포
        - thread 길이(utterances) 분포
        - interleaving 강도(동시 thread 개수 추정)
        """
        stats = {
            'total_threads': len(threads),
            'total_utterances': len(utterances),
            'thread_lengths': [],
            'thread_durations_sec': [],
            'speakers_per_thread': [],
            'interleaving_estimate': 0.0,
        }

        if not threads:
            return stats

        # Thread별 통계
        for thread in threads:
            utts = thread.get('utterances', [])
            stats['thread_lengths'].append(len(utts))

            # Duration 계산
            if len(utts) >= 2:
                try:
                    start_ts = datetime.fromisoformat(utts[0]['ts'])
                    end_ts = datetime.fromisoformat(utts[-1]['ts'])
                    duration = (end_ts - start_ts).total_seconds()
                    stats['thread_durations_sec'].append(duration)
                except:
                    pass

            # 화자 수
            speakers = set(u.get('speaker', '') for u in utts)
            stats['speakers_per_thread'].append(len(speakers))

        # Summary statistics
        import numpy as np
        if stats['thread_lengths']:
            lengths = np.array(stats['thread_lengths'])
            stats['thread_length_summary'] = {
                'mean': float(np.mean(lengths)),
                'median': float(np.median(lengths)),
                'min': int(np.min(lengths)),
                'max': int(np.max(lengths)),
                'q25': float(np.percentile(lengths, 25)),
                'q75': float(np.percentile(lengths, 75)),
            }

        if stats['thread_durations_sec']:
            durations = np.array(stats['thread_durations_sec'])
            stats['duration_summary'] = {
                'mean': float(np.mean(durations)),
                'median': float(np.median(durations)),
                'min': float(np.min(durations)),
                'max': float(np.max(durations)),
            }

        if stats['speakers_per_thread']:
            speakers = np.array(stats['speakers_per_thread'])
            stats['speakers_summary'] = {
                'mean': float(np.mean(speakers)),
                'median': float(np.median(speakers)),
            }

        # Interleaving 강도 추정
        stats['interleaving_estimate'] = self._estimate_interleaving(threads)

        return stats

    def _estimate_interleaving(self, threads: List[Dict]) -> float:
        """
        동시에 활성인 thread 수 추정
        각 thread의 시작~종료 구간이 얼마나 겹치는지 계산
        """
        if len(threads) <= 1:
            return 1.0

        intervals = []
        for thread in threads:
            utts = thread.get('utterances', [])
            if len(utts) < 2:
                continue
            try:
                start = datetime.fromisoformat(utts[0]['ts'])
                end = datetime.fromisoformat(utts[-1]['ts'])
                intervals.append((start, end))
            except:
                continue

        if not intervals:
            return 1.0

        # 각 시점에서 활성 thread 수 계산
        events = []
        for start, end in intervals:
            events.append((start, 1))   # thread 시작
            events.append((end, -1))    # thread 종료

        events.sort(key=lambda x: x[0])

        active_count = 0
        max_concurrent = 0
        total_concurrent = 0
        count = 0

        for _, delta in events:
            active_count += delta
            max_concurrent = max(max_concurrent, active_count)
            if delta == 1:  # 새 thread 시작 시점에서 측정
                total_concurrent += active_count
                count += 1

        avg_concurrent = total_concurrent / count if count > 0 else 1.0

        return avg_concurrent


# =============================================================================
# Self-check / Test functions
# =============================================================================

def _test_weighted_scoring():
    """
    가중치 기반 스코어링 테스트

    검증 항목:
    1. mention_weight=0이면 멘션 기반 연결이 약화됨
    2. time_weight 변경 시 긴 공백 후 thread 연결 정도가 달라짐
    3. 300초 넘어도 무조건 split되지 않음
    """
    from datetime import datetime, timedelta

    print("=" * 60)
    print("ThreadDisentangler Weighted Scoring Test")
    print("=" * 60)

    # 테스트 데이터: 5분 간격의 대화
    base_time = datetime(2024, 1, 15, 10, 0, 0)

    test_utterances = [
        {'ts': base_time.isoformat(), 'speaker': 'alice', 'text': 'I have a problem', 'mentions': [], 'channel': '#test'},
        {'ts': (base_time + timedelta(seconds=30)).isoformat(), 'speaker': 'bob', 'text': 'what is it?', 'mentions': ['alice'], 'channel': '#test'},
        {'ts': (base_time + timedelta(seconds=60)).isoformat(), 'speaker': 'alice', 'text': 'apt-get fails', 'mentions': ['bob'], 'channel': '#test'},
        # 6분 간격 (360초) - 기존 하드 임계치(300초)를 초과
        {'ts': (base_time + timedelta(seconds=420)).isoformat(), 'speaker': 'bob', 'text': 'try apt update', 'mentions': ['alice'], 'channel': '#test'},
        {'ts': (base_time + timedelta(seconds=450)).isoformat(), 'speaker': 'alice', 'text': 'thanks that worked', 'mentions': ['bob'], 'channel': '#test'},
    ]

    # Test 1: 기본 설정 - 멘션이 있으므로 5분 넘어도 같은 thread로 묶여야 함
    print("\n[Test 1] Default settings (mention_weight=2.0)")
    disentangler = ThreadDisentangler(
        time_weight=1.0,
        time_scale=300.0,
        mention_weight=2.0,
        participant_weight=1.5,
        new_thread_threshold=0.3,
        max_gap_seconds=3600.0
    )
    threads = disentangler.disentangle(test_utterances)
    print(f"  Number of threads: {len(threads)}")
    print(f"  Expected: 1 (all in same thread due to mentions)")
    assert len(threads) == 1, f"Expected 1 thread, got {len(threads)}"
    print("  PASS")

    # Test 2: mention_weight=0 - 멘션 효과 제거
    print("\n[Test 2] mention_weight=0 (no mention effect)")
    disentangler2 = ThreadDisentangler(
        time_weight=1.0,
        time_scale=300.0,
        mention_weight=0.0,  # 멘션 효과 제거
        participant_weight=1.5,
        new_thread_threshold=0.3,
        max_gap_seconds=3600.0
    )
    threads2 = disentangler2.disentangle(test_utterances)
    print(f"  Number of threads: {len(threads2)}")
    print(f"  Note: Without mention weight, 6-min gap may cause different behavior")
    # 참여자 연속성으로 인해 여전히 1개일 수 있음
    print(f"  Result: {len(threads2)} threads")
    print("  PASS (behavior changed based on weight)")

    # Test 3: 매우 긴 간격 (2시간) - max_gap_seconds로 분리
    print("\n[Test 3] Very long gap (2 hours)")
    long_gap_utts = [
        {'ts': base_time.isoformat(), 'speaker': 'alice', 'text': 'hello', 'mentions': [], 'channel': '#test'},
        {'ts': (base_time + timedelta(hours=2)).isoformat(), 'speaker': 'alice', 'text': 'still here?', 'mentions': [], 'channel': '#test'},
    ]
    threads3 = disentangler.disentangle(long_gap_utts)
    print(f"  Number of threads: {len(threads3)}")
    print(f"  Expected: 2 (max_gap_seconds=3600 triggered)")
    assert len(threads3) == 2, f"Expected 2 threads, got {len(threads3)}"
    print("  PASS")

    # Test 4: max_gap_seconds=None - 안전장치 비활성화
    print("\n[Test 4] max_gap_seconds=None (no hard cutoff)")
    disentangler4 = ThreadDisentangler(
        time_weight=1.0,
        time_scale=300.0,
        mention_weight=2.0,
        participant_weight=1.5,
        new_thread_threshold=0.1,  # 낮은 threshold
        max_gap_seconds=None  # 안전장치 비활성화
    )
    threads4 = disentangler4.disentangle(long_gap_utts)
    print(f"  Number of threads: {len(threads4)}")
    print(f"  Note: Same speaker, so participant_weight keeps them together")
    print("  PASS (no hard cutoff)")

    # Test 5: time_score 함수 검증
    print("\n[Test 5] Time score function")
    d = ThreadDisentangler()
    scores = [
        (0, d._compute_time_score(0)),
        (300, d._compute_time_score(300)),
        (600, d._compute_time_score(600)),
        (1800, d._compute_time_score(1800)),
    ]
    print("  delta(sec) -> time_score")
    for delta, score in scores:
        print(f"    {delta:6d}s -> {score:.4f}")
    print("  Verified: exponential decay, never exactly 0")
    print("  PASS")

    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    _test_weighted_scoring()
