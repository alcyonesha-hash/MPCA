"""
Embedding-based Topic Tracker (Unsupervised)

알고리즘 정의:
- Topic vector = mean of embeddings in sliding window
- relevance = cosine(e(x), topic_vector)
- drift if relevance < drift_threshold

설계 원칙:
- TF-IDF refit과 anchor boost를 제거함
- 이유: 단순화, 재현성, 안정성, heuristic 제거
- 고정된 사전학습 임베딩 모델을 사용하여 일관된 벡터 표현 획득
- sklearn 의존성 제거, sentence-transformers 또는 transformers로 동작

Thread-based Topical Fit (v2.1):
- LLM 응답이 올바른 thread에 속하는지 검증
- Disentanglement에서 분류된 thread vector와 응답을 비교
- LLM이 여러 thread가 섞인 context에서 잘못된 thread에 응답하는 것을 감지

호환성:
- metrics.py의 기존 인터페이스와 완전 호환
- update(), reset(), compute_cosine(), compute_relevance(),
  has_anchor_boost(), is_topic_shift() 메서드 시그니처 유지
- 새 메서드: compute_thread_fit(), set_thread_vector()
"""

import numpy as np
from typing import Dict, Optional, Tuple, List
from collections import deque
from functools import lru_cache
import logging

logger = logging.getLogger(__name__)


class TopicTracker:
    """
    Track current topic using sliding window of sentence embeddings.

    핵심 변경사항 (vs 이전 TF-IDF 버전):
    1. TF-IDF refit 제거 → 고정 임베딩 모델 사용
    2. anchor boost 제거 → 순수 cosine similarity만 사용
    3. 캐싱으로 동일 텍스트 재계산 방지

    임베딩 우선순위:
    1. sentence-transformers (all-MiniLM-L6-v2)
    2. transformers mean-pooling (fallback)
    """

    def __init__(
        self,
        window_size: int = 7,
        drift_threshold: float = 0.30,
        embedding_model: str = "auto"
    ):
        """
        Args:
            window_size: Number of recent utterances for topic vector (k)
                - Default 7: Data-derived from filtered thread length median (7 utterances).
            drift_threshold: Cosine threshold below which is topic shift (tau)
                - Default 0.30: Data-derived optimal threshold (F1=85.3)
                  from intra-thread Q25 (0.352) and inter-thread Q75 (0.244).
            embedding_model: "auto", "sentence-transformers", or "hf-meanpool"
        """
        self.window_size = window_size
        self.drift_threshold = drift_threshold
        self.embedding_model_name = embedding_model

        # Sliding window: stores (text, embedding) tuples
        self.window: deque = deque(maxlen=window_size)
        self.current_topic_vec: Optional[np.ndarray] = None

        # Embedding cache (LRU)
        self._embedding_cache: Dict[str, np.ndarray] = {}
        self._cache_max_size = 1000

        # Initialize embedding model
        self._model = None
        self._tokenizer = None
        self._device = None
        self._model_type = None  # "sentence-transformers" or "hf-meanpool"

        # Thread-based topical fit (v2.1)
        self._thread_vector: Optional[np.ndarray] = None
        self._thread_id: Optional[str] = None

        self._init_embedding_model(embedding_model)

    def _init_embedding_model(self, model_preference: str):
        """
        Initialize embedding model with fallback.

        Priority:
        1. sentence-transformers (if available and preferred)
        2. transformers mean-pooling (fallback)
        """
        if model_preference in ("auto", "sentence-transformers"):
            if self._try_init_sentence_transformers():
                return

        if model_preference in ("auto", "hf-meanpool"):
            if self._try_init_hf_meanpool():
                return

        # Last resort: try sentence-transformers even if hf-meanpool was preferred
        if model_preference == "hf-meanpool":
            if self._try_init_sentence_transformers():
                return

        raise RuntimeError(
            "Failed to initialize embedding model. "
            "Install sentence-transformers or transformers+torch."
        )

    def _try_init_sentence_transformers(self) -> bool:
        """Try to initialize sentence-transformers model."""
        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer('all-MiniLM-L6-v2')
            self._model_type = "sentence-transformers"
            logger.info("TopicTracker: Using sentence-transformers (all-MiniLM-L6-v2)")
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

            # Try to use same model family as sentence-transformers
            model_name = "sentence-transformers/all-MiniLM-L6-v2"
            try:
                self._tokenizer = AutoTokenizer.from_pretrained(model_name)
                self._model = AutoModel.from_pretrained(model_name)
            except Exception:
                # Fallback to distilbert
                model_name = "distilbert-base-uncased"
                self._tokenizer = AutoTokenizer.from_pretrained(model_name)
                self._model = AutoModel.from_pretrained(model_name)
                logger.info(f"TopicTracker: Using fallback model {model_name}")

            # Set device
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self._model = self._model.to(self._device)
            self._model.eval()

            self._model_type = "hf-meanpool"
            logger.info(f"TopicTracker: Using transformers mean-pooling ({model_name}) on {self._device}")
            return True
        except ImportError:
            logger.debug("transformers or torch not available")
            return False
        except Exception as e:
            logger.warning(f"Failed to load transformers model: {e}")
            return False

    def _get_embedding(self, text: str) -> np.ndarray:
        """
        Get embedding for text with caching.

        Returns:
            L2-normalized embedding vector
        """
        # Check cache
        if text in self._embedding_cache:
            return self._embedding_cache[text]

        # Compute embedding
        if self._model_type == "sentence-transformers":
            embedding = self._model.encode(text, convert_to_numpy=True)
        elif self._model_type == "hf-meanpool":
            embedding = self._compute_hf_meanpool_embedding(text)
        else:
            raise RuntimeError("No embedding model initialized")

        # L2 normalize for stable cosine computation
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        # Cache management (simple LRU-like)
        if len(self._embedding_cache) >= self._cache_max_size:
            # Remove oldest entries (first 10%)
            keys_to_remove = list(self._embedding_cache.keys())[:self._cache_max_size // 10]
            for key in keys_to_remove:
                del self._embedding_cache[key]

        self._embedding_cache[text] = embedding
        return embedding

    def _compute_hf_meanpool_embedding(self, text: str) -> np.ndarray:
        """
        Compute embedding using HuggingFace transformers with mean-pooling.

        Mean-pooling: average of last_hidden_state weighted by attention_mask
        """
        import torch

        # Tokenize
        inputs = self._tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True
        )
        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        # Forward pass
        with torch.no_grad():
            outputs = self._model(**inputs)
            last_hidden = outputs.last_hidden_state  # [1, seq_len, hidden_dim]
            attention_mask = inputs["attention_mask"]  # [1, seq_len]

        # Mean pooling
        mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden.size()).float()
        sum_embeddings = torch.sum(last_hidden * mask_expanded, dim=1)
        sum_mask = mask_expanded.sum(dim=1).clamp(min=1e-9)
        mean_pooled = sum_embeddings / sum_mask

        return mean_pooled.cpu().numpy().flatten()

    def update(self, utterance: Dict):
        """
        Add utterance to sliding window and update topic vector.

        Args:
            utterance: Dict with 'text' key
        """
        text = utterance.get('text', '')
        if not text or not text.strip():
            return

        # Get embedding (cached)
        embedding = self._get_embedding(text)

        # Add to window
        self.window.append((text, embedding))

        # Update topic vector
        if len(self.window) >= 1:
            self._update_topic_vector()

    def _update_topic_vector(self):
        """
        Compute topic vector as mean of window embeddings.

        Topic vector = (1/k) * sum(embeddings in window)
        Then L2 normalized for stable cosine computation.
        """
        if not self.window:
            self.current_topic_vec = None
            return

        embeddings = np.array([emb for _, emb in self.window])
        mean_vec = np.mean(embeddings, axis=0)

        # L2 normalize
        norm = np.linalg.norm(mean_vec)
        if norm > 0:
            mean_vec = mean_vec / norm

        self.current_topic_vec = mean_vec

    def reset(self):
        """
        Reset topic tracker state (e.g., for new thread).

        Note: Embedding cache is preserved for efficiency.
        """
        self.window.clear()
        self.current_topic_vec = None

    def compute_cosine(self, utterance_text: str) -> float:
        """
        Compute raw cosine similarity between utterance and topic vector.

        Returns:
            Cosine similarity clamped to [0, 1].
            Returns 1.0 if no topic vector yet (initial state = everything relevant).
        """
        # Initial state: no topic yet, everything is relevant
        if self.current_topic_vec is None or len(self.window) < 2:
            return 1.0

        if not utterance_text or not utterance_text.strip():
            return 0.5  # Fallback for empty text

        # Get embedding (normalized)
        utt_embedding = self._get_embedding(utterance_text)

        # Cosine similarity (both vectors are L2 normalized, so dot product = cosine)
        cosine = float(np.dot(self.current_topic_vec, utt_embedding))

        # Clamp to [0, 1] for consistency with previous behavior
        return max(0.0, cosine)

    def compute_relevance(self, utterance_text: str) -> float:
        """
        Compute topical relevance of utterance.

        Note: With anchor boost removed, this is identical to compute_cosine().
        Kept as separate method for API compatibility with metrics.py.

        Returns:
            Relevance score in [0, 1], where 1 = highly relevant.
        """
        # No anchor boost, relevance == cosine
        return self.compute_cosine(utterance_text)

    def has_anchor_boost(self, utterance_text: str) -> bool:
        """
        Check if utterance would receive anchor boost.

        Note: Anchor boost has been removed for simplicity.
        Always returns False for API compatibility.
        """
        return False

    def is_topic_shift(self, utterance_text: str) -> bool:
        """
        Check if utterance represents a topic shift (drift).

        Topic shift is detected when relevance < drift_threshold.

        Returns:
            True if utterance is off-topic (drift detected).
        """
        relevance = self.compute_relevance(utterance_text)
        return relevance < self.drift_threshold

    # =========================================================================
    # Thread-based Topical Fit (v2.1)
    # =========================================================================

    def compute_thread_vector(self, utterances: List[Dict]) -> np.ndarray:
        """
        Thread의 발화들로부터 thread vector를 계산.

        Args:
            utterances: Thread에 속한 발화 리스트 [{'text': ...}, ...]

        Returns:
            L2-normalized thread vector (384차원)
        """
        if not utterances:
            return None

        embeddings = []
        for utt in utterances:
            text = utt.get('text', '') if isinstance(utt, dict) else str(utt)
            if text and text.strip():
                emb = self._get_embedding(text)
                embeddings.append(emb)

        if not embeddings:
            return None

        # Mean pooling
        thread_vec = np.mean(embeddings, axis=0)

        # L2 normalize
        norm = np.linalg.norm(thread_vec)
        if norm > 0:
            thread_vec = thread_vec / norm

        return thread_vec

    def set_thread_vector(self, thread_vector: np.ndarray, thread_id: str = None):
        """
        현재 평가 대상 thread의 vector를 설정.

        Args:
            thread_vector: 미리 계산된 thread vector
            thread_id: Thread 식별자 (디버깅용)
        """
        self._thread_vector = thread_vector
        self._thread_id = thread_id

    def compute_thread_fit(self, response_text: str) -> Tuple[float, bool]:
        """
        LLM 응답이 현재 thread에 맞는지 평가.

        이것이 핵심 메트릭: LLM이 여러 thread가 섞인 context에서
        올바른 thread에 응답하고 있는지 확인.

        Args:
            response_text: LLM이 생성한 응답

        Returns:
            (cosine_similarity, is_correct_thread)
            - cosine_similarity: 응답과 thread vector의 유사도 [0, 1]
            - is_correct_thread: drift_threshold 이상이면 True
        """
        if self._thread_vector is None:
            # Thread vector가 설정되지 않으면 기존 방식 사용
            relevance = self.compute_relevance(response_text)
            return relevance, relevance >= self.drift_threshold

        if not response_text or not response_text.strip():
            return 0.0, False

        # 응답 임베딩 계산
        response_emb = self._get_embedding(response_text)

        # Thread vector와 cosine similarity
        cosine = float(np.dot(self._thread_vector, response_emb))
        cosine = max(0.0, cosine)  # Clamp to [0, 1]

        is_correct = cosine >= self.drift_threshold

        return cosine, is_correct

    def compute_thread_fit_multi(
        self,
        response_text: str,
        thread_vectors: Dict[str, np.ndarray]
    ) -> Dict[str, float]:
        """
        응답이 여러 thread 중 어느 것에 가장 가까운지 계산.

        Args:
            response_text: LLM 응답
            thread_vectors: {thread_id: vector} 딕셔너리

        Returns:
            {thread_id: cosine_similarity} 딕셔너리
        """
        if not response_text or not response_text.strip():
            return {tid: 0.0 for tid in thread_vectors}

        response_emb = self._get_embedding(response_text)

        similarities = {}
        for thread_id, thread_vec in thread_vectors.items():
            if thread_vec is not None:
                cosine = float(np.dot(thread_vec, response_emb))
                similarities[thread_id] = max(0.0, cosine)
            else:
                similarities[thread_id] = 0.0

        return similarities

    def check_thread_confusion(
        self,
        response_text: str,
        target_thread_id: str,
        all_thread_vectors: Dict[str, np.ndarray]
    ) -> Dict:
        """
        LLM이 thread를 혼동했는지 검사.

        Args:
            response_text: LLM 응답
            target_thread_id: 응답해야 할 thread ID
            all_thread_vectors: 현재 활성화된 모든 thread vectors

        Returns:
            {
                'target_similarity': float,  # 타겟 thread와의 유사도
                'max_other_similarity': float,  # 다른 thread 중 최대 유사도
                'confused_with': str or None,  # 혼동된 thread ID
                'is_confused': bool,  # 혼동 여부
            }
        """
        similarities = self.compute_thread_fit_multi(response_text, all_thread_vectors)

        target_sim = similarities.get(target_thread_id, 0.0)

        # 다른 thread 중 최대 유사도
        other_sims = {
            tid: sim for tid, sim in similarities.items()
            if tid != target_thread_id
        }

        if other_sims:
            max_other_tid = max(other_sims, key=other_sims.get)
            max_other_sim = other_sims[max_other_tid]
        else:
            max_other_tid = None
            max_other_sim = 0.0

        # 혼동 판정: 다른 thread 유사도가 타겟보다 높으면 혼동
        is_confused = max_other_sim > target_sim and max_other_sim >= self.drift_threshold

        return {
            'target_similarity': target_sim,
            'max_other_similarity': max_other_sim,
            'confused_with': max_other_tid if is_confused else None,
            'is_confused': is_confused,
            'all_similarities': similarities,
        }

    # =========================================================================
    # Primary Thread Detection (v2.2) - 실시간 응답 결정용
    # =========================================================================

    def find_primary_thread(
        self,
        utterance_text: str,
        thread_vectors: Dict[str, np.ndarray],
        thread_metadata: Dict[str, Dict],
        agent_name: str = "agent",
        weights: Dict[str, float] = None
    ) -> Dict:
        """
        최근 활성 thread 중 "주요 thread"를 판단.

        세 가지 기준을 결합:
        1. 시간 근접도 (recency): 최근 활성화된 thread
        2. 에이전트 참여도 (participation): 에이전트가 참여 중인 thread
        3. 의미 유사도 (similarity): 새 발화와 가장 유사한 thread

        Args:
            utterance_text: 새로 들어온 발화
            thread_vectors: {thread_id: np.ndarray} thread 벡터들
            thread_metadata: {thread_id: {'last_ts': datetime, 'participants': set, ...}}
            agent_name: 에이전트 이름 (참여도 계산용)
            weights: 가중치 {'recency': α, 'participation': β, 'similarity': γ}
                     기본값: {'recency': 0.3, 'participation': 0.3, 'similarity': 0.4}

        Returns:
            {
                'primary_thread_id': str,  # 주요 thread ID
                'primary_score': float,  # 주요 thread 스코어
                'similarity_to_primary': float,  # 발화와 주요 thread의 유사도
                'should_respond': bool,  # 응답 여부 (similarity >= threshold)
                'all_scores': Dict[str, Dict],  # 모든 thread별 상세 스코어
            }
        """
        if weights is None:
            weights = {
                'recency': 0.3,
                'participation': 0.3,
                'similarity': 0.4
            }

        if not thread_vectors:
            return {
                'primary_thread_id': None,
                'primary_score': 0.0,
                'similarity_to_primary': 0.0,
                'should_respond': False,
                'all_scores': {},
            }

        # 발화 임베딩 계산
        utt_embedding = self._get_embedding(utterance_text) if utterance_text.strip() else None

        # 시간 정규화를 위한 최신 timestamp 찾기
        latest_ts = None
        for tid, meta in thread_metadata.items():
            if tid in thread_vectors:
                ts = meta.get('last_ts')
                if ts and (latest_ts is None or ts > latest_ts):
                    latest_ts = ts

        all_scores = {}
        best_thread_id = None
        best_total_score = -1.0

        for thread_id, thread_vec in thread_vectors.items():
            if thread_vec is None:
                continue

            meta = thread_metadata.get(thread_id, {})

            # 1. 시간 근접도 (recency): exp(-delta / scale)
            recency_score = 0.0
            if latest_ts and meta.get('last_ts'):
                delta_seconds = (latest_ts - meta['last_ts']).total_seconds()
                # time_scale = 300초 (5분) 기준
                recency_score = np.exp(-delta_seconds / 300.0)
            else:
                recency_score = 0.5  # 시간 정보 없으면 중간값

            # 2. 에이전트 참여도 (participation)
            participation_score = 0.0
            participants = meta.get('participants', set())
            if agent_name in participants:
                participation_score = 1.0
            # 에이전트가 참여 안 했으면 0.0

            # 3. 의미 유사도 (similarity)
            similarity_score = 0.0
            if utt_embedding is not None:
                similarity_score = float(np.dot(thread_vec, utt_embedding))
                similarity_score = max(0.0, similarity_score)

            # 종합 스코어
            total_score = (
                weights['recency'] * recency_score +
                weights['participation'] * participation_score +
                weights['similarity'] * similarity_score
            )

            all_scores[thread_id] = {
                'recency': recency_score,
                'participation': participation_score,
                'similarity': similarity_score,
                'total': total_score,
            }

            if total_score > best_total_score:
                best_total_score = total_score
                best_thread_id = thread_id

        # 주요 thread와의 유사도 확인
        similarity_to_primary = 0.0
        if best_thread_id and best_thread_id in all_scores:
            similarity_to_primary = all_scores[best_thread_id]['similarity']

        # 응답 여부: 주요 thread와의 유사도가 threshold 이상
        should_respond = similarity_to_primary >= self.drift_threshold

        return {
            'primary_thread_id': best_thread_id,
            'primary_score': best_total_score,
            'similarity_to_primary': similarity_to_primary,
            'should_respond': should_respond,
            'all_scores': all_scores,
        }

    def should_respond_to_utterance(
        self,
        utterance_text: str,
        thread_vectors: Dict[str, np.ndarray],
        thread_metadata: Dict[str, Dict],
        agent_name: str = "agent",
        weights: Dict[str, float] = None
    ) -> Tuple[bool, str, float]:
        """
        새 발화에 응답해야 하는지 결정 (간단한 인터페이스).

        Args:
            utterance_text: 새로 들어온 발화
            thread_vectors: {thread_id: np.ndarray}
            thread_metadata: {thread_id: {'last_ts': datetime, 'participants': set}}
            agent_name: 에이전트 이름
            weights: 가중치 (optional)

        Returns:
            (should_respond, primary_thread_id, similarity)
        """
        result = self.find_primary_thread(
            utterance_text, thread_vectors, thread_metadata,
            agent_name, weights
        )
        return (
            result['should_respond'],
            result['primary_thread_id'],
            result['similarity_to_primary']
        )


# =============================================================================
# Self-test function
# =============================================================================

def _self_test():
    """
    Self-test for TopicTracker.

    Tests:
    1. Install-related context → install-related query should have high relevance
    2. Install-related context → unrelated query should have lower relevance
    3. reset() should restore initial state (relevance = 1.0)
    4. has_anchor_boost() should always return False
    """
    print("=" * 60)
    print("TopicTracker Self-Test (Embedding-based)")
    print("=" * 60)

    tracker = TopicTracker(window_size=5, drift_threshold=0.3)
    print(f"\nModel type: {tracker._model_type}")
    print(f"Window size: {tracker.window_size}")
    print(f"Drift threshold: {tracker.drift_threshold}")

    # Test 1: Initial state
    print("\n[Test 1] Initial state (no context)")
    relevance = tracker.compute_relevance("any random text")
    print(f"  Relevance: {relevance:.4f}")
    assert relevance == 1.0, f"Expected 1.0 in initial state, got {relevance}"
    print("  PASS: Initial relevance is 1.0")

    # Test 2: Build install-related context
    print("\n[Test 2] Build install-related context")
    install_context = [
        {"text": "I'm trying to install python packages on Ubuntu"},
        {"text": "use pip install or apt-get install for system packages"},
        {"text": "make sure you have the right dependencies installed"},
    ]
    for utt in install_context:
        tracker.update(utt)
        print(f"  Added: '{utt['text'][:50]}...'")

    # Test 3: Install-related query should have high relevance
    print("\n[Test 3] Install-related query")
    install_query = "how do I install numpy using pip?"
    relevance_install = tracker.compute_relevance(install_query)
    print(f"  Query: '{install_query}'")
    print(f"  Relevance: {relevance_install:.4f}")
    # Should be reasonably high (> 0.3 at least for related content)
    print(f"  Expected: > 0.3 (related to install topic)")

    # Test 4: Unrelated query should have lower relevance
    print("\n[Test 4] Unrelated query")
    unrelated_query = "what time is the football game tonight?"
    relevance_unrelated = tracker.compute_relevance(unrelated_query)
    print(f"  Query: '{unrelated_query}'")
    print(f"  Relevance: {relevance_unrelated:.4f}")
    print(f"  Expected: < relevance of install query ({relevance_install:.4f})")

    # Compare
    if relevance_install > relevance_unrelated:
        print("  PASS: Install query more relevant than unrelated query")
    else:
        print("  WARNING: Expected install query to be more relevant")

    # Test 5: Topic shift detection
    print("\n[Test 5] Topic shift detection")
    is_shift_install = tracker.is_topic_shift(install_query)
    is_shift_unrelated = tracker.is_topic_shift(unrelated_query)
    print(f"  Install query is_topic_shift: {is_shift_install}")
    print(f"  Unrelated query is_topic_shift: {is_shift_unrelated}")
    print(f"  (drift_threshold = {tracker.drift_threshold})")

    # Test 6: has_anchor_boost always False
    print("\n[Test 6] has_anchor_boost() always returns False")
    boost1 = tracker.has_anchor_boost("apt-get install python")
    boost2 = tracker.has_anchor_boost("random text without keywords")
    print(f"  'apt-get install python': {boost1}")
    print(f"  'random text': {boost2}")
    assert boost1 == False and boost2 == False, "has_anchor_boost should always be False"
    print("  PASS: has_anchor_boost always returns False")

    # Test 7: reset() restores initial state
    print("\n[Test 7] reset() restores initial state")
    tracker.reset()
    relevance_after_reset = tracker.compute_relevance("any text after reset")
    print(f"  Relevance after reset: {relevance_after_reset:.4f}")
    assert relevance_after_reset == 1.0, f"Expected 1.0 after reset, got {relevance_after_reset}"
    print("  PASS: reset() restores initial state (relevance = 1.0)")

    # Test 8: Different drift thresholds
    print("\n[Test 8] Drift threshold sensitivity")
    tracker_strict = TopicTracker(window_size=5, drift_threshold=0.5)
    tracker_loose = TopicTracker(window_size=5, drift_threshold=0.1)

    for utt in install_context:
        tracker_strict.update(utt)
        tracker_loose.update(utt)

    test_query = "network configuration for wifi adapter"
    shift_strict = tracker_strict.is_topic_shift(test_query)
    shift_loose = tracker_loose.is_topic_shift(test_query)
    rel = tracker_strict.compute_relevance(test_query)

    print(f"  Query: '{test_query}'")
    print(f"  Relevance: {rel:.4f}")
    print(f"  is_topic_shift (threshold=0.5): {shift_strict}")
    print(f"  is_topic_shift (threshold=0.1): {shift_loose}")
    print("  PASS: Drift threshold affects topic shift detection")

    print("\n" + "=" * 60)
    print("All self-tests completed!")
    print("=" * 60)


if __name__ == "__main__":
    _self_test()
