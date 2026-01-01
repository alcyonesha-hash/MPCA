# Ubuntu IRC Behavioral Cues Analysis Report

## Dataset Information
- **Source**: Kummerfeld et al. (ACL 2019) - IRC Disentanglement
- **HuggingFace ID**: `jkkummerfeld/irc_disentangle`
- **Total Messages**: 221,619
- **Total Reply Links**: 60,665
- **Unique Users**: 17,779

---

## 1. Response Latency

Response latency follows a **Log-Normal distribution**.

| Parameter | Value |
|-----------|-------|
| μ (log-scale mean) | 0.2857 |
| σ (log-scale std) | 0.5817 |
| Mean | 2.08 min |
| Median | 1.00 min |
| 25th percentile | 1.00 min |
| 75th percentile | 2.00 min |
| 90th percentile | 3.00 min |
| K-S statistic | 0.4266 |
| K-S p-value | 0.0000 |
| N samples | 31,924 |

**Usage**: `delay = np.random.lognormal(0.286, 0.582)`


---

## 2. Turn Length (Words per Message)

Turn length follows a **Gamma distribution**.

| Parameter | Value |
|-----------|-------|
| Shape (k) | 1.3499 |
| Scale (θ) | 7.6052 |
| Mean | 10.27 words |
| Median | 8.00 words |
| Short (≤5 words) | 37.0% |
| Medium (6-20 words) | 51.3% |
| Long (>20 words) | 11.8% |
| N samples | 221,592 |

**Usage**: `words = int(np.random.gamma(1.350, 7.605))`

---

## 3. Message Chunking

Consecutive messages follow a **Geometric distribution**.

| Parameter | Value |
|-----------|-------|
| p (probability) | 0.8610 |
| Mean chunk size | 1.16 |
| Single message | 87.7% |
| Double (2) | 9.7% |
| Triple (3) | 1.8% |
| 4+ messages | 0.8% |
| **Multi-message rate** | 12.3% |

> **Reference**: Baron (2010) reported ~16% multi-message rate in IM conversations.
> Our finding: **12.3%**

**Usage**: `chunk_size = np.random.geometric(0.8610)`

---

## 4. Typing Imperfection

| Pattern | Rate |
|---------|------|
| Self-corrections (*word) | 0.54% |
| Abbreviations (lol, u, thx, etc.) | 3.92% |
| Ellipsis (...) | 7.43% |

---

## 5. Reactive Behaviors

| Behavior | Rate |
|----------|------|
| Short responses (≤3 words) | 23.88% |
| Acknowledgments (ok, yes, thanks) | 12.36% |
| Backchannels (hmm, oh, huh) | 3.77% |
| Laughter (lol, haha) | 1.18% |
| Emoticons (:), :D, etc.) | 10.02% |
| **Overall Reactive** | 41.88% |

---

## 6. Response Decision

| Metric | Value |
|--------|-------|
| Mention rate (username:) | 29.15% |
| Self-reply rate | 0.00% |
| Avg replies per message | 0.274 |
| Messages that are replies | 27.4% |

---

## Summary: Key Parameters for Human-like Chatbot

```python
import numpy as np

# Response timing (minutes)
delay = np.random.lognormal(mu=0.286, sigma=0.582)

# Message length (words)
word_count = int(np.random.gamma(shape=1.350, scale=7.605))

# Message chunking
chunk_size = np.random.geometric(p=0.8610)

# Reactive behavior probability
reactive_prob = 0.419

# Typing imperfection rates
correction_prob = 0.0054
abbreviation_prob = 0.0392
```

---

## References

1. Kummerfeld, J. K., et al. (2019). "A Large-Scale Corpus for Conversation Disentanglement." ACL.
2. Baron, N. S. (2010). "Discourse Structures in Instant Messaging." Language@Internet.
3. Gnewuch, U., et al. (2018). "Faster is Not Always Better: Understanding the Effect of Dynamic Response Delays in Human-Chatbot Interaction." ECIS.

---

*Generated on 2025-12-28 01:19:38*
