#!/usr/bin/env python3
"""
Ubuntu IRC Disentanglement Dataset Analysis
Extracts 6 behavioral cues for human-like chatbot research
"""

import json
import re
from collections import defaultdict
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import load_dataset

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

print("=" * 60)
print("Ubuntu IRC Disentanglement Dataset Analysis")
print("=" * 60)

# =============================================================================
# 1. Load Dataset
# =============================================================================
print("\n[1/7] Loading dataset from HuggingFace...")
dataset = load_dataset("jkkummerfeld/irc_disentangle", "ubuntu")
print(f"    Dataset loaded successfully!")
print(f"    Train: {len(dataset['train'])} samples")
print(f"    Validation: {len(dataset['validation'])} samples")
print(f"    Test: {len(dataset['test'])} samples")

# =============================================================================
# 2. Parse Messages
# =============================================================================
print("\n[2/7] Parsing messages...")

def parse_message(raw_text):
    """Parse IRC message: [HH:MM] <username> message text"""
    # Pattern: [HH:MM] <username> message
    pattern = r'\[(\d{2}):(\d{2})\]\s*<([^>]+)>\s*(.*)'
    match = re.match(pattern, raw_text)
    if match:
        hour, minute, username, text = match.groups()
        return {
            'hour': int(hour),
            'minute': int(minute),
            'username': username.strip(),
            'text': text.strip()
        }
    return None

# Process all splits
all_messages = []
user_to_indices = defaultdict(list)  # Track where each user's messages are

current_idx = 0
for split_name in ['train', 'validation', 'test']:
    split_data = dataset[split_name]

    for i in range(len(split_data)):
        item = split_data[i]
        date_str = item.get('date', '2004-12-01')
        raw = item['raw']

        parsed = parse_message(raw)
        if parsed:
            # Check for username mention at start (reply pattern)
            mention_match = re.match(r'^(\w+):', parsed['text'])
            mentioned_user = mention_match.group(1) if mention_match else None

            all_messages.append({
                'idx': current_idx,
                'date': date_str,
                'hour': parsed['hour'],
                'minute': parsed['minute'],
                'username': parsed['username'],
                'text': parsed['text'],
                'word_count': len(parsed['text'].split()),
                'char_count': len(parsed['text']),
                'mentioned_user': mentioned_user
            })

            user_to_indices[parsed['username']].append(current_idx)
            current_idx += 1

df = pd.DataFrame(all_messages)
print(f"    Total parsed messages: {len(df)}")
print(f"    Unique users: {df['username'].nunique()}")
print(f"    Date range: {df['date'].min()} to {df['date'].max()}")

# =============================================================================
# 3. Build Reply Links (from username: mentions)
# =============================================================================
print("\n[3/7] Building reply links from mentions...")

reply_links = []
# Group by date for finding recent messages from mentioned user
df_grouped = df.groupby('date')

for date, group in df_grouped:
    group = group.sort_values(['hour', 'minute', 'idx']).reset_index(drop=True)

    # Build index of user's last message
    user_last_msg = {}

    for local_idx, row in group.iterrows():
        username = row['username']
        mentioned = row['mentioned_user']
        global_idx = row['idx']

        # If this message mentions another user, create reply link
        if mentioned and mentioned in user_last_msg and mentioned != username:
            source_idx = user_last_msg[mentioned]
            reply_links.append({
                'source_idx': source_idx,
                'reply_idx': global_idx,
                'source_user': mentioned,
                'reply_user': username
            })

        # Update last message for this user
        user_last_msg[username] = global_idx

df_replies = pd.DataFrame(reply_links)
print(f"    Total reply links (from mentions): {len(df_replies)}")
if len(df_replies) > 0:
    print(f"    Unique replying users: {df_replies['reply_user'].nunique()}")

# =============================================================================
# 4. Analysis Functions
# =============================================================================

def analyze_response_latency(df, df_replies):
    """4.1 Response Latency Analysis - Log-Normal fitting"""
    print("\n[4.1] Analyzing Response Latency...")

    if len(df_replies) == 0:
        print("      No reply links found!")
        return None, np.array([])

    latencies = []
    for _, row in df_replies.iterrows():
        source = df.iloc[row['source_idx']]
        reply = df.iloc[row['reply_idx']]

        # Calculate time difference in minutes
        source_mins = source['hour'] * 60 + source['minute']
        reply_mins = reply['hour'] * 60 + reply['minute']

        # If same date
        if source['date'] == reply['date']:
            diff = reply_mins - source_mins
            if diff >= 0:
                latencies.append(diff)
        else:
            # Cross-day: assume reply is next day
            diff = (24 * 60 - source_mins) + reply_mins
            if 0 < diff < 60 * 12:
                latencies.append(diff)

    latencies = np.array([l for l in latencies if l > 0])

    if len(latencies) < 10:
        print("      Not enough latency samples!")
        return None, latencies

    # Log-Normal fitting
    log_latencies = np.log(latencies)
    mu = np.mean(log_latencies)
    sigma = np.std(log_latencies)

    # K-S test
    try:
        ks_stat, ks_pval = stats.kstest(latencies, 'lognorm', args=(sigma, 0, np.exp(mu)))
    except:
        ks_stat, ks_pval = 0, 0

    result = {
        'distribution': 'lognormal',
        'mu': float(mu),
        'sigma': float(sigma),
        'mean_minutes': float(np.mean(latencies)),
        'median_minutes': float(np.median(latencies)),
        'std_minutes': float(np.std(latencies)),
        'percentile_25': float(np.percentile(latencies, 25)),
        'percentile_75': float(np.percentile(latencies, 75)),
        'percentile_90': float(np.percentile(latencies, 90)),
        'ks_statistic': float(ks_stat),
        'ks_pvalue': float(ks_pval),
        'n_samples': len(latencies)
    }

    print(f"      μ={mu:.3f}, σ={sigma:.3f}")
    print(f"      Mean: {result['mean_minutes']:.1f} min, Median: {result['median_minutes']:.1f} min")
    print(f"      K-S test: stat={ks_stat:.4f}, p={ks_pval:.4f}")

    return result, latencies

def analyze_turn_length(df):
    """4.2 Turn Length Analysis - Gamma fitting"""
    print("\n[4.2] Analyzing Turn Length...")

    word_counts = df['word_count'].values
    word_counts = word_counts[word_counts > 0]

    # Gamma fitting
    shape, loc, scale = stats.gamma.fit(word_counts, floc=0)

    # K-S test
    ks_stat, ks_pval = stats.kstest(word_counts, 'gamma', args=(shape, loc, scale))

    result = {
        'distribution': 'gamma',
        'shape_k': float(shape),
        'scale_theta': float(scale),
        'mean_words': float(np.mean(word_counts)),
        'median_words': float(np.median(word_counts)),
        'std_words': float(np.std(word_counts)),
        'percentile_25': float(np.percentile(word_counts, 25)),
        'percentile_75': float(np.percentile(word_counts, 75)),
        'percentile_90': float(np.percentile(word_counts, 90)),
        'pct_short_le5': float(np.mean(word_counts <= 5) * 100),
        'pct_medium_6_20': float(np.mean((word_counts > 5) & (word_counts <= 20)) * 100),
        'pct_long_gt20': float(np.mean(word_counts > 20) * 100),
        'ks_statistic': float(ks_stat),
        'ks_pvalue': float(ks_pval),
        'n_samples': len(word_counts)
    }

    print(f"      shape(k)={shape:.3f}, scale(θ)={scale:.3f}")
    print(f"      Mean: {result['mean_words']:.1f} words, Median: {result['median_words']:.1f} words")
    print(f"      Short(≤5): {result['pct_short_le5']:.1f}%, Long(>20): {result['pct_long_gt20']:.1f}%")

    return result, word_counts

def analyze_message_chunking(df):
    """4.3 Message Chunking Analysis - Geometric fitting"""
    print("\n[4.3] Analyzing Message Chunking...")

    df_sorted = df.sort_values(['date', 'hour', 'minute', 'idx']).reset_index(drop=True)

    chunks = []
    current_chunk = 1
    prev_user = None
    prev_time = None
    prev_date = None

    for _, row in df_sorted.iterrows():
        curr_time = row['hour'] * 60 + row['minute']
        curr_date = row['date']
        curr_user = row['username']

        if prev_user == curr_user and prev_date == curr_date:
            time_diff = curr_time - prev_time
            if 0 <= time_diff <= 1:
                current_chunk += 1
            else:
                chunks.append(current_chunk)
                current_chunk = 1
        else:
            if prev_user is not None:
                chunks.append(current_chunk)
            current_chunk = 1

        prev_user = curr_user
        prev_time = curr_time
        prev_date = curr_date

    chunks.append(current_chunk)
    chunks = np.array(chunks)

    # Geometric distribution fitting
    mean_chunk = np.mean(chunks)
    p_geometric = 1.0 / mean_chunk

    total_chunks = len(chunks)

    result = {
        'distribution': 'geometric',
        'p': float(p_geometric),
        'mean_chunk_size': float(mean_chunk),
        'median_chunk_size': float(np.median(chunks)),
        'max_chunk_size': int(np.max(chunks)),
        'pct_single': float(np.mean(chunks == 1) * 100),
        'pct_double': float(np.mean(chunks == 2) * 100),
        'pct_triple': float(np.mean(chunks == 3) * 100),
        'pct_4plus': float(np.mean(chunks >= 4) * 100),
        'pct_multi': float(np.mean(chunks > 1) * 100),
        'n_chunks': total_chunks
    }

    print(f"      Geometric p={p_geometric:.4f}")
    print(f"      Mean chunk size: {mean_chunk:.2f}")
    print(f"      Single: {result['pct_single']:.1f}%, Multi(>1): {result['pct_multi']:.1f}%")
    print(f"      (Baron 2010 reference: ~16% multi-message)")

    return result, chunks

def analyze_typing_imperfection(df):
    """4.4 Typing Imperfection Analysis"""
    print("\n[4.4] Analyzing Typing Imperfection...")

    correction_pattern = r'\*\w+'
    ellipsis_pattern = r'\.{2,}'

    abbreviations = [
        r'\bu\b', r'\bur\b', r'\bty\b', r'\bthx\b', r'\bthnx\b',
        r'\blol\b', r'\blmao\b', r'\brofl\b', r'\bnp\b', r'\bidk\b',
        r'\bbrb\b', r'\bgtg\b', r'\bg2g\b', r'\bafk\b', r'\bbtw\b',
        r'\bimo\b', r'\bimho\b', r'\bomg\b', r'\bwth\b', r'\bwtf\b',
        r'\bpls\b', r'\bplz\b', r'\btbh\b', r'\bfyi\b', r'\basap\b',
        r'\bk\b', r'\bkk\b', r'\bw/\b', r'\bw/o\b', r'\bb4\b',
        r'\bc\b', r'\br\b', r'\by\b', r'\bn\b'
    ]
    abbrev_pattern = '|'.join(abbreviations)

    corrections = 0
    ellipses = 0
    abbreviation_msgs = 0

    for text in df['text'].str.lower():
        if re.search(correction_pattern, str(text)):
            corrections += 1
        if re.search(ellipsis_pattern, str(text)):
            ellipses += 1
        if re.search(abbrev_pattern, str(text), re.IGNORECASE):
            abbreviation_msgs += 1

    total = len(df)

    result = {
        'correction_rate': float(corrections / total * 100),
        'abbreviation_rate': float(abbreviation_msgs / total * 100),
        'ellipsis_rate': float(ellipses / total * 100),
        'correction_count': corrections,
        'abbreviation_count': abbreviation_msgs,
        'ellipsis_count': ellipses,
        'total_messages': total
    }

    print(f"      Correction (*word): {result['correction_rate']:.2f}%")
    print(f"      Abbreviations: {result['abbreviation_rate']:.2f}%")
    print(f"      Ellipsis (...): {result['ellipsis_rate']:.2f}%")

    return result

def analyze_reactive_behaviors(df):
    """4.5 Reactive Behaviors Analysis"""
    print("\n[4.5] Analyzing Reactive Behaviors...")

    acknowledgments = r'\b(ok|okay|yes|yeah|yea|yep|yup|sure|right|thanks|thank|ty|thx|np|yw|alright|agreed|exactly|indeed)\b'
    backchannels = r'\b(hmm+|hm+|oh|ah|huh|mhm|uh|uhh|umm?|well)\b'
    laughter = r'\b(lol|lmao|rofl|haha|hehe|hihi|xd|lul)\b'
    emoticons = r'[:;8=][-\']?[)(\[\]DPpOo3><|/\\]+|[)(\[\]DPp]+[-\']?[:;8=]|\^\^|<3|:[-]?[)D(P]'

    texts = df['text'].str.lower()
    word_counts = df['word_count']

    ack_count = sum(1 for t in texts if re.search(acknowledgments, str(t), re.IGNORECASE))
    back_count = sum(1 for t in texts if re.search(backchannels, str(t), re.IGNORECASE))
    laugh_count = sum(1 for t in texts if re.search(laughter, str(t), re.IGNORECASE))
    emote_count = sum(1 for t in texts if re.search(emoticons, str(t)))
    short_count = sum(1 for wc in word_counts if wc <= 3)

    reactive_count = 0
    for t, wc in zip(texts, word_counts):
        is_reactive = (
            re.search(acknowledgments, str(t), re.IGNORECASE) or
            re.search(backchannels, str(t), re.IGNORECASE) or
            re.search(laughter, str(t), re.IGNORECASE) or
            re.search(emoticons, str(t)) or
            wc <= 3
        )
        if is_reactive:
            reactive_count += 1

    total = len(df)

    result = {
        'short_response_rate': float(short_count / total * 100),
        'acknowledgment_rate': float(ack_count / total * 100),
        'backchannel_rate': float(back_count / total * 100),
        'laughter_rate': float(laugh_count / total * 100),
        'emoticon_rate': float(emote_count / total * 100),
        'overall_reactive_rate': float(reactive_count / total * 100),
        'counts': {
            'short': short_count,
            'acknowledgment': ack_count,
            'backchannel': back_count,
            'laughter': laugh_count,
            'emoticon': emote_count,
            'overall_reactive': reactive_count
        },
        'total_messages': total
    }

    print(f"      Short (≤3 words): {result['short_response_rate']:.2f}%")
    print(f"      Acknowledgments: {result['acknowledgment_rate']:.2f}%")
    print(f"      Backchannels: {result['backchannel_rate']:.2f}%")
    print(f"      Laughter: {result['laughter_rate']:.2f}%")
    print(f"      Emoticons: {result['emoticon_rate']:.2f}%")
    print(f"      Overall Reactive: {result['overall_reactive_rate']:.2f}%")

    return result

def analyze_response_decision(df, df_replies):
    """4.6 Response Decision Analysis"""
    print("\n[4.6] Analyzing Response Decision...")

    total_messages = len(df)
    total_reply_links = len(df_replies)

    # Self-reply: messages that reply to same user
    self_replies = 0
    if len(df_replies) > 0:
        for _, row in df_replies.iterrows():
            if row['source_user'] == row['reply_user']:
                self_replies += 1

        replied_messages = df_replies['source_idx'].nunique()
        reply_messages = df_replies['reply_idx'].nunique()
    else:
        replied_messages = 0
        reply_messages = 0

    # Messages with mentions (indicating reply intent)
    mention_rate = df['mentioned_user'].notna().mean() * 100

    result = {
        'self_reply_rate': float(self_replies / total_reply_links * 100) if total_reply_links > 0 else 0,
        'avg_replies_per_message': float(total_reply_links / total_messages),
        'pct_messages_replied_to': float(replied_messages / total_messages * 100),
        'pct_messages_are_replies': float(reply_messages / total_messages * 100),
        'mention_rate': float(mention_rate),
        'self_reply_count': self_replies,
        'total_reply_links': total_reply_links,
        'total_messages': total_messages
    }

    print(f"      Self-reply rate: {result['self_reply_rate']:.2f}%")
    print(f"      Avg replies per message: {result['avg_replies_per_message']:.3f}")
    print(f"      Messages with mentions: {result['mention_rate']:.1f}%")
    print(f"      Messages that are replies: {result['pct_messages_are_replies']:.1f}%")

    return result

# =============================================================================
# 5. Run All Analyses
# =============================================================================
print("\n" + "=" * 60)
print("Running Analyses...")
print("=" * 60)

latency_result, latencies = analyze_response_latency(df, df_replies)
length_result, word_counts = analyze_turn_length(df)
chunking_result, chunks = analyze_message_chunking(df)
imperfection_result = analyze_typing_imperfection(df)
reactive_result = analyze_reactive_behaviors(df)
decision_result = analyze_response_decision(df, df_replies)

# =============================================================================
# 6. Create Visualization
# =============================================================================
print("\n[5/7] Creating visualization...")

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('Ubuntu IRC Behavioral Cues Distribution Analysis', fontsize=14, fontweight='bold')

# 4.1 Response Latency
ax1 = axes[0, 0]
if latency_result and len(latencies) > 0:
    latencies_plot = latencies[latencies <= np.percentile(latencies, 95)]
    ax1.hist(latencies_plot, bins=50, density=True, alpha=0.7, color='steelblue', edgecolor='white')
    x = np.linspace(0.1, np.max(latencies_plot), 200)
    pdf = stats.lognorm.pdf(x, latency_result['sigma'], 0, np.exp(latency_result['mu']))
    ax1.plot(x, pdf, 'r-', linewidth=2, label=f'Log-Normal\nμ={latency_result["mu"]:.2f}, σ={latency_result["sigma"]:.2f}')
    ax1.set_xlabel('Response Latency (minutes)')
    ax1.set_ylabel('Density')
    ax1.set_title('Response Latency Distribution')
    ax1.legend(loc='upper right')
    ax1.set_xlim(0, np.percentile(latencies, 95))
else:
    ax1.text(0.5, 0.5, 'No latency data', ha='center', va='center', transform=ax1.transAxes)
    ax1.set_title('Response Latency Distribution')

# 4.2 Turn Length
ax2 = axes[0, 1]
word_counts_plot = word_counts[word_counts <= np.percentile(word_counts, 95)]
ax2.hist(word_counts_plot, bins=30, density=True, alpha=0.7, color='forestgreen', edgecolor='white')
x = np.linspace(0.1, np.max(word_counts_plot), 200)
pdf = stats.gamma.pdf(x, length_result['shape_k'], 0, length_result['scale_theta'])
ax2.plot(x, pdf, 'r-', linewidth=2, label=f'Gamma\nk={length_result["shape_k"]:.2f}, θ={length_result["scale_theta"]:.2f}')
ax2.set_xlabel('Turn Length (words)')
ax2.set_ylabel('Density')
ax2.set_title('Turn Length Distribution')
ax2.legend(loc='upper right')

# 4.3 Message Chunking
ax3 = axes[0, 2]
chunk_labels = ['1', '2', '3', '4+']
chunk_values = [
    chunking_result['pct_single'],
    chunking_result['pct_double'],
    chunking_result['pct_triple'],
    chunking_result['pct_4plus']
]
colors = plt.cm.Oranges(np.linspace(0.3, 0.8, 4))
bars = ax3.bar(chunk_labels, chunk_values, color=colors, edgecolor='white')
ax3.set_xlabel('Consecutive Messages')
ax3.set_ylabel('Percentage (%)')
ax3.set_title(f'Message Chunking\n(p={chunking_result["p"]:.3f})')
for bar, val in zip(bars, chunk_values):
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
             f'{val:.1f}%', ha='center', va='bottom', fontsize=9)

# 4.4 Typing Imperfection
ax4 = axes[1, 0]
imperf_labels = ['Corrections\n(*word)', 'Abbreviations\n(lol, u, etc)', 'Ellipsis\n(...)']
imperf_values = [
    imperfection_result['correction_rate'],
    imperfection_result['abbreviation_rate'],
    imperfection_result['ellipsis_rate']
]
colors = plt.cm.Purples(np.linspace(0.4, 0.8, 3))
bars = ax4.bar(imperf_labels, imperf_values, color=colors, edgecolor='white')
ax4.set_ylabel('Percentage (%)')
ax4.set_title('Typing Imperfection Rates')
for bar, val in zip(bars, imperf_values):
    ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
             f'{val:.2f}%', ha='center', va='bottom', fontsize=9)

# 4.5 Reactive Behaviors
ax5 = axes[1, 1]
react_labels = ['Short\n(≤3 words)', 'Ack', 'Backchannel', 'Laughter', 'Emoticon']
react_values = [
    reactive_result['short_response_rate'],
    reactive_result['acknowledgment_rate'],
    reactive_result['backchannel_rate'],
    reactive_result['laughter_rate'],
    reactive_result['emoticon_rate']
]
colors = plt.cm.Blues(np.linspace(0.3, 0.8, 5))
bars = ax5.bar(react_labels, react_values, color=colors, edgecolor='white')
ax5.set_ylabel('Percentage (%)')
ax5.set_title(f'Reactive Behaviors\n(Overall: {reactive_result["overall_reactive_rate"]:.1f}%)')
for bar, val in zip(bars, react_values):
    ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
             f'{val:.1f}%', ha='center', va='bottom', fontsize=8)

# 4.6 Response Decision
ax6 = axes[1, 2]
decision_labels = ['Mention\nRate', 'Self-Reply', 'Are Replies']
decision_values = [
    decision_result['mention_rate'],
    decision_result['self_reply_rate'],
    decision_result['pct_messages_are_replies']
]
colors = plt.cm.Greens(np.linspace(0.4, 0.8, 3))
bars = ax6.bar(decision_labels, decision_values, color=colors, edgecolor='white')
ax6.set_ylabel('Percentage (%)')
ax6.set_title(f'Response Decision\n(Avg {decision_result["avg_replies_per_message"]:.2f} replies/msg)')
for bar, val in zip(bars, decision_values):
    ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
             f'{val:.1f}%', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig('distribution_analysis.png', dpi=150, bbox_inches='tight')
print("    Saved: distribution_analysis.png")

# =============================================================================
# 7. Save Results
# =============================================================================
print("\n[6/7] Saving JSON results...")

results = {
    "dataset_info": {
        "name": "Ubuntu IRC Disentanglement",
        "source": "Kummerfeld et al. (ACL 2019)",
        "huggingface_id": "jkkummerfeld/irc_disentangle",
        "total_messages": len(df),
        "total_reply_links": len(df_replies),
        "unique_users": int(df['username'].nunique()),
        "date_range": {
            "start": df['date'].min(),
            "end": df['date'].max()
        }
    },
    "response_latency": latency_result if latency_result else {"error": "insufficient data"},
    "turn_length": length_result,
    "message_chunking": chunking_result,
    "typing_imperfection": imperfection_result,
    "reactive_behaviors": reactive_result,
    "response_decision": decision_result
}

with open('irc_rhythm_parameters.json', 'w', encoding='utf-8') as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
print("    Saved: irc_rhythm_parameters.json")

# =============================================================================
# 8. Generate Report
# =============================================================================
print("\n[7/7] Generating analysis report...")

latency_section = ""
if latency_result:
    latency_section = f"""## 1. Response Latency

Response latency follows a **Log-Normal distribution**.

| Parameter | Value |
|-----------|-------|
| μ (log-scale mean) | {latency_result['mu']:.4f} |
| σ (log-scale std) | {latency_result['sigma']:.4f} |
| Mean | {latency_result['mean_minutes']:.2f} min |
| Median | {latency_result['median_minutes']:.2f} min |
| 25th percentile | {latency_result['percentile_25']:.2f} min |
| 75th percentile | {latency_result['percentile_75']:.2f} min |
| 90th percentile | {latency_result['percentile_90']:.2f} min |
| K-S statistic | {latency_result['ks_statistic']:.4f} |
| K-S p-value | {latency_result['ks_pvalue']:.4f} |
| N samples | {latency_result['n_samples']:,} |

**Usage**: `delay = np.random.lognormal({latency_result['mu']:.3f}, {latency_result['sigma']:.3f})`
"""
else:
    latency_section = "## 1. Response Latency\n\nInsufficient data for latency analysis."

report = f"""# Ubuntu IRC Behavioral Cues Analysis Report

## Dataset Information
- **Source**: Kummerfeld et al. (ACL 2019) - IRC Disentanglement
- **HuggingFace ID**: `jkkummerfeld/irc_disentangle`
- **Total Messages**: {len(df):,}
- **Total Reply Links**: {len(df_replies):,}
- **Unique Users**: {df['username'].nunique():,}

---

{latency_section}

---

## 2. Turn Length (Words per Message)

Turn length follows a **Gamma distribution**.

| Parameter | Value |
|-----------|-------|
| Shape (k) | {length_result['shape_k']:.4f} |
| Scale (θ) | {length_result['scale_theta']:.4f} |
| Mean | {length_result['mean_words']:.2f} words |
| Median | {length_result['median_words']:.2f} words |
| Short (≤5 words) | {length_result['pct_short_le5']:.1f}% |
| Medium (6-20 words) | {length_result['pct_medium_6_20']:.1f}% |
| Long (>20 words) | {length_result['pct_long_gt20']:.1f}% |
| N samples | {length_result['n_samples']:,} |

**Usage**: `words = int(np.random.gamma({length_result['shape_k']:.3f}, {length_result['scale_theta']:.3f}))`

---

## 3. Message Chunking

Consecutive messages follow a **Geometric distribution**.

| Parameter | Value |
|-----------|-------|
| p (probability) | {chunking_result['p']:.4f} |
| Mean chunk size | {chunking_result['mean_chunk_size']:.2f} |
| Single message | {chunking_result['pct_single']:.1f}% |
| Double (2) | {chunking_result['pct_double']:.1f}% |
| Triple (3) | {chunking_result['pct_triple']:.1f}% |
| 4+ messages | {chunking_result['pct_4plus']:.1f}% |
| **Multi-message rate** | {chunking_result['pct_multi']:.1f}% |

> **Reference**: Baron (2010) reported ~16% multi-message rate in IM conversations.
> Our finding: **{chunking_result['pct_multi']:.1f}%**

**Usage**: `chunk_size = np.random.geometric({chunking_result['p']:.4f})`

---

## 4. Typing Imperfection

| Pattern | Rate |
|---------|------|
| Self-corrections (*word) | {imperfection_result['correction_rate']:.2f}% |
| Abbreviations (lol, u, thx, etc.) | {imperfection_result['abbreviation_rate']:.2f}% |
| Ellipsis (...) | {imperfection_result['ellipsis_rate']:.2f}% |

---

## 5. Reactive Behaviors

| Behavior | Rate |
|----------|------|
| Short responses (≤3 words) | {reactive_result['short_response_rate']:.2f}% |
| Acknowledgments (ok, yes, thanks) | {reactive_result['acknowledgment_rate']:.2f}% |
| Backchannels (hmm, oh, huh) | {reactive_result['backchannel_rate']:.2f}% |
| Laughter (lol, haha) | {reactive_result['laughter_rate']:.2f}% |
| Emoticons (:), :D, etc.) | {reactive_result['emoticon_rate']:.2f}% |
| **Overall Reactive** | {reactive_result['overall_reactive_rate']:.2f}% |

---

## 6. Response Decision

| Metric | Value |
|--------|-------|
| Mention rate (username:) | {decision_result['mention_rate']:.2f}% |
| Self-reply rate | {decision_result['self_reply_rate']:.2f}% |
| Avg replies per message | {decision_result['avg_replies_per_message']:.3f} |
| Messages that are replies | {decision_result['pct_messages_are_replies']:.1f}% |

---

## Summary: Key Parameters for Human-like Chatbot

```python
import numpy as np

# Response timing (minutes)
{"delay = np.random.lognormal(mu=" + f"{latency_result['mu']:.3f}, sigma={latency_result['sigma']:.3f})" if latency_result else "# Latency data unavailable"}

# Message length (words)
word_count = int(np.random.gamma(shape={length_result['shape_k']:.3f}, scale={length_result['scale_theta']:.3f}))

# Message chunking
chunk_size = np.random.geometric(p={chunking_result['p']:.4f})

# Reactive behavior probability
reactive_prob = {reactive_result['overall_reactive_rate']/100:.3f}

# Typing imperfection rates
correction_prob = {imperfection_result['correction_rate']/100:.4f}
abbreviation_prob = {imperfection_result['abbreviation_rate']/100:.4f}
```

---

## References

1. Kummerfeld, J. K., et al. (2019). "A Large-Scale Corpus for Conversation Disentanglement." ACL.
2. Baron, N. S. (2010). "Discourse Structures in Instant Messaging." Language@Internet.
3. Gnewuch, U., et al. (2018). "Faster is Not Always Better: Understanding the Effect of Dynamic Response Delays in Human-Chatbot Interaction." ECIS.

---

*Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""

with open('analysis_report.md', 'w', encoding='utf-8') as f:
    f.write(report)
print("    Saved: analysis_report.md")

print("\n" + "=" * 60)
print("Analysis Complete!")
print("=" * 60)
print("\nOutput files:")
print("  1. irc_rhythm_parameters.json - All extracted parameters")
print("  2. distribution_analysis.png  - Distribution visualizations")
print("  3. analysis_report.md         - Full analysis report")
print("=" * 60)
