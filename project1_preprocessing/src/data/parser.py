"""IRC log parser - handles raw logs and JSONL formats"""

import re
import json
from typing import List, Dict, Optional
from datetime import datetime
from dateutil import parser as date_parser


class IRCParser:
    """Parse IRC logs into structured utterances"""

    # Pattern: [2024-01-15 14:23:12] <alice> message text
    IRC_LOG_PATTERN = re.compile(
        r'^\[(?P<timestamp>[^\]]+)\]\s+<(?P<speaker>[^>]+)>\s+(?P<text>.*)$'
    )

    def __init__(self, default_channel: str = "#ubuntu"):
        self.default_channel = default_channel

    def parse_file(self, file_path: str, format: str = "auto") -> List[Dict]:
        """
        Parse IRC log file

        Args:
            file_path: Path to log file
            format: "raw", "jsonl", or "auto" (detect automatically)

        Returns:
            List of utterance dicts with keys: ts, speaker, text, channel
        """
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            first_line = f.readline()
            f.seek(0)

            # Auto-detect format
            if format == "auto":
                format = "jsonl" if first_line.strip().startswith('{') else "raw"

            if format == "jsonl":
                return self._parse_jsonl(f)
            else:
                return self._parse_raw(f)

    def _parse_raw(self, file_obj) -> List[Dict]:
        """Parse raw IRC log format"""
        utterances = []

        for line_num, line in enumerate(file_obj, 1):
            line = line.strip()
            if not line:
                continue

            match = self.IRC_LOG_PATTERN.match(line)
            if not match:
                continue  # Skip malformed lines

            try:
                timestamp = date_parser.parse(match.group('timestamp'))
            except:
                continue  # Skip if timestamp unparseable

            speaker = match.group('speaker').strip()
            text = match.group('text').strip()

            # Extract mentions (words starting with @ or standalone nicks followed by :)
            mentions = self._extract_mentions(text)

            utterances.append({
                'utt_id': f"line_{line_num}",
                'ts': timestamp.isoformat(),
                'speaker': speaker,
                'text': text,
                'channel': self.default_channel,
                'mentions': mentions
            })

        return utterances

    def _parse_jsonl(self, file_obj) -> List[Dict]:
        """Parse JSONL format"""
        utterances = []

        for line_num, line in enumerate(file_obj, 1):
            line = line.strip()
            if not line:
                continue

            try:
                data = json.loads(line)

                # Normalize field names
                ts = data.get('ts') or data.get('timestamp')
                speaker = data.get('speaker') or data.get('user') or data.get('nick')
                text = data.get('text') or data.get('message')
                channel = data.get('channel', self.default_channel)

                if not (ts and speaker and text):
                    continue

                # Ensure ISO format timestamp
                if not isinstance(ts, str) or 'T' not in ts:
                    try:
                        ts = date_parser.parse(str(ts)).isoformat()
                    except:
                        continue

                mentions = data.get('mentions', self._extract_mentions(text))

                utterances.append({
                    'utt_id': data.get('utt_id', f"line_{line_num}"),
                    'ts': ts,
                    'speaker': speaker,
                    'text': text,
                    'channel': channel,
                    'mentions': mentions
                })

            except json.JSONDecodeError:
                continue

        return utterances

    def _extract_mentions(self, text: str, speaker_list: List[str] = None) -> List[str]:
        """
        Extract mentioned usernames from text

        Patterns:
        1. @nick - always captured
        2. nick: or nick, or nick; at line start - validated against speaker_list
        3. \\bnick\\b word boundary match - validated against speaker_list

        Args:
            text: The message text
            speaker_list: Valid speaker names to validate mentions against
        """
        mentions = []
        speaker_set = set(s.lower() for s in (speaker_list or []))

        # Pattern 1: @username - always valid
        mentions.extend(re.findall(r'@(\w+)', text))

        # Pattern 2: nick: or nick, or nick; at line start
        line_start_match = re.match(r'^(\w+)[:,;]\s*', text)
        if line_start_match:
            nick = line_start_match.group(1)
            # If speaker_list exists, validate; otherwise accept
            if not speaker_set or nick.lower() in speaker_set:
                mentions.append(nick)

        # Pattern 3: word boundary match for known speakers
        if speaker_set:
            words = re.findall(r'\b(\w+)\b', text)
            for word in words:
                if word.lower() in speaker_set and word not in mentions:
                    mentions.append(word)

        return list(set(mentions))  # Remove duplicates

    def extract_mentions_with_context(
        self,
        text: str,
        speaker: str,
        speaker_list: List[str]
    ) -> List[str]:
        """
        Extract mentions with full context (for accurate mention detection)

        Args:
            text: The message text
            speaker: The speaker of this message (to exclude self-mentions)
            speaker_list: All known speakers in the thread

        Returns:
            List of mentioned usernames (excluding self-mentions)
        """
        mentions = self._extract_mentions(text, speaker_list)
        # Exclude self-mentions
        return [m for m in mentions if m.lower() != speaker.lower()]
