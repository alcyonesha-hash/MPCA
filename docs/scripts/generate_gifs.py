#!/usr/bin/env python3
"""
Video Generator for Survey - Using Real IRC Conversations

Generates comparison videos showing:
1. Timing difference (with natural delays vs instant)
2. Chunking difference (split messages vs single long message)

Features:
- Profile avatars for each speaker (colored circles with initials)
- Auto-scroll as conversation progresses
- Natural, slower message timing

Uses real conversation data from Ubuntu IRC channel.
Output: MP4 videos that show chat messages appearing over time (no loop issue)
"""

import os
import subprocess
from PIL import Image, ImageDraw, ImageFont
import imageio
import numpy as np
import hashlib

# Configuration
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'gifs')
FPS = 15  # Frames per second (higher for smoother video)
WIDTH = 960  # Higher resolution (divisible by 16)
HEIGHT = 720  # Higher resolution (divisible by 16)
FONT_SIZE = 22  # Even larger font for better readability
BUBBLE_PADDING = 14
MESSAGE_GAP = 20
AVATAR_SIZE = 44  # Larger avatars
AVATAR_MARGIN = 14

# Timing multiplier - higher = slower messages (more natural)
TIMING_MULTIPLIER = 2.5  # 2.5x slower than before

# Colors
BG_COLOR = (248, 249, 250)
USER_BUBBLE = (227, 242, 253)
AGENT_BUBBLE = (102, 126, 234)
TEXT_COLOR = (51, 51, 51)
AGENT_TEXT = (255, 255, 255)
SENDER_COLOR = (85, 85, 85)

# Avatar colors for different users (consistent per username)
AVATAR_COLORS = [
    (76, 175, 80),    # Green
    (33, 150, 243),   # Blue
    (255, 152, 0),    # Orange
    (156, 39, 176),   # Purple
    (0, 188, 212),    # Cyan
    (255, 87, 34),    # Deep Orange
    (63, 81, 181),    # Indigo
    (233, 30, 99),    # Pink
    (139, 195, 74),   # Light Green
    (121, 85, 72),    # Brown
]

# Agent avatar color (distinct)
AGENT_AVATAR_COLOR = (102, 126, 234)  # Purple-blue (same as bubble)


def get_avatar_color(username, is_agent=False):
    """Get consistent color for a username"""
    if is_agent:
        return AGENT_AVATAR_COLOR
    # Hash username to get consistent color index
    hash_val = int(hashlib.md5(username.encode()).hexdigest(), 16)
    return AVATAR_COLORS[hash_val % len(AVATAR_COLORS)]


def get_font(size=FONT_SIZE):
    """Get a font that supports Korean, falling back to default if needed"""
    # Try Korean-supporting fonts first
    korean_fonts = [
        "/System/Library/Fonts/AppleSDGothicNeo.ttc",  # macOS Korean font
        "/System/Library/Fonts/Supplemental/AppleGothic.ttf",  # macOS fallback
        "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",  # Linux Korean font
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",  # Linux Noto CJK
    ]

    for font_path in korean_fonts:
        try:
            return ImageFont.truetype(font_path, size)
        except:
            continue

    # Fallback to system fonts
    try:
        return ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", size)
    except:
        try:
            return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size)
        except:
            return ImageFont.load_default()


def wrap_text(text, font, max_width, draw):
    """Wrap text to fit within max_width"""
    words = text.split()
    lines = []
    current_line = []

    for word in words:
        test_line = ' '.join(current_line + [word])
        bbox = draw.textbbox((0, 0), test_line, font=font)
        if bbox[2] - bbox[0] <= max_width:
            current_line.append(word)
        else:
            if current_line:
                lines.append(' '.join(current_line))
            current_line = [word]

    if current_line:
        lines.append(' '.join(current_line))

    return lines


def draw_avatar(draw, x, y, username, is_agent=False):
    """Draw a circular avatar with initials"""
    color = get_avatar_color(username, is_agent)

    # Draw circle
    draw.ellipse(
        [x, y, x + AVATAR_SIZE, y + AVATAR_SIZE],
        fill=color
    )

    # Draw initials (first 1-2 characters)
    initials = username[0].upper()
    if len(username) > 1 and username[1].isalpha():
        initials = username[:2].upper()

    font = get_font(16)  # Larger font for avatar initials
    bbox = draw.textbbox((0, 0), initials, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]

    text_x = x + (AVATAR_SIZE - text_width) // 2
    text_y = y + (AVATAR_SIZE - text_height) // 2 - 2

    draw.text((text_x, text_y), initials, font=font, fill=(255, 255, 255))


def calculate_message_height(message, font, draw, max_bubble_width):
    """Calculate the height of a message bubble"""
    text_parts = message['text'].split('\n')
    en_text = text_parts[0]
    ko_text = text_parts[1] if len(text_parts) > 1 else None

    en_lines = wrap_text(en_text, font, max_bubble_width - 2 * BUBBLE_PADDING, draw)
    ko_font = get_font(18)
    ko_lines = wrap_text(ko_text, ko_font, max_bubble_width - 2 * BUBBLE_PADDING, draw) if ko_text else []

    line_height = font.size + 3
    ko_line_height = ko_font.size + 2
    en_text_height = len(en_lines) * line_height
    ko_text_height = len(ko_lines) * ko_line_height if ko_lines else 0
    ko_spacing = 6 if ko_lines else 0

    bubble_height = en_text_height + ko_text_height + ko_spacing + 2 * BUBBLE_PADDING + 18
    return bubble_height + MESSAGE_GAP


def draw_message(draw, y_pos, message, font, is_agent=False):
    """Draw a single chat message bubble with avatar and bilingual support"""
    max_bubble_width = WIDTH - 80 - AVATAR_SIZE - AVATAR_MARGIN

    sender = message.get('sender', 'Agent' if is_agent else 'User')

    # Split text into English and Korean parts
    text_parts = message['text'].split('\n')
    en_text = text_parts[0]
    ko_text = text_parts[1] if len(text_parts) > 1 else None

    # Wrap English text
    en_lines = wrap_text(en_text, font, max_bubble_width - 2 * BUBBLE_PADDING, draw)

    # Wrap Korean text with smaller font
    ko_font = get_font(18)  # Slightly smaller font for Korean
    ko_lines = wrap_text(ko_text, ko_font, max_bubble_width - 2 * BUBBLE_PADDING, draw) if ko_text else []

    # Calculate bubble size
    line_height = font.size + 3
    ko_line_height = ko_font.size + 2
    en_text_height = len(en_lines) * line_height
    ko_text_height = len(ko_lines) * ko_line_height if ko_lines else 0
    ko_spacing = 6 if ko_lines else 0  # Extra space between EN and KO
    bubble_height = en_text_height + ko_text_height + ko_spacing + 2 * BUBBLE_PADDING + 18  # Extra for sender name

    # Calculate text width (max of English and Korean lines)
    max_line_width = 0
    for line in en_lines:
        bbox = draw.textbbox((0, 0), line, font=font)
        max_line_width = max(max_line_width, bbox[2] - bbox[0])
    for line in ko_lines:
        bbox = draw.textbbox((0, 0), line, font=ko_font)
        max_line_width = max(max_line_width, bbox[2] - bbox[0])

    bubble_width = max_line_width + 2 * BUBBLE_PADDING

    # Position based on agent/user
    if is_agent:
        avatar_x = WIDTH - AVATAR_SIZE - 10
        bubble_x = avatar_x - bubble_width - AVATAR_MARGIN
        bubble_color = AGENT_BUBBLE
        text_color = AGENT_TEXT
        ko_text_color = (200, 200, 255)  # Lighter color for Korean in agent bubble
    else:
        avatar_x = 10
        bubble_x = avatar_x + AVATAR_SIZE + AVATAR_MARGIN
        bubble_color = USER_BUBBLE
        text_color = TEXT_COLOR
        ko_text_color = (120, 120, 120)  # Gray for Korean in user bubble

    # Draw avatar
    draw_avatar(draw, avatar_x, y_pos, sender, is_agent)

    # Draw bubble
    draw.rounded_rectangle(
        [bubble_x, y_pos, bubble_x + bubble_width, y_pos + bubble_height],
        radius=8,
        fill=bubble_color
    )

    # Draw sender name
    small_font = get_font(14)  # Larger sender name font
    draw.text((bubble_x + BUBBLE_PADDING, y_pos + 4), sender, font=small_font, fill=SENDER_COLOR if not is_agent else (200, 200, 255))

    # Draw English text
    text_y = y_pos + 16
    for line in en_lines:
        draw.text((bubble_x + BUBBLE_PADDING, text_y), line, font=font, fill=text_color)
        text_y += line_height

    # Draw Korean text (smaller, different color)
    if ko_lines:
        text_y += ko_spacing
        for line in ko_lines:
            draw.text((bubble_x + BUBBLE_PADDING, text_y), line, font=ko_font, fill=ko_text_color)
            text_y += ko_line_height

    return y_pos + bubble_height + MESSAGE_GAP


def calculate_total_height(messages, font, draw):
    """Calculate total height of all messages"""
    max_bubble_width = WIDTH - 80 - AVATAR_SIZE - AVATAR_MARGIN
    total = 15  # Initial padding
    for msg in messages:
        total += calculate_message_height(msg, font, draw, max_bubble_width)
    return total


def create_frame(messages_to_show, font, scroll_offset=0):
    """Create a single frame with visible messages and scroll offset"""
    # Create a larger canvas for all messages
    temp_img = Image.new('RGB', (WIDTH, 3000), BG_COLOR)
    temp_draw = ImageDraw.Draw(temp_img)

    y_pos = 15
    for msg in messages_to_show:
        is_agent = msg['role'] == 'agent'
        y_pos = draw_message(temp_draw, y_pos, msg, font, is_agent)

    # Calculate scroll offset to show latest messages
    total_height = y_pos
    if total_height > HEIGHT - 20:
        scroll_offset = max(0, total_height - HEIGHT + 40)
    else:
        scroll_offset = 0

    # Crop the visible area
    visible_area = temp_img.crop((0, scroll_offset, WIDTH, scroll_offset + HEIGHT))

    return np.array(visible_area)


def calculate_reading_time(text):
    """Calculate realistic reading time based on text length"""
    words = len(text.split())
    # Slower reading for more natural feel
    reading_time = max(1.5, min(5.0, words / 2.5))
    return reading_time * TIMING_MULTIPLIER


def calculate_typing_time(text):
    """Calculate realistic typing time based on text length"""
    chars = len(text)
    # Average typing speed: ~4 chars per second (slower, more natural)
    typing_time = max(1.0, min(4.0, chars / 4.0))
    return typing_time * TIMING_MULTIPLIER


def generate_timing_video(messages, output_path, with_timing=True):
    """
    Generate video showing messages appearing with or without timing delays
    Uses 'ts' field from messages for timing control
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    font = get_font()
    frames = []

    messages_shown = []
    last_ts = 0

    for i, msg in enumerate(messages):
        # Get timing from message (scaled by timing multiplier)
        if with_timing:
            msg_ts = msg.get('ts', 0) * TIMING_MULTIPLIER
        else:
            # For no-timing mode:
            # - User messages: keep same timing as full version (natural conversation flow)
            # - Agent messages: appear quickly (noTimingTs)
            if msg['role'] == 'agent':
                msg_ts = msg.get('noTimingTs', msg.get('ts', 0)) * TIMING_MULTIPLIER
            else:
                msg_ts = msg.get('ts', 0) * TIMING_MULTIPLIER

        # Calculate delay since last message
        if i > 0:
            delay_ms = msg_ts - last_ts
            delay_seconds = delay_ms / 1000.0

            # Cap delay for reasonable video length
            delay_seconds = min(delay_seconds, 6.0)

            # Show current state for delay duration
            if delay_seconds > 0.1:
                delay_frames = int(delay_seconds * FPS)
                frame = create_frame(messages_shown, font)
                for _ in range(delay_frames):
                    frames.append(frame)

        last_ts = msg_ts

        # Add message
        messages_shown.append(msg)

        # Show new message
        frame = create_frame(messages_shown, font)

        # Hold for reading time
        # User messages should have same hold time in both versions
        # Only agent messages differ between timing/no-timing
        if msg['role'] == 'user':
            # User messages: same hold time in both A and B
            hold_time = calculate_reading_time(msg['text']) * 0.5
        else:
            # Agent messages: differ based on timing mode
            if with_timing:
                hold_time = calculate_reading_time(msg['text']) * 0.5
            else:
                hold_time = 0.3 * TIMING_MULTIPLIER

        hold_frames = max(1, int(hold_time * FPS))
        for _ in range(hold_frames):
            frames.append(frame)

    # Hold final state
    final_hold = int(3 * FPS)
    for _ in range(final_hold):
        frames.append(frames[-1] if frames else create_frame([], font))

    # Save as MP4 video (no loop issue)
    mp4_path = output_path.replace('.gif', '.mp4')
    if not mp4_path.endswith('.mp4'):
        mp4_path = output_path

    # Use imageio-ffmpeg to save as MP4 with high quality
    imageio.mimsave(mp4_path, frames, fps=FPS, codec='libx264', quality=9, pixelformat='yuv420p')
    print(f"Generated: {mp4_path}")




def main():
    """Generate survey videos for timing comparison (Sets 5 and 6)"""

    print("Generating survey videos for timing comparison...")
    print(f"Output directory: {OUTPUT_DIR}")

    # ============================================
    # Set 5: Python version library conflicts (timing comparison)
    # Topic: Python 버전별 라이브러리 충돌
    # ============================================
    timing_1_messages = [
        {'role': 'user', 'sender': 'david', 'text': 'Code that worked on Python 3.8 throws errors on 3.11\nPython 3.8에서 되던 코드가 3.11에서 에러 나요', 'ts': 0},
        {'role': 'user', 'sender': 'emma', 'text': 'I also got the same issue after upgrading\n저도 업그레이드 후에 같은 문제 생겼어요', 'ts': 2500},
        {'role': 'user', 'sender': 'frank', 'text': 'david: Which packages are you using? Numpy? Pandas?\ndavid: 어떤 패키지 쓰세요? Numpy? Pandas?', 'ts': 5000},
        {'role': 'agent', 'sender': 'helper', 'text': 'david: This is likely due to deprecated syntax removal\ndavid: 아마 deprecated 문법이 삭제된 것 같아요', 'ts': 9000, 'noTimingTs': 5100},
        {'role': 'agent', 'sender': 'helper', 'text': 'Check if you\'re using "collections.Callable" - it moved to collections.abc\ncollections.Callable 쓰는지 확인해보세요 - collections.abc로 이동됐어요', 'ts': 12000, 'noTimingTs': 5200},
        {'role': 'user', 'sender': 'david', 'text': 'Yes! I\'m using some older ML libraries that depend on that\n맞아요! 그걸 사용하는 오래된 ML 라이브러리를 쓰고 있어요', 'ts': 15000},
        {'role': 'user', 'sender': 'emma', 'text': 'frank: I\'m using scikit-learn and tensorflow\nfrank: scikit-learn이랑 tensorflow 쓰고 있어요', 'ts': 17500},
        {'role': 'agent', 'sender': 'helper', 'text': 'emma: Check tensorflow version compatibility\nemma: tensorflow 버전 호환성 확인해보세요', 'ts': 21500, 'noTimingTs': 17600},
        {'role': 'agent', 'sender': 'helper', 'text': 'TF 2.10+ is needed for Python 3.11\nPython 3.11에는 TF 2.10 이상이 필요해요', 'ts': 24500, 'noTimingTs': 17700},
        {'role': 'user', 'sender': 'frank', 'text': 'You can also try pyenv to manage multiple Python versions\npyenv로 여러 Python 버전을 관리할 수도 있어요', 'ts': 27500},
        {'role': 'agent', 'sender': 'helper', 'text': 'david: Consider using a virtual environment with Python 3.8\ndavid: Python 3.8로 가상환경 쓰는 것도 고려해보세요', 'ts': 31500, 'noTimingTs': 27600},
        {'role': 'user', 'sender': 'david', 'text': 'Thanks, I\'ll try creating a separate venv for this project\n감사해요, 이 프로젝트용으로 별도 venv 만들어볼게요', 'ts': 34500},
    ]

    generate_timing_video(
        timing_1_messages,
        os.path.join(OUTPUT_DIR, 'timing_full_1.mp4'),
        with_timing=True
    )
    generate_timing_video(
        timing_1_messages,
        os.path.join(OUTPUT_DIR, 'timing_notiming_1.mp4'),
        with_timing=False
    )

    # ============================================
    # Set 6: Excel VLOOKUP formula (timing comparison)
    # Topic: 엑셀 VLOOKUP 수식
    # ============================================
    timing_2_messages = [
        {'role': 'user', 'sender': 'grace', 'text': 'My VLOOKUP returns #N/A even though the value exists in the table\nVLOOKUP이 #N/A를 반환하는데 값이 테이블에 분명히 있어요', 'ts': 0},
        {'role': 'user', 'sender': 'henry', 'text': 'grace: Are you using exact match or approximate match?\ngrace: 정확히 일치로 찾고 있어요? 아니면 유사 일치?', 'ts': 2500},
        {'role': 'user', 'sender': 'iris', 'text': 'I had the same issue last week, turned out to be extra spaces\n저도 지난주에 같은 문제 있었는데, 공백이 원인이었어요', 'ts': 5000},
        {'role': 'agent', 'sender': 'helper', 'text': 'grace: Most common cause is trailing spaces in cells\ngrace: 가장 흔한 원인은 셀에 있는 뒤쪽 공백이에요', 'ts': 9000, 'noTimingTs': 5100},
        {'role': 'agent', 'sender': 'helper', 'text': 'Try wrapping your lookup value with TRIM()\nlookup 값을 TRIM()으로 감싸보세요', 'ts': 12000, 'noTimingTs': 5200},
        {'role': 'user', 'sender': 'grace', 'text': 'henry: I\'m using FALSE for exact match\nhenry: 정확히 일치하려고 FALSE 쓰고 있어요', 'ts': 15000},
        {'role': 'user', 'sender': 'henry', 'text': 'Also check if the data types match - text vs number\n데이터 타입도 확인해보세요 - 텍스트 vs 숫자', 'ts': 17500},
        {'role': 'agent', 'sender': 'helper', 'text': 'grace: You can use =VLOOKUP(TRIM(A1),B:C,2,FALSE)\ngrace: =VLOOKUP(TRIM(A1),B:C,2,FALSE)로 써보세요', 'ts': 21500, 'noTimingTs': 17600},
        {'role': 'agent', 'sender': 'helper', 'text': 'Or use XLOOKUP if you have Excel 365\n엑셀 365 있으면 XLOOKUP도 좋아요', 'ts': 24500, 'noTimingTs': 17700},
        {'role': 'user', 'sender': 'iris', 'text': 'XLOOKUP is so much better, no more column index counting\nXLOOKUP 진짜 좋아요, 열 번호 안 세도 돼요', 'ts': 27500},
        {'role': 'agent', 'sender': 'helper', 'text': 'grace: If it\'s a number stored as text, use VALUE() function\ngrace: 텍스트로 저장된 숫자면 VALUE() 함수 쓰세요', 'ts': 31500, 'noTimingTs': 27600},
        {'role': 'user', 'sender': 'grace', 'text': 'TRIM() worked! There were invisible spaces. Thank you all!\nTRIM()으로 됐어요! 보이지 않는 공백이 있었네요. 모두 감사해요!', 'ts': 34500},
    ]

    generate_timing_video(
        timing_2_messages,
        os.path.join(OUTPUT_DIR, 'timing_full_2.mp4'),
        with_timing=True
    )
    generate_timing_video(
        timing_2_messages,
        os.path.join(OUTPUT_DIR, 'timing_notiming_2.mp4'),
        with_timing=False
    )

    print("\nAll videos generated successfully!")
    print(f"Total: 4 videos in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
