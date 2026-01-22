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
TIMING_MULTIPLIER = 1.5  # 1.5x (faster than before, was 2.5)

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




def generate_baseline_video(user_messages, agent_messages, output_path, speed_multiplier=1.0):
    """
    Generate video for baseline comparison
    Shows user messages then agent responses
    speed_multiplier: higher = faster video (2.0 = 2x faster)
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    font = get_font()
    frames = []

    messages_shown = []
    last_ts = 0

    # Combine user messages and agent messages
    all_messages = user_messages + agent_messages

    # Adjust timing multiplier for speed
    effective_multiplier = TIMING_MULTIPLIER / speed_multiplier

    for i, msg in enumerate(all_messages):
        msg_ts = msg.get('ts', 0) * effective_multiplier

        # Calculate delay since last message
        if i > 0:
            delay_ms = msg_ts - last_ts
            delay_seconds = delay_ms / 1000.0
            delay_seconds = min(delay_seconds, 6.0)

            if delay_seconds > 0.1:
                delay_frames = int(delay_seconds * FPS)
                frame = create_frame(messages_shown, font)
                for _ in range(delay_frames):
                    frames.append(frame)

        last_ts = msg_ts
        messages_shown.append(msg)
        frame = create_frame(messages_shown, font)

        # Hold for reading time (adjusted for speed)
        hold_time = calculate_reading_time(msg['text']) * 0.5 / speed_multiplier
        hold_frames = max(1, int(hold_time * FPS))
        for _ in range(hold_frames):
            frames.append(frame)

    # Hold final state (adjusted for speed)
    final_hold = int(3 * FPS / speed_multiplier)
    for _ in range(final_hold):
        frames.append(frames[-1] if frames else create_frame([], font))

    mp4_path = output_path.replace('.gif', '.mp4')
    if not mp4_path.endswith('.mp4'):
        mp4_path = output_path

    imageio.mimsave(mp4_path, frames, fps=FPS, codec='libx264', quality=9, pixelformat='yuv420p')
    print(f"Generated: {mp4_path}")


def main():
    """Generate survey videos for timing and baseline comparisons"""

    print("Generating survey videos...")
    print(f"Output directory: {OUTPUT_DIR}")

    # ============================================
    # Set 5: Python TensorFlow version conflict (timing comparison)
    # ============================================
    timing_1_messages = [
        {'role': 'user', 'sender': 'david', 'text': 'My TensorFlow code that worked on Python 3.8 throws AttributeError on 3.11\nPython 3.8에서 되던 TensorFlow 코드가 3.11에서 AttributeError 나요', 'ts': 0},
        {'role': 'user', 'sender': 'emma', 'text': 'What error exactly?\n정확히 무슨 에러?', 'ts': 2500},
        {'role': 'user', 'sender': 'david', 'text': "AttributeError: module 'tensorflow' has no attribute 'Session'\nAttributeError: module 'tensorflow' has no attribute 'Session'이요", 'ts': 5000},
        {'role': 'agent', 'sender': 'helper', 'text': 'david: tf.Session was removed in TF2. Use tf.compat.v1.Session instead\ndavid: tf.Session은 TF2에서 삭제됐어요. tf.compat.v1.Session 쓰세요', 'ts': 9000, 'noTimingTs': 5100},
        {'role': 'agent', 'sender': 'helper', 'text': 'Or migrate to TF2 eager execution\n아니면 TF2 즉시 실행으로 마이그레이션하세요', 'ts': 11500, 'noTimingTs': 5200},
        {'role': 'user', 'sender': 'david', 'text': "I'm using TensorFlow 2.10. Is that the issue?\nTensorFlow 2.10 쓰는데 그게 문제인가요?", 'ts': 14000},
        {'role': 'agent', 'sender': 'helper', 'text': 'david: TF 2.10 only supports up to Python 3.10\ndavid: TF 2.10은 Python 3.10까지만 지원해요', 'ts': 18000, 'noTimingTs': 14100},
        {'role': 'agent', 'sender': 'helper', 'text': 'Upgrade to TF 2.15: pip install tensorflow==2.15.0\nTF 2.15로 업그레이드: pip install tensorflow==2.15.0', 'ts': 21000, 'noTimingTs': 14200},
        {'role': 'user', 'sender': 'david', 'text': 'Works now with 2.15, thanks!\n2.15로 하니까 돼요, 감사합니다!', 'ts': 24000},
    ]

    generate_timing_video(timing_1_messages, os.path.join(OUTPUT_DIR, 'timing_full_1.mp4'), with_timing=True)
    generate_timing_video(timing_1_messages, os.path.join(OUTPUT_DIR, 'timing_notiming_1.mp4'), with_timing=False)

    # ============================================
    # Set 6: Excel VLOOKUP error (timing comparison)
    # ============================================
    timing_2_messages = [
        {'role': 'user', 'sender': 'amy', 'text': 'My VLOOKUP keeps returning #N/A even though the value exists\nVLOOKUP이 값이 있는데도 #N/A가 나와요', 'ts': 0},
        {'role': 'user', 'sender': 'brian', 'text': 'Are you using exact or approximate match?\n정확히 일치? 근사 일치?', 'ts': 2500},
        {'role': 'user', 'sender': 'amy', 'text': "I put TRUE at the end, isn't that exact match?\n끝에 TRUE 넣었는데 그게 정확히 일치 아니에요?", 'ts': 5000},
        {'role': 'agent', 'sender': 'helper', 'text': 'amy: TRUE means approximate match, use FALSE for exact\namy: TRUE는 근사 일치예요, 정확히 일치는 FALSE 쓰세요', 'ts': 9000, 'noTimingTs': 5100},
        {'role': 'agent', 'sender': 'helper', 'text': 'Approximate match also requires sorted data\n근사 일치는 데이터 정렬도 필요해요', 'ts': 11500, 'noTimingTs': 5200},
        {'role': 'user', 'sender': 'amy', 'text': 'Changed to FALSE but still not working\nFALSE로 바꿨는데 여전히 안 돼요', 'ts': 14000},
        {'role': 'agent', 'sender': 'helper', 'text': 'amy: There might be hidden spaces in your cells\namy: 셀에 숨겨진 공백이 있을 수 있어요', 'ts': 18000, 'noTimingTs': 14100},
        {'role': 'agent', 'sender': 'helper', 'text': 'Try: =VLOOKUP(TRIM(A1),B:C,2,FALSE)\n=VLOOKUP(TRIM(A1),B:C,2,FALSE)를 사용해보세요', 'ts': 21000, 'noTimingTs': 14200},
        {'role': 'user', 'sender': 'amy', 'text': 'That was it! Hidden spaces were the problem\n그거였어요! 숨겨진 공백이 문제였네요', 'ts': 24000},
    ]

    generate_timing_video(timing_2_messages, os.path.join(OUTPUT_DIR, 'timing_full_2.mp4'), with_timing=True)
    generate_timing_video(timing_2_messages, os.path.join(OUTPUT_DIR, 'timing_notiming_2.mp4'), with_timing=False)

    # ============================================
    # Set 7: Book translation - Friends "We were on a break!" (baseline comparison)
    # ============================================
    baseline_1_user = [
        {'role': 'user', 'sender': 'editor', 'text': "How should we translate 'We were on a break!' from Friends? It's a running joke\n프렌즈의 'We were on a break!' 어떻게 번역할까요? 반복되는 유머예요", 'ts': 0},
        {'role': 'user', 'sender': 'translatorA', 'text': "Literally '우리 그때 잠깐 헤어졌었잖아!' - but it doesn't hit the same\n직역하면 '우리 그때 잠깐 헤어졌었잖아!' - 근데 임팩트가 달라요", 'ts': 2500},
        {'role': 'user', 'sender': 'translatorB', 'text': "'잠시 쉬는 거였다고!' sounds more defensive like Ross\n'잠시 쉬는 거였다고!'가 Ross처럼 방어적으로 들려요", 'ts': 5000},
    ]

    baseline_1_full = [
        {'role': 'agent', 'sender': 'helper', 'text': 'editor: The humor relies on Ross repeating this defensively for years\neditor: 유머는 Ross가 수년간 방어적으로 반복하는 데서 와요', 'ts': 9000},
        {'role': 'agent', 'sender': 'helper', 'text': "Korean viewers need to feel his desperation, not just understand the words\n한국 시청자들이 단어가 아닌 그의 절박함을 느껴야 해요", 'ts': 11500},
        {'role': 'user', 'sender': 'editor', 'text': 'So which captures that desperation better?\n그럼 그 절박함을 더 잘 살리는 건 뭐예요?', 'ts': 14000},
        {'role': 'agent', 'sender': 'helper', 'text': "editor: '잠깐 쉬는 거였다고!' with the emphasis marker keeps the exasperation\neditor: '잠깐 쉬는 거였다고!'에 강조 어미를 쓰면 짜증이 살아요", 'ts': 18000},
        {'role': 'agent', 'sender': 'helper', 'text': "The '다고' ending in Korean conveys 'I keep telling you this!'\n한국어의 '다고' 어미가 '계속 말하잖아!'를 전달해요", 'ts': 21000},
        {'role': 'agent', 'sender': 'helper', 'text': 'Consistency matters too - use the same phrase every time Ross says it\n일관성도 중요해요 - Ross가 말할 때마다 같은 표현을 쓰세요', 'ts': 24000},
    ]

    baseline_1_baseline = [
        {'role': 'agent', 'sender': 'helper', 'text': "This phrase can be translated as '우리 그때 헤어진 거였어!' or '잠시 쉬는 거였다고!'. When translating idiomatic expressions, consider the target audience and cultural context. Literal translation preserves the original structure but may lose nuance. Liberal translation captures meaning but changes form. The best approach depends on whether the text prioritizes accuracy or readability.\n이 문구는 '우리 그때 헤어진 거였어!' 또는 '잠시 쉬는 거였다고!'로 번역할 수 있습니다. 관용 표현을 번역할 때는 대상 독자와 문화적 맥락을 고려하세요.", 'ts': 9000},
        {'role': 'user', 'sender': 'editor', 'text': 'So which captures that desperation better?\n그럼 그 절박함을 더 잘 살리는 건 뭐예요?', 'ts': 14000},
        {'role': 'agent', 'sender': 'helper', 'text': 'Translation choice depends on your translation philosophy and target readers.\n번역 선택은 번역 철학과 대상 독자에 달려 있습니다.', 'ts': 18000},
    ]

    generate_baseline_video(baseline_1_user, baseline_1_full, os.path.join(OUTPUT_DIR, 'baseline_full_1.mp4'), speed_multiplier=2.0)
    generate_baseline_video(baseline_1_user, baseline_1_baseline, os.path.join(OUTPUT_DIR, 'baseline_baseline_1.mp4'), speed_multiplier=2.0)

    # ============================================
    # Set 8: Database model selection (baseline comparison)
    # ============================================
    baseline_2_user = [
        {'role': 'user', 'sender': 'pm', 'text': 'The vendor used MongoDB but we need complex joins for reporting. Should we migrate to PostgreSQL?\n외주가 MongoDB 썼는데 리포팅에 복잡한 조인 필요해요. PostgreSQL로 마이그레이션해야 하나요?', 'ts': 0},
        {'role': 'user', 'sender': 'dev', 'text': 'Our e-commerce data has clear product-order-customer relationships\n우리 이커머스 데이터는 상품-주문-고객 관계가 명확해요', 'ts': 2500},
        {'role': 'user', 'sender': 'analyst', 'text': 'I need to run monthly sales reports with multiple table joins\n저는 여러 테이블 조인으로 월간 매출 리포트 돌려야 해요', 'ts': 5000},
    ]

    baseline_2_full = [
        {'role': 'agent', 'sender': 'helper', 'text': 'pm: MongoDB lookups work but are 3-5x slower than SQL joins for your use case\npm: MongoDB lookup도 되지만 이 경우 SQL join보다 3-5배 느려요', 'ts': 9000},
        {'role': 'agent', 'sender': 'helper', 'text': 'For e-commerce with clear relationships, PostgreSQL is the better fit\n명확한 관계가 있는 이커머스엔 PostgreSQL이 더 맞아요', 'ts': 11500},
        {'role': 'user', 'sender': 'pm', 'text': 'How complex would the migration be?\n마이그레이션 복잡도는 어때요?', 'ts': 14000},
        {'role': 'agent', 'sender': 'helper', 'text': 'pm: Check nesting depth first - flat documents migrate easily\npm: 중첩 깊이부터 확인하세요 - 평면 문서는 쉽게 이전돼요', 'ts': 18000},
        {'role': 'agent', 'sender': 'helper', 'text': 'Deeply nested arrays need careful denormalization planning\n깊게 중첩된 배열은 비정규화 계획이 필요해요', 'ts': 21000},
        {'role': 'agent', 'sender': 'helper', 'text': "analyst: For your monthly reports, PostgreSQL materialized views would be ideal\nanalyst: 월간 리포트엔 PostgreSQL materialized view가 이상적이에요", 'ts': 24000},
    ]

    baseline_2_baseline = [
        {'role': 'agent', 'sender': 'helper', 'text': "MongoDB and PostgreSQL serve different use cases. MongoDB is a NoSQL document database that stores data in JSON-like documents, offering flexible schemas. PostgreSQL is a relational database that uses tables with predefined schemas and supports complex queries with SQL. MongoDB is better for unstructured data and rapid development, while PostgreSQL excels at complex transactions and data integrity. The choice depends on your data structure, query patterns, and scalability requirements.\nMongoDB와 PostgreSQL은 다른 용도로 사용됩니다. MongoDB는 JSON 형태 문서를 저장하는 NoSQL 문서 데이터베이스로 유연한 스키마를 제공합니다. PostgreSQL은 관계형 데이터베이스입니다.", 'ts': 9000},
        {'role': 'user', 'sender': 'pm', 'text': 'How complex would the migration be?\n마이그레이션 복잡도는 어때요?', 'ts': 14000},
        {'role': 'agent', 'sender': 'helper', 'text': 'Migration complexity varies depending on your specific data and requirements.\n마이그레이션 복잡도는 특정 데이터와 요구사항에 따라 다릅니다.', 'ts': 18000},
    ]

    generate_baseline_video(baseline_2_user, baseline_2_full, os.path.join(OUTPUT_DIR, 'baseline_full_2.mp4'), speed_multiplier=2.0)
    generate_baseline_video(baseline_2_user, baseline_2_baseline, os.path.join(OUTPUT_DIR, 'baseline_baseline_2.mp4'), speed_multiplier=2.0)

    print("\nAll videos generated successfully!")
    print(f"Total: 8 videos in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
