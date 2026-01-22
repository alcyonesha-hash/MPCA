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
FPS = 10  # Frames per second
WIDTH = 608  # Divisible by 16 for video codec compatibility
HEIGHT = 512  # Divisible by 16 for video codec compatibility
FONT_SIZE = 13
BUBBLE_PADDING = 8
MESSAGE_GAP = 14
AVATAR_SIZE = 28
AVATAR_MARGIN = 8

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

    font = get_font(11)
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
    ko_font = get_font(11)
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
    ko_font = get_font(11)  # Smaller font for Korean
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
    small_font = get_font(9)
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
    temp_img = Image.new('RGB', (WIDTH, 2000), BG_COLOR)
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

    # Use imageio-ffmpeg to save as MP4
    imageio.mimsave(mp4_path, frames, fps=FPS, codec='libx264', quality=8)
    print(f"Generated: {mp4_path}")


def generate_chunking_video(messages, single_response, output_path, with_chunking=True):
    """
    Generate video showing chunked vs single response
    Uses 'chunkDelay' field for chunked timing
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    font = get_font()
    frames = []

    messages_shown = []
    last_ts = 0

    if with_chunking:
        # Show messages with chunked agent responses
        for i, msg in enumerate(messages):
            msg_ts = msg.get('ts', i * 2000) * TIMING_MULTIPLIER

            # Calculate delay
            if i > 0:
                delay_ms = msg_ts - last_ts
                delay_seconds = min(delay_ms / 1000.0, 6.0)

                if delay_seconds > 0.1:
                    delay_frames = int(delay_seconds * FPS)
                    frame = create_frame(messages_shown, font)
                    for _ in range(delay_frames):
                        frames.append(frame)

            last_ts = msg_ts

            messages_shown.append(msg)
            frame = create_frame(messages_shown, font)

            # Hold time
            hold_time = calculate_reading_time(msg['text']) * 0.5
            hold_frames = max(1, int(hold_time * FPS))
            for _ in range(hold_frames):
                frames.append(frame)
    else:
        # Show user messages first
        user_messages = [m for m in messages if m['role'] == 'user']
        for i, msg in enumerate(user_messages):
            messages_shown.append(msg)
            frame = create_frame(messages_shown, font)
            hold_time = calculate_reading_time(msg['text']) * 0.6
            for _ in range(int(hold_time * FPS)):
                frames.append(frame)

        # Brief delay then single long response
        for _ in range(int(1.0 * FPS)):
            frames.append(create_frame(messages_shown, font))

        # Add single long response
        single_msg = {'role': 'agent', 'sender': 'helper', 'text': single_response}
        messages_shown.append(single_msg)

        frame = create_frame(messages_shown, font)
        hold_time = calculate_reading_time(single_response) * 0.4
        for _ in range(int(hold_time * FPS)):
            frames.append(frame)

    # Hold final state
    for _ in range(int(3 * FPS)):
        frames.append(frames[-1])

    # Save as MP4 video (no loop issue)
    mp4_path = output_path.replace('.gif', '.mp4')
    if not mp4_path.endswith('.mp4'):
        mp4_path = output_path

    # Use imageio-ffmpeg to save as MP4
    imageio.mimsave(mp4_path, frames, fps=FPS, codec='libx264', quality=8)
    print(f"Generated: {mp4_path}")


def main():
    """Generate all survey GIFs using real IRC conversation data"""

    print("Generating survey GIFs with real IRC conversations...")
    print(f"Output directory: {OUTPUT_DIR}")

    # ============================================
    # Set 5: System suspend/resume + dark theme (timing comparison)
    # Source: Lines 286-334 from ubuntu_merged.txt
    # ============================================
    timing_1_messages = [
        {'role': 'user', 'sender': 'arkanoid', 'text': 'My system fails to resume to gnome desktop after suspend. When it resumes gnome shell is a black screen\n절전 모드 후 그놈 데스크톱이 안 켜져요. 검은 화면만 나와요', 'ts': 0},
        {'role': 'user', 'sender': 'derek-shnosh', 'text': 'In Ubuntu 23.10, some apps are not honoring the dark theme for parts of the window\n우분투 23.10에서 일부 앱이 다크 테마를 적용 안 해요', 'ts': 2000},
        {'role': 'user', 'sender': 'lotuspsychje', 'text': 'arkanoid: can you share your dmesg with the volunteers please\narkanoid: dmesg 로그 공유해 주세요', 'ts': 4000},
        {'role': 'agent', 'sender': 'helper', 'text': 'arkanoid: Try loginctl terminate-session\narkanoid: loginctl terminate-session 해보세요', 'ts': 8000, 'noTimingTs': 4100},
        {'role': 'agent', 'sender': 'helper', 'text': 'This resets the gnome session without reboot\n재부팅 없이 그놈 세션을 재시작해요', 'ts': 10500, 'noTimingTs': 4200},
        {'role': 'user', 'sender': 'arkanoid', 'text': 'just tried systemctl restart systemd-logind, got visual login back\nsystemctl restart systemd-logind 했더니 로그인 화면 돌아왔어요', 'ts': 13000},
        {'role': 'user', 'sender': 'ioria', 'text': "derek-shnosh: run in terminal 'G_MESSAGES_DEBUG=all hexchat'\nderek-shnosh: 터미널에서 G_MESSAGES_DEBUG=all hexchat 실행해봐요", 'ts': 15000},
        {'role': 'agent', 'sender': 'helper', 'text': 'derek-shnosh: Check your gtk-theme setting\nderek-shnosh: gtk-theme 설정 확인해보세요', 'ts': 19000, 'noTimingTs': 15100},
        {'role': 'user', 'sender': 'derek-shnosh', 'text': 'Figured it out! gtk-theme was set to Yaru-blue, changed to Yaru-blue-dark\n해결! gtk-theme이 Yaru-blue였는데 Yaru-blue-dark로 바꿨어요', 'ts': 23000},
        {'role': 'agent', 'sender': 'helper', 'text': 'arkanoid: Good, the session restart worked\narkanoid: 세션 재시작이 됐네요', 'ts': 26000, 'noTimingTs': 23100},
        {'role': 'agent', 'sender': 'helper', 'text': 'This is likely bug #1968907 in gnome-shell\ngnome-shell 버그 #1968907일 가능성이 높아요', 'ts': 28500, 'noTimingTs': 23200},
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
    # Set 6: Terminal ctrl-a + phased updates (timing comparison)
    # Source: Lines 450-499, 623-645 from ubuntu_merged.txt
    # ============================================
    timing_2_messages = [
        {'role': 'user', 'sender': 'en1gma', 'text': 'ctrl-a does not copy all in terminal but works in firefox. why?\n터미널에서 ctrl-a가 전체선택이 안 되는데 파이어폭스에선 돼요. 왜죠?', 'ts': 0},
        {'role': 'user', 'sender': 'ELFrederich', 'text': '7 packages can be upgraded but apt upgrade keeps them back\n7개 패키지 업그레이드 가능한데 apt upgrade가 보류해요', 'ts': 2500},
        {'role': 'user', 'sender': 'pragmaticenigma', 'text': "ctrl+a doesn't copy anything, it's select all. Terminal doesn't support that command\nctrl+a는 복사가 아니라 전체선택이에요. 터미널은 그 명령을 지원 안 해요", 'ts': 5000},
        {'role': 'agent', 'sender': 'helper', 'text': 'en1gma: Terminal apps handle shortcuts differently\nen1gma: 터미널 앱은 단축키를 다르게 처리해요', 'ts': 9000, 'noTimingTs': 5100},
        {'role': 'agent', 'sender': 'helper', 'text': 'ctrl-a in bash moves cursor to line start\nbash에서 ctrl-a는 커서를 줄 처음으로 이동해요', 'ts': 11500, 'noTimingTs': 5200},
        {'role': 'user', 'sender': 'lotuspsychje', 'text': 'ELFrederich: those are phased updates\nELFrederich: 그건 단계적 업데이트예요', 'ts': 14000},
        {'role': 'user', 'sender': 'en1gma', 'text': 'i was reading that /etc/bash.bashrc needs to be edited\n/etc/bash.bashrc를 수정해야 한다고 읽었어요', 'ts': 16500},
        {'role': 'agent', 'sender': 'helper', 'text': 'ELFrederich: Phased updates roll out gradually\nELFrederich: 단계적 업데이트는 점진적으로 배포돼요', 'ts': 20500, 'noTimingTs': 16600},
        {'role': 'agent', 'sender': 'helper', 'text': "Just wait a few days and they'll be available\n며칠 기다리면 사용 가능해져요", 'ts': 23000, 'noTimingTs': 16700},
        {'role': 'user', 'sender': 'pragmaticenigma', 'text': "en1gma: don't edit that file, it could break your system\nen1gma: 그 파일 수정하지 마세요, 시스템 망가질 수 있어요", 'ts': 26000},
        {'role': 'agent', 'sender': 'helper', 'text': "en1gma: Edit ~/.bashrc instead, it's safer\nen1gma: 대신 ~/.bashrc를 수정하세요, 더 안전해요", 'ts': 30000, 'noTimingTs': 26100},
        {'role': 'user', 'sender': 'en1gma', 'text': 'ok thanks, will try that\n네 감사해요, 해볼게요', 'ts': 33000},
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

    # ============================================
    # Set 9: Network bridge setup (chunking comparison)
    # Source: Lines 521-586 from ubuntu_merged.txt
    # ============================================
    chunking_1_messages = [
        {'role': 'user', 'sender': 'alcosta', 'text': 'I want to setup bridged networking with Virtual Machines on ubuntu 20.04\n우분투 20.04에서 가상머신 브리지 네트워킹 설정하고 싶어요', 'ts': 0},
        {'role': 'user', 'sender': 'sarnold', 'text': 'usual approach is to create a new bridge, add physical NICs, set IP on bridge\n보통 새 브리지 만들고, 물리 NIC 추가하고, 브리지에 IP 설정해요', 'ts': 3000},
        {'role': 'user', 'sender': 'leftyfb', 'text': 'alcosta: they should only need a network bridge\nalcosta: 네트워크 브리지만 있으면 돼요', 'ts': 5500},
        {'role': 'agent', 'sender': 'helper', 'text': 'alcosta: First run nmcli con add ifname br0 type bridge con-name br0\nalcosta: 먼저 nmcli con add ifname br0 type bridge con-name br0 실행', 'ts': 9500},
        {'role': 'agent', 'sender': 'helper', 'text': 'Then: nmcli con add type bridge-slave ifname enp6s0 master br0\n그다음: nmcli con add type bridge-slave ifname enp6s0 master br0', 'ts': 12500},
        {'role': 'user', 'sender': 'alcosta', 'text': "Something isn't right, the commands don't match my case\n뭔가 이상해요, 명령어가 제 경우에 안 맞아요", 'ts': 16000},
        {'role': 'user', 'sender': 'leftyfb', 'text': 'it looks like you did lots of bad things, delete the bridge interfaces\n뭔가 많이 잘못된 것 같네요, 브리지 인터페이스 삭제하세요', 'ts': 18500},
        {'role': 'agent', 'sender': 'helper', 'text': 'alcosta: Check nmcli con show to see current connections\nalcosta: nmcli con show로 현재 연결 확인하세요', 'ts': 22500},
        {'role': 'agent', 'sender': 'helper', 'text': 'Delete the broken ones with nmcli con delete <uuid>\n망가진 건 nmcli con delete <uuid>로 삭제하세요', 'ts': 25500},
        {'role': 'agent', 'sender': 'helper', 'text': 'Then start fresh with the bridge commands\n그다음 브리지 명령어로 새로 시작하세요', 'ts': 28500},
        {'role': 'user', 'sender': 'alcosta', 'text': 'OK, deleted them. Now what?\n네, 삭제했어요. 이제 뭐해요?', 'ts': 32000},
        {'role': 'user', 'sender': 'leftyfb', 'text': 'reboot, then br0 should have an IP address\n재부팅하면 br0에 IP 주소가 있을 거예요', 'ts': 34500},
    ]
    chunking_1_single = "alcosta: To set up bridged networking, run these commands in order: First, nmcli con add ifname br0 type bridge con-name br0. Then nmcli con add type bridge-slave ifname enp6s0 master br0. If you have existing broken bridge configs, delete them with nmcli con delete <uuid> first. You can check current connections with nmcli con show. After creating the bridge, reboot and br0 should get an IP address. Then configure your VMs to use br0 as the network interface.\nalcosta: 브리지 네트워킹 설정하려면 순서대로: 먼저 nmcli con add ifname br0 type bridge con-name br0. 그다음 nmcli con add type bridge-slave ifname enp6s0 master br0. 기존 브리지 설정이 망가졌으면 nmcli con delete <uuid>로 먼저 삭제. nmcli con show로 현재 연결 확인 가능. 브리지 만든 후 재부팅하면 br0에 IP 주소가 할당돼요. 그다음 VM을 br0 네트워크 인터페이스로 설정하세요."

    generate_chunking_video(
        chunking_1_messages,
        chunking_1_single,
        os.path.join(OUTPUT_DIR, 'chunking_full_1.mp4'),
        with_chunking=True
    )
    generate_chunking_video(
        chunking_1_messages,
        chunking_1_single,
        os.path.join(OUTPUT_DIR, 'chunking_nochunk_1.mp4'),
        with_chunking=False
    )

    # ============================================
    # Set 10: GRUB/EFI boot repair (chunking comparison)
    # Source: Lines 716-799 from ubuntu_merged.txt
    # ============================================
    chunking_2_messages = [
        {'role': 'user', 'sender': 'cahoots', 'text': 'grub-install gives warning: EFI variables cannot be set on this system\ngrub-install이 경고해요: 이 시스템에서 EFI 변수를 설정할 수 없다고', 'ts': 0},
        {'role': 'user', 'sender': 'EriC^^', 'text': 'in which mode are you booting? uefi? csm legacy?\n어떤 모드로 부팅하고 있어요? uefi? csm legacy?', 'ts': 3000},
        {'role': 'user', 'sender': 'cahoots', 'text': "I'm in uefi, csm legacy is disabled\nuefi예요, csm legacy는 비활성화됐어요", 'ts': 6000},
        {'role': 'agent', 'sender': 'helper', 'text': 'cahoots: First mount your root partition\ncahoots: 먼저 루트 파티션을 마운트하세요', 'ts': 10000},
        {'role': 'agent', 'sender': 'helper', 'text': 'sudo mount /dev/nvme0n1p3 /mnt\nsudo mount /dev/nvme0n1p3 /mnt', 'ts': 13500},
        {'role': 'user', 'sender': 'EriC^^', 'text': 'then run the for loop to bind mount /dev /proc /sys /run\n그다음 for 루프로 /dev /proc /sys /run 바인드 마운트하세요', 'ts': 17000},
        {'role': 'agent', 'sender': 'helper', 'text': 'Next: for i in /dev /proc /sys /run; do sudo mount -R $i /mnt$i; done\n다음: for i in /dev /proc /sys /run; do sudo mount -R $i /mnt$i; done', 'ts': 21000},
        {'role': 'agent', 'sender': 'helper', 'text': 'Then: sudo chroot /mnt\n그다음: sudo chroot /mnt', 'ts': 24500},
        {'role': 'user', 'sender': 'cahoots', 'text': "mount /boot/efi says can't find UUID\nmount /boot/efi가 UUID를 찾을 수 없다고 해요", 'ts': 28000},
        {'role': 'user', 'sender': 'EriC^^', 'text': 'the uuid changed, edit /etc/fstab with the new one from blkid\nuuid가 바뀌었어요, blkid에서 새 uuid로 /etc/fstab 수정하세요', 'ts': 31000},
        {'role': 'agent', 'sender': 'helper', 'text': 'cahoots: Run blkid to find the new UUID\ncahoots: blkid로 새 UUID 확인하세요', 'ts': 35000},
        {'role': 'agent', 'sender': 'helper', 'text': 'Update /etc/fstab, then mount /boot/efi\n/etc/fstab 수정하고 mount /boot/efi 하세요', 'ts': 38500},
        {'role': 'agent', 'sender': 'helper', 'text': 'Finally: grub-install --target=x86_64-efi\n마지막으로: grub-install --target=x86_64-efi', 'ts': 42000},
        {'role': 'user', 'sender': 'cahoots', 'text': 'installation finished, no error reported!\n설치 완료, 에러 없어요!', 'ts': 46000},
    ]
    chunking_2_single = "cahoots: To fix grub installation, follow these steps: Mount your root partition with sudo mount /dev/nvme0n1p3 /mnt. Then bind mount the virtual filesystems with for i in /dev /proc /sys /run; do sudo mount -R $i /mnt$i; done. Chroot into the system with sudo chroot /mnt. If mount /boot/efi fails due to UUID mismatch, run blkid to find the new UUID, update /etc/fstab with the correct UUID, then mount /boot/efi again. Finally run grub-install --target=x86_64-efi to install grub. After that, exit the chroot and reboot.\ncahoots: grub 설치를 수정하려면: sudo mount /dev/nvme0n1p3 /mnt로 루트 파티션 마운트. 그다음 for i in /dev /proc /sys /run; do sudo mount -R $i /mnt$i; done으로 가상 파일시스템 바인드 마운트. sudo chroot /mnt로 시스템에 진입. UUID 불일치로 mount /boot/efi가 실패하면 blkid로 새 UUID 확인, /etc/fstab 수정 후 다시 마운트. 마지막으로 grub-install --target=x86_64-efi로 grub 설치. 그 후 chroot 나와서 재부팅."

    generate_chunking_video(
        chunking_2_messages,
        chunking_2_single,
        os.path.join(OUTPUT_DIR, 'chunking_full_2.mp4'),
        with_chunking=True
    )
    generate_chunking_video(
        chunking_2_messages,
        chunking_2_single,
        os.path.join(OUTPUT_DIR, 'chunking_nochunk_2.mp4'),
        with_chunking=False
    )

    print("\nAll GIFs generated successfully!")
    print(f"Total: 8 GIFs in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
