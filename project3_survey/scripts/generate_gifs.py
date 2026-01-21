#!/usr/bin/env python3
"""
GIF/Video Generator for Survey

Generates comparison videos showing:
1. Timing difference (with delays vs instant)
2. Chunking difference (split messages vs single long message)

Output: MP4 videos that show chat messages appearing over time
"""

import os
import json
from datetime import datetime
from PIL import Image, ImageDraw, ImageFont
import imageio
import numpy as np

# Configuration
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'gifs')
FPS = 10  # Frames per second
WIDTH = 600
HEIGHT = 400
FONT_SIZE = 14
BUBBLE_PADDING = 10
MESSAGE_GAP = 15

# Colors
BG_COLOR = (248, 249, 250)
USER_BUBBLE = (227, 242, 253)
AGENT_BUBBLE = (102, 126, 234)
TEXT_COLOR = (51, 51, 51)
AGENT_TEXT = (255, 255, 255)
SENDER_COLOR = (102, 102, 102)


def get_font(size=FONT_SIZE):
    """Get a font, falling back to default if needed"""
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


def draw_message(draw, y_pos, message, font, is_agent=False):
    """Draw a single chat message bubble"""
    max_bubble_width = WIDTH - 100

    # Wrap text
    lines = wrap_text(message['text'], font, max_bubble_width - 2 * BUBBLE_PADDING, draw)

    # Calculate bubble size
    line_height = font.size + 4
    text_height = len(lines) * line_height
    bubble_height = text_height + 2 * BUBBLE_PADDING + 20  # Extra for sender name

    # Calculate text width
    max_line_width = 0
    for line in lines:
        bbox = draw.textbbox((0, 0), line, font=font)
        max_line_width = max(max_line_width, bbox[2] - bbox[0])

    bubble_width = max_line_width + 2 * BUBBLE_PADDING

    # Position
    if is_agent:
        x = WIDTH - bubble_width - 20
        bubble_color = AGENT_BUBBLE
        text_color = AGENT_TEXT
    else:
        x = 20
        bubble_color = USER_BUBBLE
        text_color = TEXT_COLOR

    # Draw bubble
    draw.rounded_rectangle(
        [x, y_pos, x + bubble_width, y_pos + bubble_height],
        radius=10,
        fill=bubble_color
    )

    # Draw sender name
    sender = message.get('sender', 'Agent' if is_agent else 'User')
    small_font = get_font(10)
    draw.text((x + BUBBLE_PADDING, y_pos + 5), sender, font=small_font, fill=SENDER_COLOR if not is_agent else (200, 200, 255))

    # Draw text
    text_y = y_pos + 20
    for line in lines:
        draw.text((x + BUBBLE_PADDING, text_y), line, font=font, fill=text_color)
        text_y += line_height

    return y_pos + bubble_height + MESSAGE_GAP


def create_frame(messages_to_show, font):
    """Create a single frame with visible messages"""
    img = Image.new('RGB', (WIDTH, HEIGHT), BG_COLOR)
    draw = ImageDraw.Draw(img)

    y_pos = 20
    for msg in messages_to_show:
        is_agent = msg['role'] == 'agent'
        y_pos = draw_message(draw, y_pos, msg, font, is_agent)

        if y_pos > HEIGHT - 50:
            break

    return np.array(img)


def calculate_reading_time(text):
    """Calculate realistic reading time based on text length (words per minute)"""
    words = len(text.split())
    # Average reading speed: ~200 WPM, but for chat context slower (~150 WPM)
    # Minimum 1.5 seconds, maximum 5 seconds
    reading_time = max(1.5, min(5.0, words / 2.5))
    return reading_time


def calculate_typing_time(text):
    """Calculate realistic typing time based on text length"""
    chars = len(text)
    # Average typing speed: ~40 WPM = ~200 CPM = ~3.3 CPS
    # For chat, faster: ~5 CPS, with minimum 1 second
    typing_time = max(1.0, min(4.0, chars / 5.0))
    return typing_time


def generate_timing_video(messages, delays, output_path, with_timing=True):
    """
    Generate video showing messages appearing with or without timing delays

    Args:
        messages: List of message dicts
        delays: List of delays in seconds for each message (used as base for agent)
        output_path: Output file path
        with_timing: If True, use natural delays for ALL speakers; if False, instant appearance
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    font = get_font()
    frames = []

    messages_shown = []

    for i, msg in enumerate(messages):
        # Calculate natural delay based on speaker and context
        if with_timing and i > 0:
            prev_msg = messages[i - 1]

            if msg['role'] == 'user':
                # User needs time to read previous message, then type
                read_time = calculate_reading_time(prev_msg['text'])
                type_time = calculate_typing_time(msg['text'])
                natural_delay = read_time + type_time
            else:
                # Agent "thinks" then responds - use provided delay or calculate
                if i < len(delays) and delays[i] > 0:
                    natural_delay = delays[i]
                else:
                    # Calculate based on response complexity
                    natural_delay = calculate_typing_time(msg['text']) + 1.0

            # Show current state for delay duration
            delay_frames = int(natural_delay * FPS)
            frame = create_frame(messages_shown, font)
            for _ in range(delay_frames):
                frames.append(frame)

        # Add message
        messages_shown.append(msg)

        # Show new message appearing
        frame = create_frame(messages_shown, font)

        # Hold for a moment to let viewer read
        hold_time = calculate_reading_time(msg['text']) * 0.5 if with_timing else 0.1
        hold_frames = int(hold_time * FPS)
        for _ in range(hold_frames):
            frames.append(frame)

    # Hold final state
    final_hold = int(2 * FPS)
    for _ in range(final_hold):
        frames.append(frames[-1] if frames else create_frame([], font))

    # Save as GIF
    gif_path = output_path.replace('.mp4', '.gif')
    imageio.mimsave(gif_path, frames, duration=1000/FPS, loop=0)
    print(f"Generated: {gif_path}")


def generate_chunking_video(user_msg, agent_response, output_path, with_chunking=True):
    """
    Generate video showing chunked vs single response

    Args:
        user_msg: User message dict
        agent_response: Full agent response text
        output_path: Output file path
        with_chunking: If True, split into chunks; if False, single message
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    font = get_font()
    frames = []

    # Show user message first with natural reading time
    messages_shown = [user_msg]
    frame = create_frame(messages_shown, font)
    read_time = calculate_reading_time(user_msg['text'])
    for _ in range(int(read_time * FPS)):
        frames.append(frame)

    if with_chunking:
        # Split response into chunks (at sentence boundaries, max ~12 words)
        words = agent_response.split()
        chunks = []
        current_chunk = []

        for word in words:
            current_chunk.append(word)
            if len(current_chunk) >= 12 or word.endswith(('.', '!', '?')):
                chunks.append(' '.join(current_chunk))
                current_chunk = []

        if current_chunk:
            chunks.append(' '.join(current_chunk))

        # Show chunks with natural typing delays
        for i, chunk in enumerate(chunks):
            # Add thinking/typing delay before chunk appears
            if i == 0:
                # First chunk: agent processing time
                thinking_delay = 2.0
            else:
                # Subsequent chunks: typing delay based on length
                thinking_delay = calculate_typing_time(chunk)

            # Show delay (thinking indicator could be added here)
            delay_frame = create_frame(messages_shown, font)
            for _ in range(int(thinking_delay * FPS)):
                frames.append(delay_frame)

            chunk_msg = {'role': 'agent', 'sender': 'helper', 'text': chunk}
            messages_shown.append(chunk_msg)

            frame = create_frame(messages_shown, font)

            # Hold for reading time
            hold_time = calculate_reading_time(chunk) * 0.6
            for _ in range(int(hold_time * FPS)):
                frames.append(frame)
    else:
        # Single message - appears after brief processing delay
        processing_delay = 0.3  # Instant appearance for comparison
        for _ in range(int(processing_delay * FPS)):
            frames.append(create_frame(messages_shown, font))

        agent_msg = {'role': 'agent', 'sender': 'helper', 'text': agent_response}
        messages_shown.append(agent_msg)

        frame = create_frame(messages_shown, font)
        # Hold for reading time
        hold_time = calculate_reading_time(agent_response) * 0.5
        for _ in range(int(hold_time * FPS)):
            frames.append(frame)

    # Hold final state
    for _ in range(int(2 * FPS)):
        frames.append(frames[-1])

    # Save as GIF
    gif_path = output_path.replace('.mp4', '.gif')
    imageio.mimsave(gif_path, frames, duration=1000/FPS, loop=0)
    print(f"Generated: {gif_path}")


def main():
    """Generate all survey videos"""

    print("Generating survey GIF/videos...")
    print(f"Output directory: {OUTPUT_DIR}")

    # ============================================
    # Timing comparison videos (Sets 5-6)
    # ============================================

    # Set 5: Quick help exchange
    timing_messages_1 = [
        {'role': 'user', 'sender': 'user1', 'text': 'How do I restart nginx?'},
        {'role': 'agent', 'sender': 'helper', 'text': 'sudo systemctl restart nginx'},
        {'role': 'user', 'sender': 'user1', 'text': 'thanks!'},
        {'role': 'agent', 'sender': 'helper', 'text': 'No problem.'},
    ]
    timing_delays_1 = [0, 5, 0, 3]  # delays in seconds

    generate_timing_video(
        timing_messages_1,
        timing_delays_1,
        os.path.join(OUTPUT_DIR, 'timing_full_1.mp4'),
        with_timing=True
    )
    generate_timing_video(
        timing_messages_1,
        [0, 0, 0, 0],
        os.path.join(OUTPUT_DIR, 'timing_notiming_1.mp4'),
        with_timing=False
    )

    # Set 6: Technical question
    timing_messages_2 = [
        {'role': 'user', 'sender': 'newbie', 'text': 'What command shows running processes?'},
        {'role': 'agent', 'sender': 'helper', 'text': 'Use ps aux or htop.'},
        {'role': 'user', 'sender': 'newbie', 'text': 'whats the difference?'},
        {'role': 'agent', 'sender': 'helper', 'text': 'htop is interactive with colors.'},
        {'role': 'agent', 'sender': 'helper', 'text': 'ps is just text output.'},
    ]
    timing_delays_2 = [0, 7, 0, 6, 2]

    generate_timing_video(
        timing_messages_2,
        timing_delays_2,
        os.path.join(OUTPUT_DIR, 'timing_full_2.mp4'),
        with_timing=True
    )
    generate_timing_video(
        timing_messages_2,
        [0, 0, 0, 0, 0],
        os.path.join(OUTPUT_DIR, 'timing_notiming_2.mp4'),
        with_timing=False
    )

    # ============================================
    # Chunking comparison videos (Sets 9-10)
    # ============================================

    # Set 9: nginx configuration
    user_msg_1 = {'role': 'user', 'sender': 'dev1', 'text': 'How do I set up nginx as reverse proxy?'}
    agent_response_1 = "First install nginx with apt. Then edit /etc/nginx/sites-available/default. Add proxy_pass http://localhost:3000 in the location block. Finally run nginx -t to test and systemctl reload nginx."

    generate_chunking_video(
        user_msg_1,
        agent_response_1,
        os.path.join(OUTPUT_DIR, 'chunking_full_1.mp4'),
        with_chunking=True
    )
    generate_chunking_video(
        user_msg_1,
        agent_response_1,
        os.path.join(OUTPUT_DIR, 'chunking_nochunk_1.mp4'),
        with_chunking=False
    )

    # Set 10: system update steps
    user_msg_2 = {'role': 'user', 'sender': 'user2', 'text': 'How do I fully update my Ubuntu system?'}
    agent_response_2 = "Run sudo apt update to refresh package lists. Then sudo apt upgrade for regular updates. Use sudo apt full-upgrade for major updates. Finally sudo apt autoremove to clean up old packages."

    generate_chunking_video(
        user_msg_2,
        agent_response_2,
        os.path.join(OUTPUT_DIR, 'chunking_full_2.mp4'),
        with_chunking=True
    )
    generate_chunking_video(
        user_msg_2,
        agent_response_2,
        os.path.join(OUTPUT_DIR, 'chunking_nochunk_2.mp4'),
        with_chunking=False
    )

    print("\nAll videos generated successfully!")
    print(f"Total: 8 videos in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
