import cv2
import numpy as np

def draw_fps(frame: np.ndarray, fps: float):
    """Compatibility helper (FPS is now integrated into header)."""
    pass

def draw_header(frame: np.ndarray, title: str = "Sign Language AI Recognizer (A-Z)", fps: float = 0.0):
    """
    Renders a modern, glassmorphic top header bar with system status and FPS.
    """
    h, w, _ = frame.shape
    bar_h = 58

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, bar_h), (18, 20, 26), -1)
    cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
    cv2.line(frame, (0, bar_h), (w, bar_h), (50, 60, 75), 1)

    # Glowing Cyan Accent Dot
    cv2.circle(frame, (28, 28), 6, (0, 240, 255), -1, cv2.LINE_AA)
    cv2.circle(frame, (28, 28), 9, (0, 240, 255), 1, cv2.LINE_AA)

    # Title
    cv2.putText(
        frame,
        title,
        (48, 35),
        cv2.FONT_HERSHEY_DUPLEX,
        0.85,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )

    # System Status Chips (Right Side)
    # Camera Status Chip
    chip_cam_x = w - 240
    cv2.rectangle(frame, (chip_cam_x, 14), (chip_cam_x + 95, 42), (32, 40, 48), -1)
    cv2.circle(frame, (chip_cam_x + 14, 28), 4, (50, 230, 80), -1, cv2.LINE_AA)
    cv2.putText(
        frame,
        "LIVE FEED",
        (chip_cam_x + 24, 33),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (200, 230, 210),
        1,
        cv2.LINE_AA
    )

    # FPS Chip
    chip_fps_x = w - 130
    cv2.rectangle(frame, (chip_fps_x, 14), (chip_fps_x + 110, 42), (32, 40, 48), -1)
    fps_color = (50, 230, 80) if fps >= 30 else (0, 210, 255)
    cv2.putText(
        frame,
        f"{fps:.1f} FPS",
        (chip_fps_x + 18, 33),
        cv2.FONT_HERSHEY_DUPLEX,
        0.55,
        fps_color,
        1,
        cv2.LINE_AA
    )

def draw_prediction_card(
    frame: np.ndarray,
    predicted_label: str,
    confidence: float,
    handedness: str = None,
    top_k_list: list = None,
    hold_ratio: float = 0.0
):
    """
    Renders an advanced AI Inference HUD showing primary prediction,
    confidence gradient bar, top-3 candidates, and commit progress timer.
    """
    h, w, _ = frame.shape
    card_x, card_y = 20, 72
    card_w, card_h = 340, 230

    # Glass Card Background
    overlay = frame.copy()
    cv2.rectangle(overlay, (card_x, card_y), (card_x + card_w, card_y + card_h), (20, 24, 32), -1)
    cv2.addWeighted(overlay, 0.88, frame, 0.12, 0, frame)

    # Outer border with highlight
    cv2.rectangle(frame, (card_x, card_y), (card_x + card_w, card_y + card_h), (55, 68, 88), 1)
    cv2.line(frame, (card_x, card_y), (card_x + 40, card_y), (0, 240, 255), 2)
    cv2.line(frame, (card_x, card_y), (card_x, card_y + 40), (0, 240, 255), 2)

    # Card Title & Handedness Badge
    cv2.putText(
        frame,
        "AI GESTURE INFERENCE",
        (card_x + 16, card_y + 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.44,
        (130, 150, 180),
        1,
        cv2.LINE_AA
    )

    if handedness:
        badge_text = f"{handedness.upper()} HAND"
        badge_w = 90
        badge_x = card_x + card_w - badge_w - 14
        cv2.rectangle(frame, (badge_x, card_y + 12), (badge_x + badge_w, card_y + 30), (38, 48, 64), -1)
        cv2.putText(
            frame,
            badge_text,
            (badge_x + 10, card_y + 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            (0, 230, 255),
            1,
            cv2.LINE_AA
        )

    # Big Detected Letter
    display_sign = predicted_label if predicted_label and predicted_label != "None" else "—"
    sign_color = (0, 255, 140) if confidence >= 0.75 else (0, 200, 255) if confidence >= 0.50 else (140, 140, 150)
    
    cv2.putText(
        frame,
        display_sign,
        (card_x + 16, card_y + 78),
        cv2.FONT_HERSHEY_DUPLEX,
        1.7,
        sign_color,
        2,
        cv2.LINE_AA
    )

    # Primary Confidence Percentage
    conf_pct = int(confidence * 100)
    cv2.putText(
        frame,
        f"{conf_pct}%",
        (card_x + 95, card_y + 72),
        cv2.FONT_HERSHEY_DUPLEX,
        0.75,
        (220, 230, 240),
        1,
        cv2.LINE_AA
    )
    cv2.putText(
        frame,
        "CONFIDENCE",
        (card_x + 95, card_y + 88),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.36,
        (130, 145, 165),
        1,
        cv2.LINE_AA
    )

    # Primary Confidence Bar
    bar_x = card_x + 16
    bar_y = card_y + 102
    bar_w = card_w - 32
    bar_h = 10
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (40, 48, 60), -1)
    
    fill_w = int(bar_w * max(0.0, min(1.0, confidence)))
    if confidence >= 0.75:
        bar_col = (0, 230, 100)
    elif confidence >= 0.50:
        bar_col = (0, 195, 255)
    else:
        bar_col = (70, 70, 230)

    if fill_w > 0:
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), bar_col, -1)

    # Commit Countdown / Hold Bar
    hold_y = bar_y + 24
    cv2.putText(
        frame,
        "STABILIZE TO COMMIT:",
        (bar_x, hold_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.38,
        (140, 160, 180),
        1,
        cv2.LINE_AA
    )

    hold_bar_w = bar_w
    hold_bar_y = hold_y + 6
    hold_bar_h = 6
    cv2.rectangle(frame, (bar_x, hold_bar_y), (bar_x + hold_bar_w, hold_bar_y + hold_bar_h), (35, 42, 54), -1)
    fill_hold = int(hold_bar_w * max(0.0, min(1.0, hold_ratio)))
    if fill_hold > 0:
        hold_col = (0, 255, 230) if hold_ratio >= 0.9 else (0, 200, 255)
        cv2.rectangle(frame, (bar_x, hold_bar_y), (bar_x + fill_hold, hold_bar_y + hold_bar_h), hold_col, -1)

    # Top-3 Predictions Sub-Panel
    pred_y = hold_bar_y + 24
    cv2.line(frame, (bar_x, pred_y - 8), (bar_x + bar_w, pred_y - 8), (40, 50, 65), 1)
    cv2.putText(
        frame,
        "TOP CANDIDATES:",
        (bar_x, pred_y + 4),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.38,
        (130, 145, 165),
        1,
        cv2.LINE_AA
    )

    if top_k_list:
        sub_x = bar_x
        for rank, (cls_name, prob) in enumerate(top_k_list[:3]):
            item_y = pred_y + 24 + rank * 18
            cv2.putText(
                frame,
                f"#{rank + 1} {cls_name}",
                (sub_x, item_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.44,
                (230, 235, 245) if rank == 0 else (160, 175, 195),
                1,
                cv2.LINE_AA
            )
            # Mini bar
            mini_x = sub_x + 65
            mini_w = 170
            cv2.rectangle(frame, (mini_x, item_y - 8), (mini_x + mini_w, item_y - 2), (38, 46, 58), -1)
            mini_fill = int(mini_w * max(0.0, min(1.0, prob)))
            mini_color = (0, 230, 100) if rank == 0 else (0, 180, 240)
            if mini_fill > 0:
                cv2.rectangle(frame, (mini_x, item_y - 8), (mini_x + mini_fill, item_y - 2), mini_color, -1)
            cv2.putText(
                frame,
                f"{int(prob * 100)}%",
                (mini_x + mini_w + 10, item_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.40,
                (170, 185, 205),
                1,
                cv2.LINE_AA
            )

def draw_sentence_panel(frame: np.ndarray, sentence: str, show_cursor: bool = True):
    """
    Renders an elegant glass dock at the bottom of the screen showing accumulated sentence,
    word count, character metrics, and interactive hotkeys.
    """
    h, w, _ = frame.shape
    dock_h = 95
    dock_y = h - dock_h

    # Translucent Bottom Dock
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, dock_y), (w, h), (16, 18, 24), -1)
    cv2.addWeighted(overlay, 0.90, frame, 0.10, 0, frame)
    cv2.line(frame, (0, dock_y), (w, dock_y), (45, 55, 70), 1)

    # Header label & Word Count Stats
    words = sentence.strip().split() if sentence.strip() else []
    word_count = len(words)
    char_count = len(sentence)

    cv2.putText(
        frame,
        "ACCUMULATED SENTENCE BUFFER",
        (24, dock_y + 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (0, 230, 255),
        1,
        cv2.LINE_AA
    )

    stats_text = f"Letters: {char_count}  |  Words: {word_count}"
    cv2.putText(
        frame,
        stats_text,
        (w - 230, dock_y + 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (140, 155, 175),
        1,
        cv2.LINE_AA
    )

    # Sentence Text Display
    cursor = " |" if show_cursor else ""
    if sentence.strip():
        display_text = sentence + cursor
        text_color = (255, 255, 255)
    else:
        display_text = "(Sign letters to form words. Hold steady for ~0.4s to commit)" + cursor
        text_color = (110, 125, 140)

    cv2.putText(
        frame,
        display_text,
        (24, dock_y + 58),
        cv2.FONT_HERSHEY_DUPLEX,
        0.85,
        text_color,
        2,
        cv2.LINE_AA
    )

    # Interactive Keyboard Controls Strip
    shortcuts = [
        "[SPACE] Space",
        "[BACKSPACE/D] Delete",
        "[S] Speak",
        "[C] Clear",
        "[Q] Quit"
    ]
    shortcut_str = "    •    ".join(shortcuts)
    cv2.putText(
        frame,
        shortcut_str,
        (24, dock_y + 83),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.39,
        (130, 145, 165),
        1,
        cv2.LINE_AA
    )

def draw_hand_bounding_box(frame: np.ndarray, hand_landmarks, label: str = ""):
    """
    Renders a corner-bracket target reticle box around the hand with sign label badge.
    """
    if not hand_landmarks:
        return

    h, w, _ = frame.shape
    lms = hand_landmarks.landmark if hasattr(hand_landmarks, "landmark") else hand_landmarks
    xs = [int(lm.x * w) for lm in lms]
    ys = [int(lm.y * h) for lm in lms]

    x_min, x_max = max(10, min(xs) - 25), min(w - 10, max(xs) + 25)
    y_min, y_max = max(70, min(ys) - 25), min(h - 100, max(ys) + 25)

    corner_len = 20
    box_color = (0, 240, 255)

    # Corner brackets (Cyberpunk reticle style)
    # Top-Left
    cv2.line(frame, (x_min, y_min), (x_min + corner_len, y_min), box_color, 2, cv2.LINE_AA)
    cv2.line(frame, (x_min, y_min), (x_min, y_min + corner_len), box_color, 2, cv2.LINE_AA)
    # Top-Right
    cv2.line(frame, (x_max, y_min), (x_max - corner_len, y_min), box_color, 2, cv2.LINE_AA)
    cv2.line(frame, (x_max, y_min), (x_max, y_min + corner_len), box_color, 2, cv2.LINE_AA)
    # Bottom-Left
    cv2.line(frame, (x_min, y_max), (x_min + corner_len, y_max), box_color, 2, cv2.LINE_AA)
    cv2.line(frame, (x_min, y_max), (x_min, y_max - corner_len), box_color, 2, cv2.LINE_AA)
    # Bottom-Right
    cv2.line(frame, (x_max, y_max), (x_max - corner_len, y_max), box_color, 2, cv2.LINE_AA)
    cv2.line(frame, (x_max, y_max), (x_max, y_max - corner_len), box_color, 2, cv2.LINE_AA)

    # Label Tag
    if label and label != "None":
        tag_w = len(label) * 11 + 18
        cv2.rectangle(frame, (x_min, y_min - 26), (x_min + tag_w, y_min), box_color, -1)
        cv2.putText(
            frame,
            label,
            (x_min + 8, y_min - 7),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (16, 20, 28),
            2,
            cv2.LINE_AA
        )
