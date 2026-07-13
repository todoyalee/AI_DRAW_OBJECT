import cv2
import mediapipe as mp
import easyocr
import numpy as np
import threading
import queue
import difflib
from time import time
from datetime import datetime
from collections import deque

# ============================================
# CONFIG
# ============================================

CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720

THICKNESS_MIN = 6
THICKNESS_MAX = 60
DEFAULT_THICKNESS = 25
DRAW_THICKNESS = float(DEFAULT_THICKNESS)   # float so the slider can change it smoothly

DRAW_GLOW = 18
GLOW_DOWNSCALE = 0.5

SMOOTHING = 0.25

# thumb-tip <-> index-tip distance (relative to hand size) below this = "pinching"
# used ONLY for UI clicks (slider / color palette), not for writing
PINCH_THRESHOLD_RATIO = 0.35

# thumb-tip <-> index-MCP(knuckle) distance (relative to hand size) above this = "thumb out"
# thumb out = pause writing. thumb tucked in = write. TUNE THIS if it feels too sensitive/insensitive.
THUMB_OUT_RATIO = 0.85

# frames a pinch must dwell on a color swatch/icon before it "clicks" (debounce so it
# doesn't fire by accident while passing through)
UI_CLICK_HOLD_FRAMES = 5

# frames a gesture must hold before it fires -- long enough to be a deliberate,
# unambiguous action (this is what tells the system "I'm done writing", instead of
# guessing from timing), short enough that it still feels responsive
CONFIRM_HOLD_FRAMES = 8   # closed fist = confirm/submit whatever you just wrote
UNDO_HOLD_FRAMES = 8      # open palm = undo/erase

# how many recent frames vote on each finger's up/down state before we trust it.
# raw per-frame landmark position is jittery enough that a single bad frame can
# drop a stroke mid-letter or misfire a gesture -- majority-voting over a short
# window smooths that out without adding noticeable input lag.
GESTURE_SMOOTH_FRAMES = 4

FUZZY_MATCH_CUTOFF = 0.55

SCAN_SLICE_COUNT = 3
ALERT_DURATION = 3.0    # seconds an alert banner stays up
POP_DURATION = 0.3      # seconds for a panel's spawn/refresh bounce

DEBUG_SHOW_OCR_CROP = False

# clinical HUD palette (BGR order, since OpenCV uses BGR not RGB)
NEON_COLORS = {
    "cyan":  (255, 220, 0),
    "amber": (0, 191, 255),
    "green": (20, 255, 57),
}
DEFAULT_COLOR_KEY = "cyan"

PANEL_BG = (35, 24, 15)        # dark navy-ish panel background
PANEL_BORDER = (255, 200, 80)  # cyan/amber border
ALERT_COLOR = (0, 60, 255)     # red-ish alert banner

# ============================================
# MEDIAPIPE
# ============================================

mp_hands = mp.solutions.hands

hands = mp_hands.Hands(
    max_num_hands=1,
    model_complexity=0,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

draw = mp.solutions.drawing_utils

# ============================================
# COMMAND VOCABULARY
# ============================================
# Every recognized word maps to an ACTION, not a picture -- this is the same
# "write a word -> something happens" mechanic as before, generalized into a
# small command registry. Adding a new command later is one dict entry below,
# not new gesture/rendering code.

COMMANDS = {
    "xray":        {"type": "scan", "modality": "XRAY"},
    "mri":         {"type": "scan", "modality": "MRI"},
    "checklist":   {"type": "checklist"},
    "allergy":     {"type": "patient", "field": "allergy"},
    "bloodtype":   {"type": "patient", "field": "bloodtype"},
    "incision":    {"type": "log", "label": "INCISION"},
    "close":       {"type": "log", "label": "CLOSE"},
    "blood":       {"type": "alert", "target": "Blood Bank"},
    "anesthesia":  {"type": "alert", "target": "Anesthesiologist"},
    "cam2":        {"type": "cam_toggle"},
    "morphine":    {"type": "drug", "name": "MORPHINE", "dose": "2-4 mg IV, q2-4h PRN"},
    "epinephrine": {"type": "drug", "name": "EPINEPHRINE", "dose": "0.1 mg IV (1:10,000) - cardiac arrest dose"},
}

# words that act on whatever's currently displayed, instead of replacing it
MODIFIER_WORDS = {"next", "zoom"}

ALL_VOCAB = list(COMMANDS.keys()) + list(MODIFIER_WORDS)

# Public-domain WHO Surgical Safety Checklist items (paraphrased), cycled with "NEXT"
CHECKLIST_STEPS = [
    "SIGN IN - Confirm patient identity, site, procedure, and consent",
    "SIGN IN - Site marked, or marking not applicable",
    "TIME OUT - Team introduces name and role",
    "TIME OUT - Surgeon reviews critical or unexpected steps",
    "TIME OUT - Antibiotic prophylaxis given in last 60 min",
    "SIGN OUT - Instrument, sponge, and needle counts correct",
    "SIGN OUT - Specimen labeled correctly",
]

# mock data for the demo only -- not real patient records
PATIENT_INFO = {
    "allergy": "Penicillin (rash), Latex (mild)",
    "bloodtype": "O Negative",
}

# ============================================
# OCR (runs in a background thread so it never blocks the camera loop)
# ============================================

reader = easyocr.Reader(['en'], gpu=False)

ocr_input_queue = queue.Queue()
ocr_result_queue = queue.Queue()


def fuzzy_match(text_nospace):
    """Snap a messy OCR guess to the closest known command word instead of
    requiring an exact match."""
    candidates_nospace = [c.replace(" ", "") for c in ALL_VOCAB]
    close = difflib.get_close_matches(
        text_nospace, candidates_nospace, n=1, cutoff=FUZZY_MATCH_CUTOFF
    )
    if not close:
        return ""
    return ALL_VOCAB[candidates_nospace.index(close[0])]


def recognize_word(image):
    result = reader.readtext(image, detail=1, paragraph=False)
    if len(result) == 0:
        return "", 0.0

    text = " ".join(r[1] for r in result).lower()
    text = text.replace("-", " ").replace("_", " ").strip()
    confidence = float(np.mean([r[2] for r in result]))

    if text in ALL_VOCAB:
        return text, confidence
    text_nospace = text.replace(" ", "")
    if text_nospace in ALL_VOCAB:
        return text_nospace, confidence
    return fuzzy_match(text_nospace), confidence


def ocr_worker():
    while True:
        img = ocr_input_queue.get()
        if img is None:
            break
        word, confidence = recognize_word(img)
        ocr_result_queue.put((word, confidence))


ocr_thread = threading.Thread(target=ocr_worker, daemon=True)
ocr_thread.start()

ocr_busy = False

# ============================================
# CAMERA
# ============================================

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)

cv2.namedWindow("Touchless OR Interface", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("Touchless OR Interface", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
try:
    # Pull the window to the front so it actually has keyboard focus.
    # (Not supported on every OpenCV build/backend - safe to ignore if it errors.)
    cv2.setWindowProperty("Touchless OR Interface", cv2.WND_PROP_TOPMOST, 1)
except cv2.error:
    pass

# ============================================
# SYNTHETIC SCAN IMAGERY
# ============================================
# Procedurally generated placeholder scans -- deliberately NOT real patient
# imagery, so there's no licensing or privacy risk in a demo. Swap in real
# licensed sample images later; nothing else in the system cares where the
# pixels come from.

SCAN_PANEL_SIZE = 420


def generate_scan_image(modality, slice_idx):
    img = np.full((SCAN_PANEL_SIZE, SCAN_PANEL_SIZE, 3), 18, dtype=np.uint8)
    rng = np.random.default_rng(seed=(hash(modality) & 0xFFFF) + slice_idx * 97)
    line_color = (235, 235, 235) if modality == "XRAY" else (255, 225, 190)
    for _ in range(6):
        cx = SCAN_PANEL_SIZE // 2 + int(rng.integers(-80, 80))
        cy = SCAN_PANEL_SIZE // 2 + int(rng.integers(-80, 80)) + slice_idx * 10
        axes = (int(rng.integers(25, 75)), int(rng.integers(40, 115)))
        angle = int(rng.integers(0, 180))
        cv2.ellipse(img, (cx, cy), axes, angle, 0, 360, line_color, thickness=int(rng.integers(2, 5)))
    cv2.putText(img, f"{modality} - SAMPLE (SYNTHETIC)", (14, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1, cv2.LINE_AA)
    cv2.putText(img, f"Slice {slice_idx + 1}/{SCAN_SLICE_COUNT}", (14, SCAN_PANEL_SIZE - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1, cv2.LINE_AA)
    return img


SCAN_IMAGES = {
    modality: [generate_scan_image(modality, i) for i in range(SCAN_SLICE_COUNT)]
    for modality in ("XRAY", "MRI")
}


def get_scan_display(view):
    img = SCAN_IMAGES[view["modality"]][view["slice"]]
    if view.get("zoom"):
        ih, iw = img.shape[:2]
        ch, cw = int(ih * 0.5), int(iw * 0.5)
        y0, x0 = (ih - ch) // 2, (iw - cw) // 2
        img = cv2.resize(img[y0:y0+ch, x0:x0+cw], (iw, ih), interpolation=cv2.INTER_LINEAR)
    return img

# ============================================
# STATE
# ============================================

canvas = None
points = deque(maxlen=5000)
drawing = False

active_view = None      # dict describing whatever info panel is on screen, or None
view_spawn_time = 0.0
cam2_on = False
event_log = []          # list of (time_str, label), most recent last

status_message = ""
status_message_until = 0.0

pen_color_key = DEFAULT_COLOR_KEY

object_x = 0
object_y = 0

smooth_x = None
smooth_y = None

frame_count = 0
frame_times = deque(maxlen=30)

confirm_counter = 0
undo_counter = 0

index_up_hist = deque(maxlen=GESTURE_SMOOTH_FRAMES)
thumb_out_hist = deque(maxlen=GESTURE_SMOOTH_FRAMES)
open_hand_hist = deque(maxlen=GESTURE_SMOOTH_FRAMES)
fist_hist = deque(maxlen=GESTURE_SMOOTH_FRAMES)

# UI layout (computed once from first frame size)
ui_layout_ready = False
slider_x1 = slider_x2 = slider_y = 0
color_icon_center = (0, 0)
color_icon_radius = 26
swatch_positions = {}
swatch_radius = 24
color_menu_open = False

ui_hover_target = None
ui_hover_counter = 0
ui_action_armed = True   # False right after a click fires, True again once pinch releases

# ============================================
# HELPERS
# ============================================

def ease_out_back(t):
    """Overshoot-then-settle easing (0->1), used for a panel's spawn bounce."""
    c1 = 1.70158
    c3 = c1 + 1
    t -= 1
    return 1 + c3 * (t ** 3) + c1 * (t ** 2)


def pop_scale(spawn_time):
    elapsed = time() - spawn_time
    if elapsed < POP_DURATION:
        return max(0.05, ease_out_back(elapsed / POP_DURATION))
    return 1.0


def paste_panel(frame, panel_img, anchor_x, anchor_y, spawn_time):
    scale = pop_scale(spawn_time)
    ph, pw = panel_img.shape[:2]
    new_w, new_h = max(1, int(pw * scale)), max(1, int(ph * scale))
    resized = cv2.resize(panel_img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    x = max(0, min(anchor_x, frame.shape[1] - new_w - 1))
    y = max(0, min(anchor_y, frame.shape[0] - new_h - 1))
    frame[y:y+new_h, x:x+new_w] = resized


def smooth_point(x, y):
    global smooth_x, smooth_y
    if smooth_x is None:
        smooth_x, smooth_y = x, y
    smooth_x = int(smooth_x + (x - smooth_x) * SMOOTHING)
    smooth_y = int(smooth_y + (y - smooth_y) * SMOOTHING)
    return smooth_x, smooth_y


def hand_size(hand, width, height):
    wrist = hand.landmark[0]
    middle_mcp = hand.landmark[9]
    wx, wy = wrist.x * width, wrist.y * height
    mx, my = middle_mcp.x * width, middle_mcp.y * height
    return np.hypot(mx - wx, my - wy)


def create_word_image(points, width, height):
    img = np.zeros((height, width), dtype=np.uint8)
    for i in range(1, len(points)):
        if points[i-1] is None or points[i] is None:
            continue
        cv2.line(img, points[i-1], points[i], 255, 35, cv2.LINE_AA)
    ys, xs = np.where(img > 0)
    if len(xs) == 0:
        return None
    pad = 40
    x1 = max(xs.min() - pad, 0)
    y1 = max(ys.min() - pad, 0)
    x2 = min(xs.max() + pad, width)
    y2 = min(ys.max() + pad, height)
    crop = img[y1:y2, x1:x2]
    # Smaller crop = EasyOCR processes far fewer pixels = noticeably faster inference.
    # 480x200 is still plenty of detail for short words; INTER_LINEAR is cheaper than CUBIC.
    crop = cv2.resize(crop, (480, 200), interpolation=cv2.INTER_LINEAR)
    kernel = np.ones((5, 5), np.uint8)
    crop = cv2.dilate(crop, kernel, 1)
    crop = cv2.erode(crop, kernel, 1)
    return crop


def setup_ui_layout(w, h):
    global slider_x1, slider_x2, slider_y, color_icon_center
    slider_y = 60
    SLIDER_LENGTH = 260  # medium-size line instead of stretching near full width
    slider_x1 = w // 2 - SLIDER_LENGTH // 2
    slider_x2 = w // 2 + SLIDER_LENGTH // 2
    color_icon_center = (slider_x2 + 55, slider_y)  # sits just beside the line's right end


def thickness_to_slider_x(thickness):
    t = (thickness - THICKNESS_MIN) / (THICKNESS_MAX - THICKNESS_MIN)
    return int(slider_x1 + t * (slider_x2 - slider_x1))


def slider_x_to_thickness(x):
    t = (x - slider_x1) / max(1, (slider_x2 - slider_x1))
    t = max(0.0, min(1.0, t))
    return THICKNESS_MIN + t * (THICKNESS_MAX - THICKNESS_MIN)


def draw_ui(frame):
    # pen-size slider bar
    cv2.line(frame, (slider_x1, slider_y), (slider_x2, slider_y), (90, 90, 90), 6)
    handle_x = thickness_to_slider_x(DRAW_THICKNESS)
    cv2.circle(frame, (handle_x, slider_y), 14, NEON_COLORS[pen_color_key], -1)
    cv2.circle(frame, (handle_x, slider_y), 16, (255, 255, 255), 2)

    # color icon (shows current color)
    cv2.circle(frame, color_icon_center, color_icon_radius, NEON_COLORS[pen_color_key], -1)
    cv2.circle(frame, color_icon_center, color_icon_radius, (255, 255, 255), 2)

    if color_menu_open:
        keys = list(NEON_COLORS.keys())
        for i, k in enumerate(keys):
            pos = (
                color_icon_center[0],
                color_icon_center[1] + color_icon_radius + 40 + i * (swatch_radius * 2 + 16),
            )
            swatch_positions[k] = pos
            cv2.circle(frame, pos, swatch_radius, NEON_COLORS[k], -1)
            cv2.circle(frame, pos, swatch_radius, (255, 255, 255), 2)


def dist(a, b):
    return np.hypot(a[0] - b[0], a[1] - b[1])


def format_time_now():
    return datetime.now().strftime("%H:%M:%S")


# ---------------- info panel renderers (each returns a fixed-size BGR image) ----------------

def render_scan_panel(view):
    img = get_scan_display(view)
    panel = np.full((SCAN_PANEL_SIZE + 50, SCAN_PANEL_SIZE + 32, 3), PANEL_BG, dtype=np.uint8)
    panel[36:36+SCAN_PANEL_SIZE, 16:16+SCAN_PANEL_SIZE] = img
    cv2.rectangle(panel, (0, 0), (panel.shape[1]-1, panel.shape[0]-1), PANEL_BORDER, 2)
    cv2.putText(panel, f"{view['modality']} VIEWER", (16, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
    return panel


def render_checklist_panel(view):
    panel = np.full((260, 620, 3), PANEL_BG, dtype=np.uint8)
    cv2.rectangle(panel, (0, 0), (panel.shape[1]-1, panel.shape[0]-1), PANEL_BORDER, 2)
    cv2.putText(panel, "SURGICAL SAFETY CHECKLIST", (20, 34),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

    step_text = CHECKLIST_STEPS[view["step"]]
    words = step_text.split(" ")
    lines, cur = [], ""
    for word in words:
        trial = (cur + " " + word).strip()
        if len(trial) > 40:
            lines.append(cur)
            cur = word
        else:
            cur = trial
    if cur:
        lines.append(cur)

    for i, line in enumerate(lines):
        cv2.putText(panel, line, (20, 76 + i * 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 220, 255), 2, cv2.LINE_AA)

    cv2.putText(panel, f"Step {view['step']+1}/{len(CHECKLIST_STEPS)}  (write NEXT to advance)",
                (20, panel.shape[0]-18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (170, 170, 170), 1, cv2.LINE_AA)
    return panel


def render_patient_panel(view):
    panel = np.full((150, 500, 3), PANEL_BG, dtype=np.uint8)
    cv2.rectangle(panel, (0, 0), (panel.shape[1]-1, panel.shape[0]-1), PANEL_BORDER, 2)
    label = "ALLERGIES" if view["field"] == "allergy" else "BLOOD TYPE"
    cv2.putText(panel, label, (20, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(panel, PATIENT_INFO[view["field"]], (20, 86),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 220, 255), 2, cv2.LINE_AA)
    cv2.putText(panel, "MOCK DATA - DEMO ONLY", (20, 128),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (140, 140, 140), 1, cv2.LINE_AA)
    return panel


def render_drug_panel(view):
    panel = np.full((150, 520, 3), PANEL_BG, dtype=np.uint8)
    cv2.rectangle(panel, (0, 0), (panel.shape[1]-1, panel.shape[0]-1), PANEL_BORDER, 2)
    cv2.putText(panel, view["name"], (20, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(panel, view["dose"], (20, 86), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 255), 2, cv2.LINE_AA)
    cv2.putText(panel, "MOCK DATA - VERIFY BEFORE USE", (20, 128),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (140, 140, 140), 1, cv2.LINE_AA)
    return panel


def render_alert_banner(frame, view):
    remaining = view["until"] - time()
    if remaining <= 0:
        return False
    pulse = 0.5 + 0.5 * np.sin(time() * 10)
    bw, bh = frame.shape[1] - 80, 70
    bx, by = 40, 100
    overlay = frame.copy()
    cv2.rectangle(overlay, (bx, by), (bx + bw, by + bh), ALERT_COLOR, -1)
    a = 0.35 + 0.25 * pulse
    cv2.addWeighted(overlay, a, frame, 1 - a, 0, frame)
    cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), (255, 255, 255), 2)
    cv2.putText(frame, f"ALERT SENT -> {view['target'].upper()}", (bx + 24, by + 45),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
    return True


def render_cam2_inset(frame):
    fh, fw = frame.shape[:2]
    iw, ih = 260, 150
    x, y = fw - iw - 30, fh - ih - 30
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 60, 140)
    edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    inset = cv2.resize(edges_bgr, (iw, ih), interpolation=cv2.INTER_LINEAR)
    frame[y:y+ih, x:x+iw] = inset
    cv2.rectangle(frame, (x, y), (x+iw, y+ih), PANEL_BORDER, 2)
    cv2.putText(frame, "CAM 2 (SIMULATED)", (x+8, y+20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)


def render_event_log(frame):
    if not event_log:
        return
    x = 25
    y = frame.shape[0] - 25 - 22 * len(event_log)
    cv2.putText(frame, "PROCEDURE LOG", (x, y - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1, cv2.LINE_AA)
    for i, (ts, label) in enumerate(event_log):
        cv2.putText(frame, f"{ts}  {label}", (x, y + i * 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 255), 1, cv2.LINE_AA)


def render_hud(frame):
    if len(frame_times) >= 2:
        fps = (len(frame_times) - 1) / (frame_times[-1] - frame_times[0])
    else:
        fps = 0.0
    cv2.putText(frame, f"{fps:4.1f} FPS", (frame.shape[1] - 150, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 120), 2, cv2.LINE_AA)

    if time() < status_message_until:
        cv2.putText(frame, status_message, (25, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 220, 255), 2, cv2.LINE_AA)

    if ocr_busy:
        pulse = int(180 + 75 * np.sin(time() * 6))
        cv2.putText(frame, "Reading...", (25, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, pulse, 255), 2)


# ---------------- command execution ----------------

def execute_command(key):
    global active_view, cam2_on, view_spawn_time
    cmd = COMMANDS[key]
    kind = cmd["type"]

    if kind == "scan":
        active_view = {"type": "scan", "modality": cmd["modality"], "slice": 0, "zoom": False}
    elif kind == "checklist":
        active_view = {"type": "checklist", "step": 0}
    elif kind == "patient":
        active_view = {"type": "patient", "field": cmd["field"]}
    elif kind == "drug":
        active_view = {"type": "drug", "name": cmd["name"], "dose": cmd["dose"]}
    elif kind == "log":
        event_log.append((format_time_now(), cmd["label"]))
        del event_log[:-5]
    elif kind == "alert":
        active_view = {"type": "alert", "target": cmd["target"], "until": time() + ALERT_DURATION}
    elif kind == "cam_toggle":
        cam2_on = not cam2_on

    view_spawn_time = time()


def apply_modifier(key):
    global view_spawn_time
    if active_view is None:
        return
    if key == "next":
        if active_view["type"] == "scan":
            active_view["slice"] = (active_view["slice"] + 1) % SCAN_SLICE_COUNT
            view_spawn_time = time()
        elif active_view["type"] == "checklist":
            active_view["step"] = (active_view["step"] + 1) % len(CHECKLIST_STEPS)
            view_spawn_time = time()
    elif key == "zoom":
        if active_view["type"] == "scan":
            active_view["zoom"] = not active_view.get("zoom", False)
            view_spawn_time = time()


# ============================================
# MAIN LOOP
# ============================================

while True:
    frame_count += 1
    success, frame = cap.read()
    if not success:
        break
    frame = cv2.flip(frame, 1)
    frame_times.append(time())

    h, w = frame.shape[:2]
    if not ui_layout_ready:
        setup_ui_layout(w, h)
        ui_layout_ready = True

    if canvas is None:
        canvas = np.zeros_like(frame)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    trigger_ocr = False
    trigger_undo = False

    if results.multi_hand_landmarks:
        hand = results.multi_hand_landmarks[0]
        draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)
        landmarks = hand.landmark

        index_tip = landmarks[8]
        index_pip = landmarks[6]
        index_mcp = landmarks[5]
        middle_tip = landmarks[12]
        middle_pip = landmarks[10]
        ring_tip = landmarks[16]
        ring_pip = landmarks[14]
        pinky_tip = landmarks[20]
        pinky_pip = landmarks[18]
        thumb_tip = landmarks[4]

        x = int(index_tip.x * w)
        y = int(index_tip.y * h)
        x, y = smooth_point(x, y)

        scale = hand_size(hand, w, h)

        tx, ty = int(thumb_tip.x * w), int(thumb_tip.y * h)

        # pinch = thumb tip touching index TIP -> used only for UI clicks
        pinch_dist = np.hypot(tx - x, ty - y)
        is_pinching = scale > 0 and (pinch_dist / scale) < PINCH_THRESHOLD_RATIO
        pinch_point = ((tx + x) // 2, (ty + y) // 2)

        # thumb "out" = thumb tip far from index KNUCKLE -> pause writing
        mcp_x, mcp_y = int(index_mcp.x * w), int(index_mcp.y * h)
        thumb_mcp_dist = np.hypot(tx - mcp_x, ty - mcp_y)
        raw_thumb_out = scale > 0 and (thumb_mcp_dist / scale) > THUMB_OUT_RATIO

        raw_index_up = index_tip.y < index_pip.y
        middle_up = middle_tip.y < middle_pip.y
        ring_up = ring_tip.y < ring_pip.y
        pinky_up = pinky_tip.y < pinky_pip.y
        raw_open_hand = raw_index_up and middle_up and ring_up and pinky_up
        raw_fist = (not raw_index_up) and (not middle_up) and (not ring_up) and (not pinky_up) and (not raw_thumb_out)

        # majority-vote each gesture over the last few frames so a single jittery
        # landmark reading doesn't drop a stroke or misfire a gesture
        index_up_hist.append(raw_index_up)
        thumb_out_hist.append(raw_thumb_out)
        open_hand_hist.append(raw_open_hand)
        fist_hist.append(raw_fist)

        index_up = sum(index_up_hist) > len(index_up_hist) / 2
        thumb_out = sum(thumb_out_hist) > len(thumb_out_hist) / 2
        open_hand = sum(open_hand_hist) > len(open_hand_hist) / 2
        fist = sum(fist_hist) > len(fist_hist) / 2

        # ---------------- UI interaction: slider + color palette ----------------
        handled_by_ui = False

        if is_pinching:
            px, py = pinch_point
            near_slider = abs(py - slider_y) < 45 and (slider_x1 - 30) < px < (slider_x2 + 30)
            near_icon = dist((px, py), color_icon_center) < color_icon_radius + 15

            if near_slider and not color_menu_open:
                target_thickness = slider_x_to_thickness(px)
                DRAW_THICKNESS = DRAW_THICKNESS + (target_thickness - DRAW_THICKNESS) * 0.3
                handled_by_ui = True
                ui_hover_target = None

            elif near_icon and not color_menu_open:
                if ui_hover_target != "icon":
                    ui_hover_target = "icon"
                    ui_hover_counter = 0
                ui_hover_counter += 1
                if ui_hover_counter >= UI_CLICK_HOLD_FRAMES and ui_action_armed:
                    color_menu_open = True
                    ui_action_armed = False
                handled_by_ui = True

            elif color_menu_open:
                hit_key = None
                for k, pos in swatch_positions.items():
                    if dist((px, py), pos) < swatch_radius + 15:
                        hit_key = k
                        break
                if hit_key:
                    if ui_hover_target != hit_key:
                        ui_hover_target = hit_key
                        ui_hover_counter = 0
                    ui_hover_counter += 1
                    if ui_hover_counter >= UI_CLICK_HOLD_FRAMES and ui_action_armed:
                        pen_color_key = hit_key
                        color_menu_open = False
                        ui_action_armed = False
                handled_by_ui = True
        else:
            ui_hover_target = None
            ui_hover_counter = 0
            ui_action_armed = True

        # ---------------- CONFIRM: closed fist held -> submit what you just wrote ----------------
        if fist and not handled_by_ui:
            confirm_counter += 1
        else:
            confirm_counter = 0

        trigger_ocr = (
            confirm_counter == CONFIRM_HOLD_FRAMES
            and len(points) > 10
            and not ocr_busy
        )

        # ---------------- UNDO: open palm held -> clear stroke, or dismiss current view ----------------
        if open_hand and not handled_by_ui:
            undo_counter += 1
        else:
            undo_counter = 0

        trigger_undo = undo_counter == UNDO_HOLD_FRAMES

        # ---------------- WRITE ----------------
        if not handled_by_ui:
            if index_up and not thumb_out and not fist:
                drawing = True
                cv2.circle(frame, (x, y), 14, (255, 255, 255), -1)
                cv2.circle(frame, (x, y), 24, NEON_COLORS[pen_color_key], 3)

                if len(points) == 0 or points[-1] is None:
                    points.append((x, y))
                else:
                    lx, ly = points[-1]
                    distance = np.hypot(x - lx, y - ly)
                    if distance > 4:
                        cv2.line(canvas, (lx, ly), (x, y), NEON_COLORS[pen_color_key],
                                  int(DRAW_THICKNESS), cv2.LINE_AA)
                        points.append((x, y))

            elif index_up and thumb_out:
                # PAUSED: thumb out -> pen lifted, reposition freely without drawing a line
                if drawing:
                    points.append(None)   # breaks the stroke so it won't connect
                drawing = False
                cv2.circle(frame, (x, y), 14, (0, 0, 255), 2)   # red ring = pen is up

            else:
                if drawing:
                    points.append(None)
                drawing = False

    else:
        smooth_x = None
        smooth_y = None
        if drawing:
            points.append(None)
        drawing = False
        confirm_counter = 0
        undo_counter = 0
        ui_hover_target = None
        ui_hover_counter = 0
        index_up_hist.clear()
        thumb_out_hist.clear()
        open_hand_hist.clear()
        fist_hist.clear()

    # ============================================
    # NEON RENDER (writing canvas glow)
    # ============================================

    small = cv2.resize(canvas, None, fx=GLOW_DOWNSCALE, fy=GLOW_DOWNSCALE,
                        interpolation=cv2.INTER_LINEAR)
    small_glow = cv2.GaussianBlur(small, (0, 0), DRAW_GLOW * GLOW_DOWNSCALE)
    glow = cv2.resize(small_glow, (canvas.shape[1], canvas.shape[0]),
                       interpolation=cv2.INTER_LINEAR)

    frame = cv2.addWeighted(frame, 1.0, glow, 0.95, 0)

    mask = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY) > 0
    frame[mask] = (255, 255, 255)   # bright white core, colored neon glow around it

    # ============================================
    # UNDO / ERASE
    # ============================================

    if trigger_undo:
        if len(points) > 0:
            canvas[:] = 0
            points.clear()
            status_message = "Cleared"
        elif active_view is not None:
            active_view = None
            status_message = "Dismissed"
        else:
            status_message = "Nothing to undo"
        status_message_until = time() + 1.5

    # ============================================
    # CONFIRM -> OCR -> COMMAND EXECUTION (non-blocking)
    # ============================================

    if trigger_ocr:
        img = create_word_image(points, frame.shape[1], frame.shape[0])
        if img is not None:
            if DEBUG_SHOW_OCR_CROP:
                cv2.imshow("OCR", img)
            ocr_input_queue.put(img)
            ocr_busy = True
        points.clear()
        canvas[:] = 0

    try:
        key_word, confidence = ocr_result_queue.get_nowait()
        ocr_busy = False
        if key_word in MODIFIER_WORDS:
            apply_modifier(key_word)
            status_message = f"{key_word.upper()} ({confidence*100:.0f}%)"
            print("Modifier:", key_word)
        elif key_word in COMMANDS:
            execute_command(key_word)
            status_message = f"{key_word.upper()} ({confidence*100:.0f}%)"
            print("Command:", key_word)
        else:
            status_message = f"Not recognized ({confidence*100:.0f}%)"
            print("Word not recognized")
        status_message_until = time() + 2.5
    except queue.Empty:
        pass

    # ============================================
    # ACTIVE VIEW (scan / checklist / patient / drug / alert)
    # ============================================

    if active_view is not None:
        vtype = active_view["type"]
        if vtype == "alert":
            still_active = render_alert_banner(frame, active_view)
            if not still_active:
                active_view = None
        else:
            panel = None
            if vtype == "scan":
                panel = render_scan_panel(active_view)
            elif vtype == "checklist":
                panel = render_checklist_panel(active_view)
            elif vtype == "patient":
                panel = render_patient_panel(active_view)
            elif vtype == "drug":
                panel = render_drug_panel(active_view)
            if panel is not None:
                paste_panel(frame, panel, w - panel.shape[1] - 40, 100, view_spawn_time)

    if cam2_on:
        render_cam2_inset(frame)

    render_event_log(frame)

    # ============================================
    # UI OVERLAY
    # ============================================

    draw_ui(frame)

    # ============================================
    # HUD
    # ============================================

    render_hud(frame)

    # ============================================
    # SHOW FRAME
    # ============================================

    cv2.imshow("Touchless OR Interface", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == 27 or key == ord("q") or key == ord("Q"):
        break
    elif key == ord("c") or key == ord("C"):
        canvas[:] = 0
        points.clear()
    elif key == ord("r") or key == ord("R"):
        canvas[:] = 0
        points.clear()
        active_view = None
        cam2_on = False
        event_log = []
        status_message = ""

# ============================================
# CLEANUP
# ============================================

ocr_input_queue.put(None)
cap.release()
cv2.destroyAllWindows()
