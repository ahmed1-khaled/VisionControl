import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import time
import math

# ============================================================
# EYE + HAND COMPUTER CONTROLLER V2
# ============================================================

# -------------------- CAMERA --------------------
CAMERA_ID = 0

# -------------------- SCREEN ---------------------
SCREEN_W, SCREEN_H = pyautogui.size()

# -------------------- EYE -----------------------
EYE_DEADZONE = 0.035
EYE_SENSITIVITY_X = 2.8
EYE_SENSITIVITY_Y = 2.4

# Smoothing: lower = smoother, higher = faster
EYE_SMOOTHING = 0.18
HAND_SMOOTHING = 0.30

# Calibration duration
CALIBRATION_SECONDS = 2.0

# -------------------- HAND ----------------------
PINCH_THRESHOLD = 0.045

CLICK_COOLDOWN = 0.45
DOUBLE_CLICK_COOLDOWN = 0.8

# -------------------- SCROLL --------------------
SCROLL_THRESHOLD = 0.025
SCROLL_SPEED = 4

# -------------------- SAFETY --------------------
pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0.01


# ============================================================
# MEDIAPIPE
# ============================================================

mp_face = mp.solutions.face_mesh
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

face_mesh = mp_face.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.55,
    min_tracking_confidence=0.55
)

hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.55,
    min_tracking_confidence=0.55
)


# ============================================================
# CAMERA
# ============================================================

cap = cv2.VideoCapture(CAMERA_ID)

if not cap.isOpened():
    raise RuntimeError("Could not open camera.")


# ============================================================
# STATE
# ============================================================

control_enabled = True

# Modes:
# "EYE"
# "HAND"
# "PAUSED"

mode = "EYE"

mouse_x, mouse_y = pyautogui.position()

last_click = 0
last_double_click = 0

dragging = False

# Eye calibration
calibrated = False
calibration_samples = []
calibration_start = time.time()

eye_center_x = 0.5
eye_center_y = 0.5

# Eye smoothing
smooth_eye_x = 0.0
smooth_eye_y = 0.0

# Hand smoothing
smooth_hand_x = SCREEN_W / 2
smooth_hand_y = SCREEN_H / 2

# Gesture timing
gesture_start = {}

# FPS
last_time = time.time()
fps = 0


# ============================================================
# HELPERS
# ============================================================

def clamp(value, minimum, maximum):
    return max(minimum, min(maximum, value))


def distance(p1, p2):
    return math.sqrt(
        (p1.x - p2.x) ** 2 +
        (p1.y - p2.y) ** 2
    )


def lerp(current, target, smoothing):
    return current + (target - current) * smoothing


def move_mouse(x, y, smoothing=0.25):
    global mouse_x, mouse_y

    mouse_x = lerp(mouse_x, x, smoothing)
    mouse_y = lerp(mouse_y, y, smoothing)

    mouse_x = clamp(mouse_x, 0, SCREEN_W - 1)
    mouse_y = clamp(mouse_y, 0, SCREEN_H - 1)

    pyautogui.moveTo(
        int(mouse_x),
        int(mouse_y),
        duration=0
    )


# ============================================================
# EYE TRACKING
# ============================================================

def get_eye_gaze(face_landmarks):

    lm = face_landmarks.landmark

    # --------------------------------------------------------
    # LEFT EYE
    # --------------------------------------------------------

    left_iris = np.mean(
        [[lm[i].x, lm[i].y] for i in [468, 469, 470, 471]],
        axis=0
    )

    left_inner = np.array([
        lm[133].x,
        lm[133].y
    ])

    left_outer = np.array([
        lm[33].x,
        lm[33].y
    ])

    left_top = np.array([
        lm[159].x,
        lm[159].y
    ])

    left_bottom = np.array([
        lm[145].x,
        lm[145].y
    ])

    # --------------------------------------------------------
    # RIGHT EYE
    # --------------------------------------------------------

    right_iris = np.mean(
        [[lm[i].x, lm[i].y] for i in [473, 474, 475, 476]],
        axis=0
    )

    right_inner = np.array([
        lm[362].x,
        lm[362].y
    ])

    right_outer = np.array([
        lm[263].x,
        lm[263].y
    ])

    right_top = np.array([
        lm[386].x,
        lm[386].y
    ])

    right_bottom = np.array([
        lm[374].x,
        lm[374].y
    ])

    # --------------------------------------------------------
    # Signed horizontal position
    # --------------------------------------------------------

    left_vec = left_inner - left_outer
    right_vec = right_inner - right_outer

    left_len = np.linalg.norm(left_vec)
    right_len = np.linalg.norm(right_vec)

    if left_len < 0.001 or right_len < 0.001:
        return None

    left_pos = np.dot(
        left_iris - left_outer,
        left_vec
    ) / (left_len ** 2)

    right_pos = np.dot(
        right_iris - right_outer,
        right_vec
    ) / (right_len ** 2)

    # --------------------------------------------------------
    # Vertical position
    # --------------------------------------------------------

    left_height = np.linalg.norm(
        left_bottom - left_top
    )

    right_height = np.linalg.norm(
        right_bottom - right_top
    )

    if left_height < 0.001 or right_height < 0.001:
        return None

    left_y = np.dot(
        left_iris - left_top,
        left_bottom - left_top
    ) / (left_height ** 2)

    right_y = np.dot(
        right_iris - right_top,
        right_bottom - right_top
    ) / (right_height ** 2)

    gaze_x = (left_pos + right_pos) / 2
    gaze_y = (left_y + right_y) / 2

    return gaze_x, gaze_y


# ============================================================
# CALIBRATION
# ============================================================

def calibrate_eye(gaze):

    global calibrated
    global eye_center_x
    global eye_center_y

    if gaze is None:
        return

    x, y = gaze

    if time.time() - calibration_start < CALIBRATION_SECONDS:

        calibration_samples.append(
            (x, y)
        )

    else:

        if len(calibration_samples) > 10:

            xs = [
                sample[0]
                for sample in calibration_samples
            ]

            ys = [
                sample[1]
                for sample in calibration_samples
            ]

            eye_center_x = np.median(xs)
            eye_center_y = np.median(ys)

        calibrated = True


# ============================================================
# EYE CONTROL
# ============================================================

def eye_control(gaze):

    global smooth_eye_x
    global smooth_eye_y

    if gaze is None:
        return "NO EYES"

    if not calibrated:
        calibrate_eye(gaze)
        return "CALIBRATING"

    gaze_x, gaze_y = gaze

    dx = gaze_x - eye_center_x
    dy = gaze_y - eye_center_y

    # --------------------------------------------------------
    # Deadzone
    # --------------------------------------------------------

    if abs(dx) < EYE_DEADZONE:
        dx = 0

    if abs(dy) < EYE_DEADZONE:
        dy = 0

    # --------------------------------------------------------
    # Smooth gaze
    # --------------------------------------------------------

    smooth_eye_x = lerp(
        smooth_eye_x,
        dx,
        EYE_SMOOTHING
    )

    smooth_eye_y = lerp(
        smooth_eye_y,
        dy,
        EYE_SMOOTHING
    )

    # --------------------------------------------------------
    # Move cursor
    # --------------------------------------------------------

    target_x = (
        pyautogui.position()[0]
        + smooth_eye_x
        * EYE_SENSITIVITY_X
        * 15
    )

    target_y = (
        pyautogui.position()[1]
        + smooth_eye_y
        * EYE_SENSITIVITY_Y
        * 15
    )

    target_x = clamp(
        target_x,
        0,
        SCREEN_W - 1
    )

    target_y = clamp(
        target_y,
        0,
        SCREEN_H - 1
    )

    move_mouse(
        target_x,
        target_y,
        EYE_SMOOTHING
    )

    # --------------------------------------------------------
    # Status
    # --------------------------------------------------------

    if abs(dx) > EYE_DEADZONE:

        if dx > 0:
            return "LOOK RIGHT"
        else:
            return "LOOK LEFT"

    if abs(dy) > EYE_DEADZONE:

        if dy > 0:
            return "LOOK DOWN"
        else:
            return "LOOK UP"

    return "EYE CENTER"


# ============================================================
# HAND FINGERS
# ============================================================

def get_finger_states(hand):

    lm = hand.landmark

    index = lm[8].y < lm[6].y
    middle = lm[12].y < lm[10].y
    ring = lm[16].y < lm[14].y
    pinky = lm[20].y < lm[18].y

    return index, middle, ring, pinky


# ============================================================
# HAND CONTROL
# ============================================================

def hand_control(hand):

    global last_click
    global last_double_click
    global dragging
    global smooth_hand_x
    global smooth_hand_y

    lm = hand.landmark

    thumb = lm[4]
    index = lm[8]
    middle = lm[12]

    index_up, middle_up, ring_up, pinky_up = \
        get_finger_states(hand)

    # ========================================================
    # FIST
    # ========================================================

    fist = (
        not index_up
        and not middle_up
        and not ring_up
        and not pinky_up
    )

    if fist:

        if dragging:

            pyautogui.mouseUp()
            dragging = False

        return "FIST / PAUSED"

    # ========================================================
    # OPEN HAND = RESUME
    # ========================================================

    open_hand = (
        index_up
        and middle_up
        and ring_up
        and pinky_up
    )

    if open_hand:

        if dragging:

            pyautogui.mouseUp()
            dragging = False

        return "OPEN HAND"

    # ========================================================
    # PINCH
    # ========================================================

    pinch = distance(
        index,
        thumb
    )

    if pinch < PINCH_THRESHOLD:

        now = time.time()

        if now - last_click > CLICK_COOLDOWN:

            pyautogui.click()

            last_click = now

            return "LEFT CLICK"

        return "PINCH"

    # ========================================================
    # INDEX + MIDDLE = DOUBLE CLICK
    # ========================================================

    if (
        index_up
        and middle_up
        and not ring_up
        and not pinky_up
    ):

        now = time.time()

        if now - last_double_click > DOUBLE_CLICK_COOLDOWN:

            pyautogui.doubleClick()

            last_double_click = now

            return "DOUBLE CLICK"

        return "TWO FINGERS"

    # ========================================================
    # INDEX ONLY = MOUSE
    # ========================================================

    if (
        index_up
        and not middle_up
        and not ring_up
        and not pinky_up
    ):

        target_x = index.x * SCREEN_W
        target_y = index.y * SCREEN_H

        smooth_hand_x = lerp(
            smooth_hand_x,
            target_x,
            HAND_SMOOTHING
        )

        smooth_hand_y = lerp(
            smooth_hand_y,
            target_y,
            HAND_SMOOTHING
        )

        move_mouse(
            smooth_hand_x,
            smooth_hand_y,
            HAND_SMOOTHING
        )

        return "HAND MOUSE"

    # ========================================================
    # THUMB + MIDDLE = RIGHT CLICK
    # ========================================================

    thumb_middle_distance = distance(
        thumb,
        middle
    )

    if (
        thumb_middle_distance < PINCH_THRESHOLD
        and not index_up
    ):

        now = time.time()

        if now - last_click > CLICK_COOLDOWN:

            pyautogui.rightClick()

            last_click = now

            return "RIGHT CLICK"

    # ========================================================
    # V SIGN + DOWN = SCROLL
    # ========================================================

    if index_up and middle_up:

        vertical_difference = middle.y - index.y

        if abs(vertical_difference) > SCROLL_THRESHOLD:

            if vertical_difference < 0:

                pyautogui.scroll(
                    SCROLL_SPEED
                )

                return "SCROLL UP"

            else:

                pyautogui.scroll(
                    -SCROLL_SPEED
                )

                return "SCROLL DOWN"

    return "HAND"


# ============================================================
# DRAW UI
# ============================================================

def draw_ui(frame, status):

    h, w = frame.shape[:2]

    # Main panel
    cv2.rectangle(
        frame,
        (10, 10),
        (500, 155),
        (15, 15, 15),
        -1
    )

    cv2.putText(
        frame,
        "EYE + HAND CONTROLLER V2",
        (25, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"MODE: {mode}",
        (25, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"STATUS: {status}",
        (25, 100),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"FPS: {fps}",
        (25, 128),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (255, 255, 255),
        1
    )

    cv2.putText(
        frame,
        "SPACE: ON/OFF | M: MODE | Q: EXIT",
        (25, 148),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.43,
        (180, 180, 180),
        1
    )

    # Calibration
    if not calibrated:

        remaining = max(
            0,
            CALIBRATION_SECONDS
            - (time.time() - calibration_start)
        )

        cv2.rectangle(
            frame,
            (10, 170),
            (400, 215),
            (20, 20, 20),
            -1
        )

        cv2.putText(
            frame,
            f"CALIBRATING... {remaining:.1f}s",
            (25, 200),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )


# ============================================================
# MAIN LOOP
# ============================================================

print("=" * 60)
print("           EYE + HAND COMPUTER CONTROLLER V2")
print("=" * 60)
print()
print("EYE MODE")
print("  Look around -> Move mouse")
print()
print("HAND MODE")
print("  Index finger              -> Move mouse")
print("  Thumb + Index             -> Left Click")
print("  Thumb + Middle             -> Right Click")
print("  Index + Middle             -> Double Click")
print("  Two fingers vertical       -> Scroll")
print("  Fist                       -> Pause")
print("  Open hand                  -> Resume")
print()
print("KEYBOARD")
print("  SPACE -> Enable / Disable")
print("  M     -> Eye / Hand Mode")
print("  Q     -> Quit")
print()
print("=" * 60)


try:

    while True:

        ret, frame = cap.read()

        if not ret:
            continue

        # Mirror camera
        frame = cv2.flip(
            frame,
            1
        )

        rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        # ----------------------------------------------------
        # PROCESS
        # ----------------------------------------------------

        face_result = face_mesh.process(
            rgb
        )

        hand_result = hands.process(
            rgb
        )

        status = "READY"

        # ----------------------------------------------------
        # FACE
        # ----------------------------------------------------

        if face_result.multi_face_landmarks:

            face = face_result.multi_face_landmarks[0]

            gaze = get_eye_gaze(face)

            if (
                control_enabled
                and mode == "EYE"
            ):

                status = eye_control(
                    gaze
                )

        else:

            if mode == "EYE":
                status = "FACE NOT DETECTED"

        # ----------------------------------------------------
        # HAND
        # ----------------------------------------------------

        if hand_result.multi_hand_landmarks:

            hand = hand_result.multi_hand_landmarks[0]

            mp_draw.draw_landmarks(
                frame,
                hand,
                mp_hands.HAND_CONNECTIONS
            )

            if mode == "HAND":

                hand_status = hand_control(
                    hand
                )

                status = hand_status

                # Fist
                if hand_status == "FIST / PAUSED":

                    control_enabled = False

                # Open hand resumes
                elif hand_status == "OPEN HAND":

                    control_enabled = True

        # ----------------------------------------------------
        # FPS
        # ----------------------------------------------------

        current_time = time.time()

        delta = current_time - last_time

        if delta > 0:

            fps = int(
                1 / delta
            )

        last_time = current_time

        # ----------------------------------------------------
        # UI
        # ----------------------------------------------------

        draw_ui(
            frame,
            status
        )

        cv2.imshow(
            "Eye + Hand Controller V2",
            frame
        )

        # ----------------------------------------------------
        # KEYBOARD
        # ----------------------------------------------------

        key = cv2.waitKey(1) & 0xFF

        # Quit
        if key == ord("q"):

            break

        # Enable / Disable
        elif key == 32:

            control_enabled = not control_enabled

            print(
                "Controller:",
                "ON"
                if control_enabled
                else "OFF"
            )

        # Mode switch
        elif key == ord("m"):

            if mode == "EYE":

                mode = "HAND"

            else:

                mode = "EYE"

            print(
                "Mode:",
                mode
            )


except KeyboardInterrupt:

    print("\nStopped by user.")


except Exception as e:

    print("\nERROR:")
    print(e)


finally:

    # Stop dragging safely
    if dragging:

        try:
            pyautogui.mouseUp()
        except:
            pass

    cap.release()

    cv2.destroyAllWindows()

    face_mesh.close()
    hands.close()

    print("Controller stopped safely.")