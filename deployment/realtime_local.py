"""
Real-time gesture recognition — local Python script.

Usage:
    python realtime_local.py

Requirements:
    pip install torch torchvision mediapipe opencv-python numpy tabulate
    (See requirements.txt for exact versions)

Directory layout expected:
    S-STRHanGe/deployment/models/   ← clone https://github.com/nielsschluesener/S-STRHanGe
    support_set/                     ← created automatically on first run
    realtime_local.py
"""

import os
import time
import pickle
import collections

import cv2
import numpy as np
import torch
import torch.nn as nn
import mediapipe as mp
from torch.autograd import Variable


# ── Configuration ──────────────────────────────────────────────────────────────
N_WAY = 5
K_SHOT = 1

GESTURES = [
    "Gesture_1",   # replace with your actual gesture names
    "Gesture_2",
    "Gesture_3",
    "Gesture_4",
    "Gesture_5",
]

MODEL_PATH    = "deployment\models"  # path to cloned repo models folder
SUPPORT_DIR   = "support_set"                   # where recorded support keypoints are saved

NUM_FRAMES    = 72      # fixed model input length (3 sec × 24 fps)
TARGET_FPS    = 24      # capture rate for support set recording
STRIDE        = 12      # run inference every N frames (12 = every 0.5 sec)
THRESHOLD     = 0.5     # minimum relation score to display a label
# ───────────────────────────────────────────────────────────────────────────────

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Model definitions ──────────────────────────────────────────────────────────
class LSTMEncoder(nn.Module):
    def __init__(self, input_size, hidden_lstm_size, num_lstm_layer):
        super().__init__()
        self.hidden_lstm_size = hidden_lstm_size
        self.num_lstm_layer   = num_lstm_layer
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_lstm_size,
            num_layers=num_lstm_layer,
            batch_first=True,
        )

    def forward(self, x):
        h0 = torch.zeros(self.num_lstm_layer, x.size(0), self.hidden_lstm_size).to(device)
        c0 = torch.zeros(self.num_lstm_layer, x.size(0), self.hidden_lstm_size).to(device)
        out, _ = self.lstm(x, (h0, c0))
        return out


class RelationNetwork(nn.Module):
    def __init__(self, input_size, hidden_lstm_size, fc_sizes, num_lstm_layer):
        super().__init__()
        self.hidden_lstm_size = hidden_lstm_size
        self.num_lstm_layer   = num_lstm_layer
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_lstm_size,
            num_layers=num_lstm_layer,
            batch_first=True,
        )
        self.layers = nn.ModuleList()
        in_size = hidden_lstm_size
        for size in fc_sizes:
            self.layers.append(nn.Linear(in_size, size))
            self.layers.append(nn.ReLU())
            in_size = size
        self.end_layer = nn.Sequential(nn.Linear(in_size, 1), nn.Sigmoid())

    def forward(self, x):
        h0 = torch.zeros(self.num_lstm_layer, x.size(0), self.hidden_lstm_size).to(device)
        c0 = torch.zeros(self.num_lstm_layer, x.size(0), self.hidden_lstm_size).to(device)
        out, _ = self.lstm(x, (h0, c0))
        out = out[:, -1, :]
        for layer in self.layers:
            out = layer(out)
        return self.end_layer(out)
# ───────────────────────────────────────────────────────────────────────────────


# ── Helper functions ───────────────────────────────────────────────────────────
def extract_keypoints(results):
    """Returns a (63,) array of hand landmarks, or zeros if no hand detected."""
    if results.multi_hand_landmarks:
        return np.array(
            [[lm.x, lm.y, lm.z] for lm in results.multi_hand_landmarks[0].landmark]
        ).flatten()
    return np.zeros(63)


def move_values(data, seq_length, direction="back"):
    """Shift non-zero frames to the back (or front) of the sequence."""
    max_values = data.max(axis=1)
    if np.array_equal(max_values, np.zeros(seq_length)):
        return np.zeros((seq_length, 63))
    i_first = next(i for i, v in enumerate(max_values) if v != 0)
    i_last  = next(i for i, v in reversed(list(enumerate(max_values))) if v != 0)
    non_zeros = data[i_first : i_last + 1]
    zeros     = np.zeros((seq_length - non_zeros.shape[0], 63))
    if direction == "back":
        return np.concatenate((zeros, non_zeros), axis=0)
    return np.concatenate((non_zeros, zeros), axis=0)


def calc_prediction(feature_encoder, relation_network, support_x, query_x,
                    num_classes, support_num_per_class, seq_length, num_units_lstm_encoder):
    support_features = feature_encoder(Variable(support_x).to(device))
    query_features   = feature_encoder(Variable(query_x).to(device))

    support_features = support_features.view(
        num_classes, support_num_per_class, seq_length, num_units_lstm_encoder
    )
    support_features     = torch.sum(support_features, 1).squeeze(1)
    support_features_ext = support_features.unsqueeze(0).repeat(1, 1, 1, 1).to(device)
    query_features_ext   = query_features.unsqueeze(0).repeat(num_classes, 1, 1, 1)
    query_features_ext   = torch.transpose(query_features_ext, 0, 1).to(device)

    relation_pairs = torch.cat((support_features_ext, query_features_ext), 3).view(
        -1, seq_length, num_units_lstm_encoder * 2
    ).to(device)
    return relation_network(relation_pairs).view(-1, num_classes).to(device)
# ───────────────────────────────────────────────────────────────────────────────


# ── Model loading ──────────────────────────────────────────────────────────────
def load_model(n_way, k_shot):
    model_name = f"{n_way}way-{k_shot}shot"
    param_path = os.path.join(MODEL_PATH, f"{model_name}_deployment_param.pkl")

    if not os.path.isfile(param_path):
        raise FileNotFoundError(
            f"Model file not found: {param_path}\n"
            "Clone the repo first:  git clone https://github.com/nielsschluesener/S-STRHanGe.git"
        )

    with open(param_path, "rb") as f:
        params = pickle.load(f)

    feature_encoder = LSTMEncoder(
        params["feature_length"],
        params["num_units_lstm_encoder"],
        params["num_lstm_layer_encoder"],
    ).to(device)

    relation_network = RelationNetwork(
        params["num_units_lstm_encoder"] * 2,
        params["num_units_lstm_relationnet"],
        params["num_units_fc_relu"],
        params["num_lstm_layer_relationnet"],
    ).to(device)

    feature_encoder.load_state_dict(
        torch.load(os.path.join(MODEL_PATH, f"{model_name}_feature_encoder.pkl"),
                   map_location=device)
    )
    relation_network.load_state_dict(
        torch.load(os.path.join(MODEL_PATH, f"{model_name}_relation_network.pkl"),
                   map_location=device)
    )

    feature_encoder.eval()
    relation_network.eval()
    return feature_encoder, relation_network, params
# ───────────────────────────────────────────────────────────────────────────────


# ── Support set recording ──────────────────────────────────────────────────────
def _record_one_sample(class_name, hands):
    """
    Opens webcam, shows a 3-second countdown, then captures NUM_FRAMES frames
    at TARGET_FPS. Returns a (NUM_FRAMES, 63) keypoints array.
    """
    cap = cv2.VideoCapture(0)
    keypoints_list = []

    # ── Countdown (3 seconds) ──
    t_start = time.time()
    while time.time() - t_start < 3.0:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        remaining = 3 - int(time.time() - t_start)
        cv2.putText(frame, f"Ready in: {remaining}",
                    (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 220, 0), 3)
        cv2.putText(frame, f"Gesture: {class_name}",
                    (30, 140), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        cv2.imshow("Support Set Recording", frame)
        cv2.waitKey(1)

    # ── Recording (collect NUM_FRAMES keypoints at TARGET_FPS) ──
    frame_interval = 1.0 / TARGET_FPS
    next_capture   = time.time()

    while len(keypoints_list) < NUM_FRAMES:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        cv2.putText(frame, "RECORDING",
                    (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 255), 3)
        cv2.putText(frame, f"{len(keypoints_list)} / {NUM_FRAMES} frames",
                    (30, 140), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        cv2.imshow("Support Set Recording", frame)
        cv2.waitKey(1)

        now = time.time()
        if now >= next_capture:
            rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(rgb)
            keypoints_list.append(extract_keypoints(result))
            next_capture += frame_interval

    cap.release()

    # Pad with zeros if fewer frames were captured
    while len(keypoints_list) < NUM_FRAMES:
        keypoints_list.append(np.zeros(63))

    return np.array(keypoints_list[:NUM_FRAMES])


def create_support_set(n_way, k_shot, classes):
    """
    Records (or loads) k_shot samples for each of the n_way classes.
    Keypoints are saved as .npy files in SUPPORT_DIR so re-recording is skipped
    on subsequent runs. Delete the folder to re-record from scratch.
    Returns support_x as a torch tensor.
    """
    os.makedirs(SUPPORT_DIR, exist_ok=True)

    existing = [
        f for f in os.listdir(SUPPORT_DIR) if f.endswith(".npy")
    ]
    if existing:
        answer = input(
            f"\nExisting support set found ({len(existing)} files). "
            "Reuse it? [Y/n]: "
        ).strip().lower()
        if answer in ("", "y", "yes"):
            print("Loading existing support set...")
            X = []
            for n in range(1, n_way + 1):
                for s in range(1, k_shot + 1):
                    path = os.path.join(SUPPORT_DIR, f"class{n}_sample{s}.npy")
                    if not os.path.isfile(path):
                        raise FileNotFoundError(
                            f"Missing: {path}. Delete the support_set/ folder and re-run."
                        )
                    X.append(np.load(path))
            return torch.from_numpy(np.array(X)).float()
        else:
            # Wipe and re-record
            import shutil
            shutil.rmtree(SUPPORT_DIR)
            os.makedirs(SUPPORT_DIR)

    X = []
    with mp.solutions.hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.5,
    ) as hands:
        for n, class_name in enumerate(classes):
            for s in range(1, k_shot + 1):
                print(f"\n[{n+1}/{n_way}] Class '{class_name}'  —  sample {s}/{k_shot}")
                print("  Look at the camera and perform the gesture after the countdown.")
                time.sleep(1)

                kp   = _record_one_sample(class_name, hands)
                kp   = move_values(kp, NUM_FRAMES)
                path = os.path.join(SUPPORT_DIR, f"class{n+1}_sample{s}.npy")
                np.save(path, kp)
                print(f"  Saved → {path}")
                X.append(kp)

    cv2.destroyAllWindows()
    print("\nAll support samples recorded.")
    return torch.from_numpy(np.array(X)).float()
# ───────────────────────────────────────────────────────────────────────────────


# ── Real-time inference ────────────────────────────────────────────────────────
def realtime_predict(support_x, classes, feature_encoder, relation_network, params):
    """
    Opens the webcam and continuously classifies gestures using a rolling
    72-frame buffer. Inference runs every STRIDE frames.

    Controls:
        q  →  quit
    """
    buffer      = collections.deque(maxlen=NUM_FRAMES)
    frame_count = 0
    label       = f"Collecting {NUM_FRAMES} frames..."
    score       = 0.0

    mp_draw  = mp.solutions.drawing_utils
    mp_hands = mp.solutions.hands

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Cannot open webcam.")
        return

    print("\nReal-time inference running. Press 'q' to quit.\n")

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.5,
    ) as hands:

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame  = cv2.flip(frame, 1)
            rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(rgb)

            # Extract keypoints and push into buffer
            kp = extract_keypoints(result)
            buffer.append(kp)
            frame_count += 1

            # Run inference once buffer is full, every STRIDE frames
            if len(buffer) == NUM_FRAMES and frame_count % STRIDE == 0:
                window = move_values(np.array(buffer), NUM_FRAMES)
                x = torch.from_numpy(np.expand_dims(window, 0)).float()

                with torch.no_grad():
                    relations = calc_prediction(
                        feature_encoder=feature_encoder,
                        relation_network=relation_network,
                        support_x=support_x,
                        query_x=x,
                        num_classes=params["num_classes"],
                        support_num_per_class=params["support_num_per_class"],
                        seq_length=params["sequence_length"],
                        num_units_lstm_encoder=params["num_units_lstm_encoder"],
                    )

                scores      = relations[0].detach().cpu().numpy()
                best_class  = int(np.argmax(scores))
                best_score  = float(scores[best_class])
                score       = best_score

                if best_score >= THRESHOLD:
                    label = classes[best_class]
                else:
                    label = "..."

            # Draw hand skeleton
            if result.multi_hand_landmarks:
                mp_draw.draw_landmarks(
                    frame,
                    result.multi_hand_landmarks[0],
                    mp_hands.HAND_CONNECTIONS,
                )

            # ── HUD ────────────────────────────────────────────────────────────
            h, w = frame.shape[:2]

            # Prediction label
            color = (0, 220, 0) if score >= THRESHOLD else (160, 160, 160)
            cv2.putText(frame, label,
                        (20, 55), cv2.FONT_HERSHEY_SIMPLEX, 1.6, color, 3)
            if score > 0:
                cv2.putText(frame, f"score: {score:.2f}",
                            (20, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

            # Buffer fill progress bar
            fill   = len(buffer) / NUM_FRAMES
            bar_x0, bar_y0 = 20, h - 30
            bar_x1, bar_y1 = 220, h - 10
            cv2.rectangle(frame, (bar_x0, bar_y0), (bar_x1, bar_y1), (50, 50, 50), -1)
            cv2.rectangle(frame,
                          (bar_x0, bar_y0),
                          (bar_x0 + int((bar_x1 - bar_x0) * fill), bar_y1),
                          (0, 200, 80), -1)
            cv2.putText(frame, f"buffer {len(buffer)}/{NUM_FRAMES}",
                        (230, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

            # Threshold indicator
            cv2.putText(frame, f"threshold: {THRESHOLD}  stride: {STRIDE}",
                        (20, h - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)
            # ───────────────────────────────────────────────────────────────────

            cv2.imshow("Real-time Gesture Recognition  [q = quit]", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()
    print("Stopped.")
# ───────────────────────────────────────────────────────────────────────────────


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"Device : {device}")
    print(f"Config : {N_WAY}-way  {K_SHOT}-shot  |  stride={STRIDE}  threshold={THRESHOLD}\n")

    print("Loading model weights...")
    feature_encoder, relation_network, params = load_model(N_WAY, K_SHOT)
    print("Model ready.\n")

    support_x = create_support_set(N_WAY, K_SHOT, GESTURES)

    realtime_predict(support_x, GESTURES, feature_encoder, relation_network, params)
