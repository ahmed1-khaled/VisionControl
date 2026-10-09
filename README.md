# VisionControl 👁️

### Eye & Hand Gesture-Based Computer Control

Control your computer mouse using real-time eye tracking and hand gestures with Python, OpenCV, and MediaPipe.

VisionControl is an experimental computer vision project that explores hands-free computer interaction through webcam-based facial and hand landmark detection.

---

## ✨ Features

* 👁️ **Eye Tracking:** Estimate gaze direction using iris and eye landmarks.
* 🖐️ **Hand Gesture Recognition:** Detect hand poses and finger positions in real time.
* 🖱️ **Cursor Movement:** Control mouse movement through eye tracking or index-finger gestures.
* 👆 **Mouse Actions:** Support left-click, right-click, and double-click gestures.
* ↕️ **Scrolling:** Scroll vertically using predefined hand gestures.
* 🎯 **Eye Calibration:** Establish a neutral gaze reference at startup.
* 📊 **Live Feedback:** Display the webcam feed, current mode, status, and FPS.
* ⚙️ **Adjustable Settings:** Tune sensitivity, smoothing, and gesture thresholds.

## 🛠️ Built With

| Technology | Purpose                                 |
| ---------- | --------------------------------------- |
| Python     | Core application logic                  |
| OpenCV     | Webcam capture and image processing     |
| MediaPipe  | Face, iris, and hand landmark detection |
| NumPy      | Numerical calculations                  |
| PyAutoGUI  | Mouse automation                        |

## 🚀 Getting Started

### Prerequisites

* Python installed on your computer
* A working webcam
* A supported desktop environment
* The required Python dependencies

### Installation

**1. Clone the repository**

```bash
git clone https://github.com/ahmed1-khaled/vision-control.git
cd vision-control
```

**2. Create a virtual environment**

```bash
python -m venv .venv
```

**3. Activate the environment**

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Linux/macOS:

```bash
source .venv/bin/activate
```

**4. Install dependencies**

```bash
pip install -r requirements.txt
```

**5. Run VisionControl**

```bash
python main.py
```

## 🎮 Controls

### Eye Mode

Move your gaze to influence the mouse cursor. The application uses a short initial calibration period to establish a neutral gaze reference.

### Hand Mode

| Hand gesture                  | Intended action |
| ----------------------------- | --------------- |
| Index finger raised           | Move cursor     |
| Thumb + index finger pinch    | Left-click      |
| Thumb + middle finger pinch   | Right-click     |
| Index + middle fingers raised | Double-click    |
| Vertical two-finger gesture   | Scroll          |
| Closed fist                   | Pause control   |
| Open hand                     | Resume control  |

Gesture recognition depends on camera position, lighting, hand orientation, and tracking quality.

### Keyboard Shortcuts

| Key     | Action                                |
| ------- | ------------------------------------- |
| `M`     | Switch between Eye Mode and Hand Mode |
| `SPACE` | Toggle controller state               |
| `Q`     | Exit the application                  |

## ⚙️ Configuration

The main script includes parameters for:

* Eye sensitivity and dead zone
* Eye and hand movement smoothing
* Calibration duration
* Pinch detection threshold
* Click cooldowns
* Scroll threshold and speed

Adjust these parameters in `main.py` to experiment with different interaction settings.

## ⚠️ Limitations

* Gaze estimation is approximate and may require further calibration.
* Cursor movement can be affected by lighting, head movement, and camera quality.
* Hand gestures may be misclassified when landmarks are occluded.
* Fixed thresholds may not work equally well for every user.
* Behavior may vary across operating systems.

## 🔒 Safety

VisionControl can control the system mouse. Keep a physical mouse available while testing, and avoid using the application for sensitive actions until its behavior has been verified.

The current implementation enables PyAutoGUI's fail-safe feature.

## 🗺️ Roadmap

* [ ] Improve gaze-to-screen mapping and cursor stability
* [ ] Make pause and resume behavior consistent across modes
* [ ] Reduce accidental gesture-triggered clicks
* [ ] Add configurable gesture mappings
* [ ] Capture and document real application demonstrations
* [ ] Add automated tests for core logic where practical

## 📌 Project Status

Experimental project under development.

## 👨‍💻 Author

**Ahmed Khaled Elkhateb**

* GitHub: [@ahmed1-khaled](https://github.com/ahmed1-khaled)
* LinkedIn: [Ahmed Khaled](https://www.linkedin.com/in/a7med-khaledai)

## 📄 License

License information will be added once the project owner selects the intended usage and distribution terms.
