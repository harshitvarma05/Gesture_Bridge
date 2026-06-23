# IDP Poster Content — Gesture-Bridge

## Project Title

**GESTURE-BRIDGE: A DISTRESS-AWARE SILENT SOS AND ASSISTIVE SIGN COMMUNICATION SYSTEM**

Add below the title: **Team ID:** [ID]  |  **Guide:** [Name, Designation, Department]

## Abstract

Gesture-Bridge is an offline assistive communication prototype that interprets hand gestures as text and speech while analysing the manner in which a gesture is performed. MediaPipe hand landmarks are used to recognize signs and measure motion speed, repetition, tremor and abrupt movement. A hidden micro-gesture channel detects silent SOS patterns such as three finger taps, fist-open sequences, thumb rubbing and palm pulses. On emergency detection, the system can display a warning, activate a Raspberry Pi buzzer, record an auditable alert and share a configured or GPS-derived location with a caregiver. Context-aware phrases, sentence formation, personalized gesture learning and confidence-based clarification extend the system beyond conventional sign recognition.

## Problem Statement

People with speech or hearing disabilities may find it difficult to communicate quickly with individuals who do not understand sign language. During fear, pain or coercion, a user may also be unable to speak, reach a phone or perform a large visible emergency gesture. Most gesture-recognition systems focus only on identifying the sign and ignore distress-related motion behaviour and covert emergency intent. A low-cost, privacy-preserving system is therefore needed to translate gestures and provide a discreet emergency channel without depending continuously on cloud connectivity.

## Objectives

- Recognize a controlled Indian Sign Language-oriented gesture vocabulary in real time.
- Convert stable gestures into contextual text, complete sentences and speech.
- Estimate distress from gesture speed, repetition, tremor and abrupt motion.
- Detect covert SOS micro-gestures without requiring speech or a phone.
- Trigger display, buzzer, caregiver-message and location-sharing actions.
- Support personalized gestures and confidence-based user clarification.
- Operate offline with low latency and migrate to Raspberry Pi hardware.

## Methodology

Use the following horizontal flow diagram:

**Camera Input** → **Frame Pre-processing** → **MediaPipe Hand Detection** → **21 Landmarks per Hand** → **Static + Temporal Gesture Recognition** → **Context/Sentence Engine** → **Text and Speech Output**

Add a safety branch after landmark extraction:

**Motion Feature Window** → **Speed + Tremor + Repetition + Abruptness** → **Distress Score / SOS Pattern** → **Alert Manager** → **Display + Buzzer + SMS/WhatsApp/Webhook + Location**

Small note below the diagram:

*A stability filter confirms a gesture over multiple frames. Low-confidence predictions remain unconfirmed instead of producing an uncertain message.*

## Experimentation / Hardware / Simulation / Software Model

### Present prototype

- Laptop/desktop webcam and display
- Python, OpenCV and MediaPipe Tasks
- Up to two hands; 21 three-dimensional landmarks per hand
- Google MediaPipe pretrained seven-pose gesture recognizer
- Fixed assistive mappings: Open Palm, Victory, Thumb Up/Down, Pointing Up, Closed Fist and I-Love-You
- Offline text-to-speech using pyttsx3
- JSONL alert audit log with acknowledgement and cancellation

### Raspberry Pi target

- Raspberry Pi 4/5 with Pi Camera or USB webcam
- Active buzzer connected through a transistor/driver to GPIO 17
- Optional GPS receiver through GPSD
- Local display; optional network only for caregiver message delivery

### Suggested visual

Place a labelled block/circuit image containing:

**Pi Camera → Raspberry Pi → Display/Speaker**  
**Raspberry Pi GPIO → Transistor/Driver → Buzzer**  
**Optional GPS → Raspberry Pi → Caregiver Alert Service**

Add a screenshot of the Gesture-Bridge dashboard beside or below the circuit diagram.

## Results

- Pretrained gesture recognition runs offline and requires no per-user model training.
- MediaPipe supplies 21 three-dimensional landmarks per detected hand for distress and SOS analysis.
- Stable recognition uses a **10-frame confirmation window** to reduce flickering output.
- **Four covert SOS patterns** and four motion-distress features are implemented.
- Emergency alerts support unique IDs, retries, audit states, acknowledgement and cancellation.
- **17 automated software tests passed**, covering SOS detection, cancelable alerts, calibration, configuration, telemetry, evaluation metrics, dataset migration, user-facing actions, contextual sentences and temporal motion features.
- Experimental custom and temporal datasets are retained for future work but excluded from the graded live path.

### Suggested result graphics

1. Large GUI screenshot with the safety score, FPS and alert status visible.
2. Small status chart:
   - Automated tests passed: **17/17**
   - SOS patterns implemented: **4**
   - Distress features implemented: **4**
   - Current measured FPS: **≈16**
3. Small dataset-readiness graphic: **Current: 3 labels → Target: complete emergency/assistive vocabulary with multiple signers**.

Do not present the provisional three-class validation score as final recognition accuracy.

## Outcome

- Developed a working gesture-to-text and offline speech prototype.
- Added a novel dual pathway for communication and distress-aware emergency intent.
- Implemented discreet SOS recognition without a separate emergency button.
- Added context-aware phrase generation and multi-sign sentence composition.
- Added six deployment contexts—General, Home, Hospital, Classroom, Public Office
  and Transport—with a complete phrase mapping for every active pretrained pose.
- Created configurable caregiver alerts with location and Raspberry Pi buzzer support.
- Added personalized calibration, dataset-quality checks and a video-based temporal training pipeline.
- Established a deployable foundation for hospitals, classrooms and public-service environments.

## References

1. F. Zhang et al., “MediaPipe Hands: On-device Real-time Hand Tracking,” arXiv:2006.10214, 2020.
2. A. Sridhar, R. G. Ganesan, P. Kumar and M. Khapra, “INCLUDE: A Large Scale Dataset for Indian Sign Language Recognition,” ACM Multimedia, pp. 1366–1375, 2020. doi:10.1145/3394171.3413528.
3. A. Joshi et al., “CISLR: Corpus for Indian Sign Language Recognition,” EMNLP, pp. 10357–10366, 2022. doi:10.18653/v1/2022.emnlp-main.707.
4. A. Duarte et al., “How2Sign: A Large-Scale Multimodal Dataset for Continuous American Sign Language,” CVPR, pp. 2735–2744, 2021.
5. E. Uboweja et al., “On-device Real-time Custom Hand Gesture Recognition,” arXiv:2309.10858, 2023.

## QR Code Content

Link one public folder or project page containing:

- Final presentation PDF/PPT
- 60–90 second working-model demonstration video
- System block diagram
- Project abstract and team details
- Optional source-code/readme link

Do not place private API keys, caregiver phone numbers, raw participant videos or confidential data in the QR-linked folder.
