# Gesture-Bridge

Gesture-Bridge is an offline assistive-communication prototype that translates a
small sign vocabulary to text/speech and watches *how* a gesture is performed for
signals of distress. It is designed to move from a laptop demo to a Raspberry Pi
without changing the recognition or safety logic.

The graded demonstration uses Google's bundled pretrained Gesture Recognizer—no
per-user training is required. Active mappings are: Open Palm → Hello, Victory →
Help, Thumb Up → Yes, Thumb Down → No, Pointing Up → Doctor, Closed Fist → Emergency,
and I-Love-You → Caregiver. These are assistive command mappings, not a claim that
the poses are standard Indian Sign Language words.

The production UI is camera-first: live recognition, communication output, safety
signals, runtime health and alert state remain visible without covering the user.
Press `G` for an on-screen guide showing the exact prototype hand poses, their
actions, the video-trained movement classes and all covert SOS patterns. The default
renderer uses native OpenCV text in a fixed 1280×720 window: it is deliberately
lighter and sharper than scaling a 720p canvas into a maximized 1080p window. Override
`GESTURE_BRIDGE_DISPLAY_WIDTH` and `GESTURE_BRIDGE_DISPLAY_HEIGHT` only when the
target hardware has enough rendering headroom. Pillow text remains an optional
desktop-only renderer via `GESTURE_BRIDGE_TEXT_RENDERER=pillow`.
MediaPipe analyzes a 640×360 frame by default; this keeps the 1280×720 presentation
sharp without making inference pay the cost of 720p input.

## What is implemented

- Normal sign recognition using MediaPipe hand landmarks.
- A rolling distress score using normalized gesture speed, tremor, repetition,
  and abrupt motion. The dashboard explains which signals raised the score.
- Four covert SOS patterns: three thumb/index taps, fist-open-fist-open, repeated
  thumb rubbing, and repeated open-palm depth pulses.
- Emergency actions: local audit log, optional GPIO buzzer, optional caregiver
  webhook (usable with an SMS/WhatsApp provider), a configured location, and a
  prominent on-screen emergency message.
- A cancelable countdown before visible/non-silent alerts; covert SOS remains
  immediate and silent.
- Context modes for General, Home, Hospital, Classroom, Public Office, and Transport.
  Every active pose has a complete phrase in every context; `X` changes mode and the
  `G` guide updates its result column immediately.
- A small offline gesture-to-sentence grammar layer.
- Text-to-sign instructions, speech output, live FPS,
  response latency, stability, confidence, and distress metrics.

Facial fear/pain and full-body instability need face/pose landmark models and a
proper consented training dataset. The dashboard exposes these as sensor hooks;
the prototype does not pretend that hand motion alone can diagnose emotion.

## Run

```bash
source venv/bin/activate
python app.py
```

Controls: `X` changes context, `Y` confirms a low-confidence suggestion, `B`
stores a calm-motion calibration after the hand has been visible for three seconds,
`A` acknowledges an alert, `K` cancels it, `P` opens text-to-sign, `R` returns to
recognition, `S` speaks the sentence, `C` resets, `G` opens the guide, `T` toggles
metric detail, `-`/`+` adjust interface size, and `Q` quits. The interface also has
a draggable scale control. Click the bottom tabs or press `1` for Recognition, `2`
for Voice, `3` for Safety, and `0` for an uncluttered camera-only view. Clicking the
selected tab again also collapses it. The full shortcut list is visible inside `G`.

Press `K` during the emergency countdown to cancel a false alarm. On clean exit,
the app writes a privacy-conscious aggregate report under `session_reports/`; it
does not store camera frames.

Each deliberate pose is spoken once after roughly 0.3 seconds of stable confidence.
Release the pose briefly before repeating the same command. Normal phrases retain
their order—Open Palm followed by Victory speaks “Hello. I need help.”—while an
Emergency may interrupt queued speech. On macOS the app uses the native `say` engine;
other platforms use the non-blocking pyttsx3 worker.

## Doctor/caregiver phone notifications

The simplest cross-platform option is an ntfy phone notification. On the Mac run:

```bash
python setup_notifications.py --contact "Dr Sharma" --location "Ward 3" --enable-live
```

Send the printed private subscription URL to the intended recipient. They install
the ntfy phone app and subscribe to that exact URL. Then verify delivery before the
demonstration:

```bash
python hardware_self_test.py --send-test-alert
```

The closed-fist Emergency pose starts the visible cancel countdown; if it is not
cancelled with `K`, the phone receives the message, reason, configured/GPS location
and alert ID. The same `.env` works on macOS. On Raspberry Pi, generate the settings
directly into its service environment or copy them there:

```bash
sudo python setup_notifications.py --contact "Dr Sharma" --location "Ward 3" \
  --enable-live --output /etc/gesture-bridge.env
sudo systemctl restart gesture-bridge
```

Keep the generated topic URL private and obtain the recipient's consent. ntfy topics
are bearer-style secrets, not medical-record storage. Twilio SMS/WhatsApp and generic
webhooks remain supported alternatives. Delivery is retried, one failed provider no
longer blocks another, and every state transition is written to the local audit log.

## Optional experimental training tools

The repository retains collection and training tools for future research, but the
live demonstration intentionally uses Google's pretrained recognizer instead. The
experimental tools accept either a listed label or a new custom label:

```bash
python collect_data.py
python train_model.py
python app.py
```

Record each gesture in varied positions and lighting. `train_model.py` writes
`isl_landmark_model.pkl` and `model_metrics.json`; these are experimental artifacts
and are not loaded by the reliable pretrained live path. The current CSV has already
been migrated to the two-hand 126-feature format. For another legacy 63-feature file:

```bash
python migrate_dataset.py isl_custom_dataset.csv isl_custom_dataset_126.csv
mv isl_custom_dataset_126.csv isl_custom_dataset.csv
```

The vocabulary includes Pain, Chest, Medicine, Call, and Caregiver. These require
personal samples before the trained recognizer can emit them.

## Video movement datasets

For dynamic signs, use labelled video rather than individual landmark rows. Good ISL
sources include INCLUDE (263 word signs), the Government of India ISL Dictionary,
and Amrita SLR. Arrange selected clips as `videos/<label>/<clip>.mp4`, then run:

```bash
python build_video_manifest.py videos --output video_manifest.csv --signer-from-filename
python train_temporal_model.py video_manifest.csv
python app.py
```

If a dataset provides official train and test directories, preserve that split:

```bash
python build_video_manifest.py dataset/train --output video_manifest.csv --split train
python build_video_manifest.py dataset/test --output video_manifest.csv --split test --append
```

Sequence extraction is cached in `.sequence_cache`, so interrupted training resumes.
The generated temporal model is not enabled in the live application because its
public-dataset domain shift caused unreliable webcam predictions. Use it for an
evaluation experiment, not the graded live demonstration. Use only ISL data for sign
meanings; ASL datasets do not share the same vocabulary.

The experimental temporal model contains 12 user-facing classes: Hello,
Thank You, Fever, Injury, Drink, Cry, Come, Give, Busy, Break, Maybe and Wrong.
`Still` is an internal idle class and is never spoken. See
`docs/dataset_report.md` for provenance, split controls and limitations.

`collect_data.py` now captures ten separately started repetitions per label and
records repetition metadata. Complete that flow for every label listed as missing
in `model_metrics.json`, then retrain. Accuracy is marked provisional until all
expected labels and recorded repetition groups exist; adjacent legacy frames are
not treated as proof of real-world generalization.

## Readiness self-test

Run the non-invasive check before a demonstration or Pi deployment:

```bash
python hardware_self_test.py
```

It checks the 126-feature dataset, trained model, camera FPS, delivery-provider
configuration, and location without sending a message or activating the buzzer.
On the Raspberry Pi, explicitly test physical outputs only when it is safe:

```bash
python hardware_self_test.py --buzzer
python hardware_self_test.py --send-test-alert
```

The second command sends only when live mode and a provider are configured. Confirm
the caregiver has consented and warn them that the message is a test.

## Participant evaluation

Copy `evaluation_trials.example.csv` to `evaluation_trials.csv`, record the expected
and observed result for each participant trial, then generate quantitative evidence:

```bash
python evaluate_trials.py evaluation_trials.csv
```

This produces accuracy, participant count, response time, false-SOS count and a
confusion matrix. Follow `docs/validation_protocol.md`; do not split frames from one
participant across training and testing.

## Raspberry Pi wiring

Use a Pi 4/5, camera, and an active buzzer through a transistor/driver—not directly
from a heavily loaded GPIO. The default BCM pin is 17. Install `gpiozero`, copy the
project, configure environment variables, and run the same `app.py`. A service
template and installer are included:

```bash
sudo ./deploy/install_pi.sh
sudo nano /etc/gesture-bridge.env
sudo systemctl start gesture-bridge
```

Recognition and safety analysis remain offline; only the optional alert
webhook requires a network.

### USB webcam on Raspberry Pi

USB UVC webcams are supported through OpenCV/V4L2. Set automatic discovery in
`/etc/gesture-bridge.env`:

```bash
GESTURE_BRIDGE_CAMERA_INDEX=-1
```

The app scans indices 0–5 and accepts only a device that returns a real frame. Test
the camera before starting the service:

```bash
ls -l /dev/video*
v4l2-ctl --list-devices
python hardware_self_test.py --camera-index -1 --frames 60
```

If the webcam is known to be `/dev/video2`, use camera index `2`. The systemd service
runs with `video` and `gpio` supplementary groups. After configuration changes run
`sudo systemctl restart gesture-bridge`; use
`journalctl -u gesture-bridge -n 100 --no-pager` to inspect the selected camera or
any permission error.

## Production boundary

The code is hardened for prototype deployment: paths are independent of the launch
directory, speech is non-blocking, logs rotate, alerts default to demo mode, models
are version-checked, and the Pi installer excludes datasets, caches and development
environments. It is not a certified emergency or medical device. A real deployment
still requires participant-level validation, caregiver consent, secret management,
delivery monitoring and hardware endurance testing.

## Safety note

This is an academic prototype, not a certified medical or emergency device.
Thresholds must be calibrated per user, covert gestures need deliberate onboarding,
and real deployments still need encrypted secret provisioning, consent, independent
delivery monitoring, and testing with the intended users.
