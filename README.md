# Gesture-Bridge

Gesture-Bridge is an offline assistive-communication prototype that translates a
small sign vocabulary to text/speech and watches *how* a gesture is performed for
signals of distress. It is designed to move from a laptop demo to a Raspberry Pi
without changing the recognition or safety logic.

The production UI is camera-first: live recognition, communication output, safety
signals, runtime health and alert state remain visible without covering the user.
Press `G` for the capability guide; it distinguishes video-trained signs, landmark
rules and gestures that still require personal enrollment.

## What is implemented

- Normal sign recognition using MediaPipe hand landmarks.
- A rolling distress score using normalized gesture speed, tremor, repetition,
  and abrupt motion. The dashboard explains which signals raised the score.
- Four covert SOS patterns: three thumb/index taps, fist-open-fist-open, repeated
  thumb rubbing, and repeated open-palm depth pulses.
- Emergency actions: local audit log, optional GPIO buzzer, optional caregiver
  webhook (usable with an SMS/WhatsApp provider), a configured location, and a
  prominent on-screen emergency message.
- Context modes for General, Hospital, Classroom, and Public Office.
- A small offline gesture-to-sentence grammar layer.
- Text-to-sign instructions, speech output, personalized model loading, live FPS,
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
a draggable scale control.

The default is intentionally safe: alerts are written to
`emergency_alerts.jsonl` but no real message is sent. Copy the values in
`config.example.env` into your Pi service environment. Set
`GESTURE_BRIDGE_LIVE_ALERTS=1` only after supplying a real webhook and testing the
recipient's consent. The webhook receives an alert ID, lifecycle state, timestamp,
reason, message, location, retry attempt, and whether the alert is silent. GPSD can
supply live GPS; Twilio variables enable direct SMS or WhatsApp delivery. Delivery
is retried and every state transition is appended to the audit log.

## Personalized gestures

The collection tool accepts either a listed label or a new custom label:

```bash
python collect_data.py
python train_model.py
python app.py
```

Record each gesture in varied positions and lighting. `train_model.py` writes
`isl_landmark_model.pkl` and `model_metrics.json`; the main app loads it and falls back to
the built-in rule recognizer when confidence is low. The current CSV has already
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
The live app automatically prefers `isl_temporal_model.pkl` once at least ten hand
frames are visible, then falls back to the personalized frame model and heuristics.
Use only ISL data for sign meanings; ASL datasets such as How2Sign are useful for
researching pose/face pipelines but do not share the same vocabulary.

The installed temporal model currently contains 12 user-facing classes: Hello,
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
