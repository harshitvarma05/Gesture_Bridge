# Gesture-Bridge Validation Protocol

This protocol turns the prototype into defensible interdisciplinary-project evidence.
Targets below are project acceptance criteria, not results until trials are completed.

## Participants and ethics

- Recruit at least five consenting participants.
- Include at least one participant whose recordings are never used for calibration or training.
- Explain that this is an academic prototype, not an emergency or medical device.
- Do not record identifiable video unless separately consented; trial CSVs need only participant IDs.

## Recognition trial

Test the seven pretrained command poses shown in the GUI. Each participant performs
every pose five times, giving at least 175 trials for five participants. Vary distance, lighting
and camera angle across trials. Record:

- participant ID and trial number;
- expected gesture;
- displayed/confirmed gesture, or blank if none;
- response time shown by the application;
- whether a false SOS occurred.

Use `evaluation_trials.example.csv` as the template and run:

```bash
python evaluate_trials.py evaluation_trials.csv --output evaluation_report.json
```

Report overall accuracy, per-class confusion, average response time and results for
the completely unseen participant. Do not quote the public-dataset score as live-user accuracy.

## Unknown-gesture trial

Ask each participant to perform ten ordinary non-sign hand movements. Record whether
the application confirms any trained sign. Target: fewer than 5% false confirmations.

## Silent SOS trial

For every SOS pattern, run ten deliberate trials and record successful triggers.
Then conduct at least 30 minutes of normal signing/movement without an SOS. Target:

- at least 90% deliberate SOS detection in the calibrated user;
- zero false caregiver deliveries during the normal-motion run;
- every visible alert can be cancelled during its countdown.

## Performance trial

Run a five-minute session on the laptop and Raspberry Pi. Use the generated JSON
session report to capture average FPS, hand visibility and unknown-frame percentage.
Project targets:

- laptop: at least 20 FPS;
- Raspberry Pi: at least 12 FPS;
- median confirmed-response time below 1.0 second;
- no crash or camera loss during the five-minute run.

## Demo runbook

1. Run `python hardware_self_test.py` before the presentation.
2. Confirm alerts are in demo mode unless the caregiver expects a test.
3. Calibrate calm motion with `B`.
4. Demonstrate Open Palm → Hello, Victory → Help and Closed Fist → Emergency.
5. Cancel the visible alert with `K`.
6. Demonstrate a silent SOS and show its audit entry.
7. Exit with `Q` and show the generated session report.

Keep a prerecorded backup demonstration in case venue camera permissions, lighting or
network access fail. The backup should be identified honestly as prerecorded.
