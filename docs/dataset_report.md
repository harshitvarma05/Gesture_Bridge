# Gesture-Bridge Dataset Report

## Selected corpus

**Indian Sign Language Video Dataset** by Prasad Shet et al.  
Source: <https://www.kaggle.com/datasets/prasadshet/indian-sign-language-video-dataset>  
Stated license: MIT  
Archive size: 3.48 GB (3.2 GiB on disk)  
SHA-256: `6fdb526da7f40818d88877a2e23413459e27d709800250b9e5b6d5e154d4f636`

The dataset contains word-level movement videos derived from official ISL examples
and recordings by four contributors. The extracted corpus has 61 labels, 3,630
videos and 1,210 independent source recordings. Most source recordings have original,
left-tilt and right-tilt variants.

## Leakage control

`prepare_isl_video_60.py` strips tilt suffixes to recover the source-recording ID.
Every original and its augmented variants are assigned to the same partition. The
result is an exact class-balanced 80/20 split wherever class counts permit:

- Training videos: 2,904
- Test videos: 726
- Source recordings: 1,210

This prevents a transformed copy of a training clip from appearing in the test set.
Signer IDs are not provided, so this is not a signer-independent evaluation.

## First assistive temporal model

The initial live model uses 750 videos across 13 relevant interaction/health classes:

`Break, Busy, Come, Cry, Drink, Fever, Give, Hello, Injury, Maybe, Still, Thank you, Wrong`

- Held-out videos: 150
- Manifest-defined grouped accuracy: 100%
- Unknown-rejection acceptance on held-out data: 146/150 (97.33%)
- Accuracy among accepted held-out videos: 100%
- Model size: 2.1 MB
- Representation: 12-step ordered two-hand shape plus centroid/scale trajectory and motion statistics

The high score indicates that this particular dataset is visually separable; it is
not evidence of universal real-world accuracy. The split is source-recording-safe but
not signer-independent. Live testing with unseen people, lighting, backgrounds and
camera positions remains required.

## Unknown gesture protection

The temporal model rejects a sequence when either:

1. the top two class probabilities are insufficiently separated, or
2. the sequence lies outside the learned class-centroid radius.

The `Still` class is treated as an idle/negative class and is never spoken to the user.
Rejected sequences fall back to the explicit prototype gesture rules.

## Remaining custom data

No public corpus covers the project's covert SOS and distress behaviours. Collect
consented, user-specific recordings for:

- three finger taps;
- fist-open-fist-open;
- thumb rubbing;
- palm pulses;
- calm versus rapid/repeated/trembling performance;
- Help, Emergency, Doctor, Water, Pain, Chest, Medicine, Call and Caregiver.

These samples must be split by participant—not by frame—to measure generalization.
