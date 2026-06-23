# Storage Audit and Cleanup

Initial audit performed on 20 June 2026:

| Item | Size | Explanation |
|---|---:|---|
| `.git/` | 29 GB | Hidden orphaned Git objects created while multi-GB dataset files existed before ignore rules |
| `data/` | 6.4 GB | 3.2 GB downloaded ZIP plus approximately 3.1 GB extracted videos |
| `venv/` | 1.8 GB | Python environment; about 1.1 GB is unused TensorFlow/Keras |
| `.sequence_cache/` | 32 MB | Reusable MediaPipe landmark sequences |
| Application code/models | under 15 MB | Source, MediaPipe task model and trained classifiers |

The Finder total is therefore roughly 38 GB even though the largest visible folder
is `data/`: `.git` is hidden by default and contains the majority of the space.

## Cleanup result

Garbage collection has now completed. Verified on 23 June 2026:

- `.git/`: approximately **6.3 MB** (previously 29 GB)
- whole project: approximately **8.2 GB** (previously 37–38 GB)
- current source, model and dataset files were preserved

## Safe cleanup

The large Git packs were unreachable. The only referenced Codex checkpoint was
inspected and contains normal source/model files, not the dataset. Run:

```bash
git gc --prune=now
```

This reduced `.git` from approximately 29 GB to a few megabytes while preserving
current files, the index and referenced checkpoints. It only needs repeating if
large ignored files are accidentally added to Git again.

After confirming the extracted videos work, the verified source archive can also be
removed to save about 3.2 GB:

```bash
rm data/raw/isl_video_60/dataset.zip
```

Its SHA-256 is recorded in `docs/dataset_report.md`, and it can be downloaded again.
TensorFlow and Keras are not used by Gesture-Bridge; recreating `venv` from
`requirements.txt` is safer than manually deleting packages and saves roughly 1.1 GB.

Do not commit `data/`, `.sequence_cache/`, `venv/`, runtime logs, API credentials or
caregiver information. The repository ignore rules now cover all of them.
