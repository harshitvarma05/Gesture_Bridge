import tempfile
import unittest
import csv
import numpy as np
from pathlib import Path
from types import SimpleNamespace as Point

from gesture_bridge.alerts import AlertManager
from gesture_bridge.intelligence import ContextInterpreter, SentenceEngine
from gesture_bridge.safety import SafetyAnalyzer
from migrate_dataset import migrate
from hardware_self_test import check_dataset
from build_video_manifest import build_manifest
from gesture_bridge.temporal import FRAME_FEATURES, SEQUENCE_STEPS, sequence_descriptor
from prepare_isl_video_60 import prepare, source_stem


def make_hand(pinched=False):
    points = [Point(x=0.5, y=0.5, z=0.0) for _ in range(21)]
    points[5] = Point(x=0.4, y=0.4, z=0.0)
    points[17] = Point(x=0.6, y=0.4, z=0.0)
    points[4] = Point(x=0.45, y=0.3, z=0.0)
    points[8] = Point(x=0.46 if pinched else 0.7, y=0.3, z=0.0)
    for index, x in zip((12, 16, 20), (0.5, 0.55, 0.6)):
        points[index] = Point(x=x, y=0.1, z=0.0)
    return points


class SafetyTests(unittest.TestCase):
    def test_three_taps_raise_silent_sos(self):
        analyzer = SafetyAnalyzer()
        state = None
        for offset, pinched in ((0, False), (.1, True), (.2, False), (.5, True), (.6, False), (.9, True)):
            state = analyzer.update([make_hand(pinched)], now=10 + offset)
        self.assertEqual(state.sos_pattern, "3 finger taps")
        self.assertEqual(state.level, "SOS")

    def test_no_hand_is_calm(self):
        self.assertEqual(SafetyAnalyzer().update([]).level, "CALM")

    def test_personal_calibration_is_persisted(self):
        with tempfile.TemporaryDirectory() as directory:
            profile = f"{directory}/profile.json"
            analyzer = SafetyAnalyzer(profile_path=profile)
            for index in range(20):
                sample = make_hand(False)
                for point in sample:
                    point.x += index * 0.001
                analyzer.update([sample], now=10 + index * 0.05)
            self.assertTrue(analyzer.calibrate_from_window())
            self.assertTrue(Path(profile).exists())


class IntelligenceTests(unittest.TestCase):
    def test_context_changes_help_phrase(self):
        interpreter = ContextInterpreter()
        interpreter.cycle()
        self.assertEqual(interpreter.context, "Hospital")
        self.assertIn("medical", interpreter.interpret("Help"))

    def test_sentence_engine_handles_help_doctor(self):
        engine = SentenceEngine()
        engine.add("Help")
        self.assertEqual(engine.add("Doctor"), "I need help. Please call a doctor.")


class AlertTests(unittest.TestCase):
    def test_demo_alert_is_only_recorded(self):
        with tempfile.TemporaryDirectory() as directory:
            manager = AlertManager(f"{directory}/alerts.jsonl")
            payload = manager.trigger("test", "message", silent=True)
            self.assertEqual(payload["mode"], "demo")
            self.assertIn("recorded", manager.last_status.lower())
            self.assertTrue(manager.acknowledge())
            self.assertEqual(manager.active_alert["state"], "ACKNOWLEDGED")
            self.assertTrue(manager.cancel())
            self.assertEqual(manager.active_alert["state"], "CANCELLED")


class DatasetTests(unittest.TestCase):
    def test_legacy_rows_are_migrated_to_126_features(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "legacy.csv"
            destination = Path(directory) / "current.csv"
            with source.open("w", newline="") as handle:
                csv.writer(handle).writerows([["label", *[f"f{i}" for i in range(63)]], ["Help", *(["0"] * 63)]])
            self.assertEqual(migrate(source, destination), 1)
            with destination.open(newline="") as handle:
                rows = list(csv.reader(handle))
            self.assertEqual(len(rows[0]), 127)
            self.assertEqual(len(rows[1]), 127)

    def test_readiness_check_accepts_current_schema(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset = Path(directory) / "dataset.csv"
            with dataset.open("w", newline="") as handle:
                csv.writer(handle).writerows([
                    ["label", *[f"f{i}" for i in range(126)]],
                    ["Help", *(["0"] * 126)],
                    ["Hello", *(["0"] * 126)],
                ])
            result = check_dataset(dataset)
            self.assertTrue(result["ok"])
            self.assertEqual(result["features"], 126)

    def test_video_manifest_infers_labels_from_folders(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "videos"
            (root / "Chest").mkdir(parents=True)
            (root / "Chest" / "sample.mp4").touch()
            output = Path(directory) / "manifest.csv"
            self.assertEqual(build_manifest(root, output), 1)
            with output.open(newline="") as handle:
                row = next(csv.DictReader(handle))
            self.assertEqual(row["label"], "Chest")

    def test_augmented_video_variants_never_cross_splits(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "videos"
            label = root / "Hello"
            label.mkdir(parents=True)
            for index in range(8):
                for suffix in ("", "_left_tilt", "_right_tilt"):
                    (label / f"clip_{index}{suffix}.mp4").touch()
            rows, _ = prepare(root, Path(directory) / "manifest.csv")
            split_by_source = {}
            for row in rows:
                key = source_stem(row["path"])
                split_by_source.setdefault(key, set()).add(row["split"])
            self.assertTrue(all(len(splits) == 1 for splits in split_by_source.values()))


class TemporalTests(unittest.TestCase):
    def test_descriptor_is_fixed_size_and_motion_order_sensitive(self):
        sequence = np.zeros((20, FRAME_FEATURES), dtype=np.float32)
        sequence[:, 0] = np.linspace(0, 1, 20)
        forward = sequence_descriptor(sequence)
        backward = sequence_descriptor(sequence[::-1])
        self.assertEqual(forward.shape, ((SEQUENCE_STEPS + 4) * FRAME_FEATURES,))
        self.assertFalse(np.allclose(forward, backward))


if __name__ == "__main__":
    unittest.main()
