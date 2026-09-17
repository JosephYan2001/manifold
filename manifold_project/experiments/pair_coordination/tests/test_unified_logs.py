import json
from pathlib import Path
import tempfile
import unittest
import zipfile
from manifold_project.experiments.pair_coordination.training.logging import append_json, read_records
from manifold_project.experiments.pair_coordination.analysis.merge_logs import merge_suite


class UnifiedLogsTests(unittest.TestCase):
    def test_streams_and_partial_read(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            events = directory/'events.jsonl'
            events.touch()
            append_json(directory/'policy.jsonl', {'round': 0})
            append_json(directory/'checks.jsonl', {'kind': 'return'})
            self.assertEqual(read_records(directory/'policy.jsonl'), [{'round': 0}])
            self.assertFalse((directory/'policy.jsonl').exists())
            with events.open('a') as stream:
                stream.write('{')
            self.assertEqual(read_records(directory/'checks.jsonl', True), [{'kind': 'return'}])
            with self.assertRaises(json.JSONDecodeError):
                read_records(directory/'policy.jsonl')

    def test_migration_preserves_data_and_skips_running(self):
        with tempfile.TemporaryDirectory() as tmp:
            suite = Path(tmp)
            (suite/'status.json').write_text(json.dumps({'jobs': {
                'training/ours/seed_1': {'status': 'complete'},
                'training/ours/seed_2': {'status': 'running'}}}))
            run = suite/'training/ours/seed_1'
            run.mkdir(parents=True)
            (run/'summary.json').write_text(json.dumps({'stop_reason': 'source_budget'}))
            append_json(run/'policy.jsonl', {'round': 1})
            append_json(run/'checks.jsonl', {'accepted': False})
            original = (run/'policy.jsonl').read_bytes()
            result = merge_suite(suite, True)
            self.assertEqual(result['old_files'], 2)
            self.assertEqual(list(run.glob('*.jsonl')), [run/'events.jsonl'])
            self.assertEqual(read_records(run/'policy.jsonl'), [{'round': 1}])
            with zipfile.ZipFile(suite/'legacy_logs.zip') as archive:
                self.assertEqual(archive.read('training/ours/seed_1/policy.jsonl'), original)
            self.assertEqual(merge_suite(suite, True)['runs'], 0)
