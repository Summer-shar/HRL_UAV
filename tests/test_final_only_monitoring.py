import contextlib
import copy
import csv
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from hrl_uav.experiment import smoke_config
from hrl_uav.qualification import metered_evaluate, run_mainline
from hrl_uav.resumption import load_source, save_boundary_checkpoint, state_equal, validate_resume_config


ROOT = Path(__file__).resolve().parents[1]


class FinalOnlyMonitoringTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.config = smoke_config(json.loads((ROOT / 'configs/hrl_resume_joint600.json').read_text()))
        self.config['training'].update(pretrain_windows=2, joint_windows=4, replay_capacity=64,
                                       ppo_update_macros=7, ppo_std_decay_windows=4)
        self.config['qualification'].update(validation_every_windows=1, tail_checkpoints=1,
                                            validation_macros=6, validation_seeds=[33000], test_seeds=[33500])
        self.config['validation_seeds'], self.config['test_seeds'] = [33000], [33500]

    def target(self):
        target = copy.deepcopy(self.config)
        target['qualification'].update(joint_evaluation_mode='final_only', checkpoint_every_windows=1)
        return target

    def test_monitoring_override_does_not_allow_learning_or_evaluation_changes(self):
        target = self.target()
        validate_resume_config(self.config, target, 2, 'joint')
        for section, key, value in (
            ('training', 'sac_update_frames', 9),
            ('qualification', 'test_seeds', [33501]),
            ('qualification', 'validation_macros', 3),
            ('reward', 'delay_weight', 0.8),
        ):
            changed = copy.deepcopy(target)
            changed[section][key] = value
            with self.assertRaisesRegex(ValueError, 'only'):
                validate_resume_config(self.config, changed, 2, 'joint')
        with self.assertRaisesRegex(ValueError, 'only'):
            validate_resume_config(self.config, target, 1, 'pretrain')

    def test_invalid_monitoring_is_rejected_before_output_creation(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'invalid'
            for key, value in (('joint_evaluation_mode', 'typo'), ('checkpoint_every_windows', 0)):
                bad = self.target()
                bad['qualification'][key] = value
                with self.assertRaises(ValueError):
                    run_mainline(bad, output)
                self.assertFalse(output.exists())

    def test_boundary_resume_skips_validation_but_preserves_exact_training(self):
        class PauseAtBoundary(Exception):
            pass

        def pause(agent, config, path, completed, next_seed):
            save_boundary_checkpoint(agent, config, path, completed, next_seed)
            if agent.stage == 'joint' and completed == 2:
                raise PauseAtBoundary()

        def rows(path):
            with path.open(encoding='utf-8-sig', newline='') as handle:
                return list(csv.DictReader(handle))

        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            root = Path(directory)
            control_report = run_mainline(self.config, root / 'control')
            with patch('hrl_uav.qualification.save_boundary_checkpoint', side_effect=pause):
                with self.assertRaises(PauseAtBoundary):
                    run_mainline(self.config, root / 'paused')
            self.assertFalse((root / 'paused/report.json').exists())
            saved, metadata = load_source(root / 'paused/joint_latest.pt', self.target())
            self.assertEqual(metadata['start_window'], 2)
            self.assertEqual(metadata['source_kind'], 'rollout_boundary')
            self.assertGreater(len(saved['training_state']['macro_rollout']), 0)

            with patch('hrl_uav.qualification.metered_evaluate', wraps=metered_evaluate) as evaluate:
                report = run_mainline(self.target(), root / 'resumed', resume_from=root / 'paused/joint_latest.pt')
            self.assertEqual([call.args[5] for call in evaluate.call_args_list],
                             ['final', 'uniform_test', 'centroid_same_sac'])
            self.assertFalse((root / 'resumed/validation.csv').exists())
            self.assertTrue((root / 'resumed/joint_latest.pt').is_file())
            self.assertEqual(report['monitoring']['joint_evaluation_mode'], 'final_only')
            self.assertEqual(report['mean'], control_report['mean'])
            self.assertEqual(report['updates_total'], control_report['updates_total'])
            self.assertEqual([row for row in rows(root / 'control/training.csv')
                              if row['stage'] == 'joint' and int(row['window']) > 2],
                             rows(root / 'resumed/training.csv'))
            full = torch.load(root / 'control/final.pt', map_location='cpu', weights_only=False)
            resumed = torch.load(root / 'resumed/final.pt', map_location='cpu', weights_only=False)
            full.pop('config')
            resumed.pop('config')
            self.assertTrue(state_equal(full, resumed))


if __name__ == '__main__':
    unittest.main()
