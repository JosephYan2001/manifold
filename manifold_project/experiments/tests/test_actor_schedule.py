"""Small CPU checks for the six-run plan; no performance training."""
from contextlib import redirect_stdout, redirect_stderr
from copy import deepcopy
from io import StringIO
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from manifold_project.experiments.common.runtime import configure_runtime
configure_runtime(True)
import torch
from manifold_project.experiments.cli import main, parser, resolve
from manifold_project.experiments.common.actor_schedule import initial_schedule, rate_at, restore_schedule
from manifold_project.experiments.common.config import load_config
from manifold_project.experiments.cooperative_navigation.current_protocol import runner_config
from manifold_project.experiments.cooperative_navigation.training.runner import Runner
from manifold_project.experiments.cooperative_navigation.training.actor_fit import fit_actor
from manifold_project.experiments.cooperative_navigation.training.storage import load_pt
from manifold_project.experiments.tests.test_direction_remediation import settings
from manifold_project.experiments.tests.test_recovery import same


class ActorScheduleTests(unittest.TestCase):
    def setUp(self):
        workspace = Path(__file__).resolve().parents[3]
        self.tmp = tempfile.TemporaryDirectory(prefix='actor_schedule_test_', dir=workspace)
        self.root = Path(self.tmp.name).resolve()
        assert self.root.is_relative_to(workspace) and self.root != workspace

    def tearDown(self):
        self.tmp.cleanup()

    def invoke(self, args):
        with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
            return main(args + ['--allow-duplicate-openmp'])

    def test_clock_values_and_legacy_recovery(self):
        c = dict(actor_lr=3e-4, actor_lr_schedule='linear', actor_lr_final=3e-5,
                 actor_lr_decay_steps=2000000)
        state = initial_schedule(c)
        self.assertEqual(rate_at(state, 8000)['actor_lr_used'], 3e-4)
        self.assertAlmostEqual(rate_at(state, 1004000)['actor_lr_used'], 1.65e-4, places=15)
        self.assertAlmostEqual(rate_at(state, 2000000)['actor_lr_used'], 3e-5, places=15)
        self.assertAlmostEqual(rate_at(state, 3000000)['actor_lr_used'], 3e-5, places=15)
        self.assertEqual(state, restore_schedule(c, deepcopy(state)))
        self.assertEqual(rate_at(restore_schedule(dict(actor_lr=3e-4), None), 1000000)['actor_lr_used'], 3e-4)
        with self.assertRaises(ValueError):
            restore_schedule(c, None)
        with self.assertRaises(ValueError):
            restore_schedule(dict(c, actor_lr_decay_steps=4000000), state)

    def test_six_configs_initialization_and_first_round_rate(self):
        config_root = Path(__file__).resolve().parents[1] / 'configs/navigation/batch_lr_2m'
        hashes, resolved = [], []
        for n in (16, 32, 64):
            for schedule in ('fixed', 'linear'):
                path = config_root/f'n{n}_{schedule}.json'
                args = parser().parse_args(['--experiments','N-C','--profile','pilot','--methods','AN',
                                            '--config',str(path),'--device','cpu','--dry-run'])
                _, c = resolve(args)[0]
                self.assertEqual((c['budget'],c['evaluation_seed'],c['evaluation_episodes']),
                                 (2000000,20262000,100))
                legacy = runner_config(c, method='AN')
                self.assertEqual(legacy['minibatch_episodes'], 2)
                self.assertEqual((legacy['direction_epochs'],legacy['actor_epochs'],legacy['critic_epochs']), (4,16,10))
                r = Runner(self.root/f'n{n}_{schedule}',legacy,'no_checks',40)
                try:
                    hashes.append(r.initial_model_sha256)
                    self.assertEqual(rate_at(r.actor_lr_schedule_state,8000)['actor_lr_used'], 3e-4)
                finally:
                    r.sampler.close()
                resolved.append({k:v for k,v in c.items() if k not in ('batch_episodes','actor_lr_schedule')})
        self.assertTrue(all(h == hashes[0] for h in hashes))
        self.assertTrue(all(c == resolved[0] for c in resolved))

    def test_fixed_matches_existing_fit_and_first_linear_round(self):
        outcomes = []
        for name in ('existing_fit','fixed','linear'):
            c = settings()
            c.update(budget=128, eval_episodes=1, final_episodes=1, eval_fractions=[0.,1.],
                     actor_fit_selection='best_train_kl', actor_epochs=2)
            if name == 'linear':
                c.update(actor_lr_schedule='linear',actor_lr_final=3e-5,actor_lr_decay_steps=128)
            r = Runner(self.root/name,c,'no_checks',40)
            actual_rates = []
            def fitting(*args, **kwargs):
                actual_rates.append(kwargs.get('lr'))
                if name == 'existing_fit':
                    kwargs.pop('lr',None)
                return fit_actor(*args,**kwargs)
            try:
                with patch('manifold_project.experiments.cooperative_navigation.training.runner.fit_actor',side_effect=fitting):
                    with redirect_stdout(StringIO()):
                        r.run(max_rounds=1)
                outcomes.append(deepcopy((r.actor.state_dict(),r.critic.state_dict(),r.optim_steps,torch.get_rng_state())))
                self.assertEqual(actual_rates,[3e-4])
                self.assertEqual(r.rounds[0]['fit_actor_lr'],r.rounds[0]['actor_lr_used'])
                self.assertEqual(r.rounds[0]['actor_lr_source_steps'],8)
                self.assertIn('critic_postfit_target_variance',r.rounds[0])
                self.assertIn('direction_train_fisher_energy',r.rounds[0])
            finally:
                r.sampler.close()
        self.assertTrue(same(outcomes[0],outcomes[1]))
        self.assertTrue(same(outcomes[1],outcomes[2]))

    def test_linear_public_resume_matches_and_extension_keeps_clock(self):
        path = self.root/'linear.json'
        path.write_text(json.dumps(dict(direction_check=False,return_check=False,save_fit_diagnostics=False,
            source_eval_episodes=1,evaluation_episodes=1,hidden=8,heads=1,relation_layers=0,
            direction_steps=1,actor_fit_steps=1,critic_steps=1,actor_lr_schedule='linear',
            actor_lr_final=3e-5,actor_lr_decay_steps=256,actor_fit_selection='best_train_kl')),encoding='utf-8')
        def fresh(name):
            return ['--experiments','N-C','--profile','smoke','--methods','AN','--device','cpu',
                    '--budget','256','--config',str(path),'--output',str(self.root/name)]
        def checkpoint(name):
            return self.root/name/'N-C_source_learning/full/AN_s40/checkpoints/final.pt'
        self.assertEqual(self.invoke(fresh('reference')),0)
        original = Runner.update
        def interrupt(r):
            if r.round == 1:
                raise RuntimeError('interrupt at a committed boundary')
            return original(r)
        with patch.object(Runner,'update',interrupt):
            self.assertEqual(self.invoke(fresh('interrupted')),2)
        self.assertEqual(self.invoke(['--resume',str(self.root/'interrupted')]),0)
        ref, resumed = load_pt(checkpoint('reference')),load_pt(checkpoint('interrupted'))
        for key in ('actor','critic','actor_optimizer','critic_optimizer','costs','counts','optim_steps',
                    'torch_rng','actor_lr_schedule_state','initial_model_sha256'):
            self.assertTrue(same(ref[key],resumed[key]),key)
        self.assertEqual([r['actor_lr_used'] for r in ref['rounds']],
                         [r['actor_lr_used'] for r in resumed['rounds']])
        before = checkpoint('reference').read_bytes()
        self.assertEqual(self.invoke(['--resume',str(self.root/'reference'),'--budget','512',
                                      '--output',str(self.root/'extended')]),0)
        extended = load_pt(checkpoint('extended'))
        self.assertEqual(extended['actor_lr_schedule_state'], ref['actor_lr_schedule_state'])
        self.assertEqual(checkpoint('reference').read_bytes(),before)
        self.assertAlmostEqual(extended['rounds'][-1]['actor_lr_used'],3e-5,places=15)

    def test_first_sampling_interruption_does_not_reset_s0(self):
        c=settings()
        c.update(actor_lr_schedule='linear',actor_lr_final=3e-5,actor_lr_decay_steps=4096)
        r=Runner(self.root/'first',c,'no_checks',40)
        r.sampler.costs['source_monitor']=8
        def interrupt(*args,**kwargs):
            r.sampler.costs['train'] += 16
            r.log('sampling',used=r.sampler.used)
            raise RuntimeError('interrupt within the first sample')
        try:
            with patch.object(r.sampler,'collect',side_effect=interrupt):
                with self.assertRaises(RuntimeError):r.update()
        finally:r.sampler.close()
        resumed=Runner(self.root/'first',c,'no_checks',40,resume=True)
        try:
            self.assertEqual(resumed.actor_lr_schedule_state['start_source_steps'],8)
            self.assertEqual(resumed.sampler.used,24)
            actual=rate_at(resumed.actor_lr_schedule_state,resumed.sampler.used)['actor_lr_used']
            self.assertAlmostEqual(actual,3e-4+(3e-5-3e-4)*16/(4096-8),places=15)
        finally:resumed.sampler.close()

    def test_historical_mappo_frozen_public_evaluation(self):
        root=Path(__file__).resolve().parents[1]
        checkpoint=root/'results/evidence/navigation/navigation_ep400_history8_2m_s40/mappo/N-C_source_learning/full/MAPPO-E_s40/checkpoints/final.pt'
        if not checkpoint.is_file():self.skipTest('Historical checkpoint is local-only')
        digest=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        config=root/'configs/navigation/batch_lr_2m/mappo_reference.json'
        with patch.object(Runner,'update',side_effect=AssertionError('Evaluation must not train')):
            code=self.invoke(['--experiments','N-T','--profile','pilot','--methods','MAPPO-E',
                '--config',str(config),'--checkpoints',str(checkpoint),'--device','cpu',
                '--evaluation-episodes','1','--evaluation-horizon','3','--output',str(self.root/'frozen')])
        self.assertEqual(code,0)
        audits=list((self.root/'frozen').rglob('evaluation_audit.json'))
        self.assertEqual(len(audits),1)
        audit=json.loads(audits[0].read_text(encoding='utf-8'))
        self.assertEqual((audit['training_environment_steps'],audit['optimizer_updates']),(0,0))
        self.assertEqual(audit['target_sizes'],[4])
        self.assertEqual(audit['evaluation_seed'],20262000)
        self.assertEqual(audit['completed_episodes'],1)
        self.assertLessEqual(audit['evaluation_steps'],3)
        self.assertEqual(audit['actor_state_sha256'],audit['actor_state_sha256_after'])
        self.assertEqual(hashlib.sha256(checkpoint.read_bytes()).hexdigest(),digest)

    def test_frozen_evaluation_failure_preserves_cost_and_retry(self):
        from manifold_project.experiments.applications import evaluation
        c=settings()
        c['current_config']['methods']=['AN']
        r=Runner(self.root/'source',c,'no_checks',40)
        try:r.checkpoint()
        finally:r.sampler.close()
        checkpoint=self.root/'source/checkpoints/final.pt'
        config=dict(device='cpu',evaluation_episodes=2,evaluation_horizon=3,
                    evaluation_seed=90919000,target_sizes=[4])
        output=self.root/'evaluation'
        original=evaluation.rollout_episode
        calls=0
        def interrupt(*args,**kwargs):
            nonlocal calls
            calls += 1
            if calls == 2:raise RuntimeError('interrupted evaluation')
            return original(*args,**kwargs)
        with patch.object(evaluation,'rollout_episode',side_effect=interrupt):
            with self.assertRaises(RuntimeError):
                evaluation.evaluate_checkpoint(checkpoint,config,output)
        first=json.loads((output/'evaluation_audit_001.json').read_text(encoding='utf-8'))
        self.assertEqual((first['status'],first['completed_episodes']),('failed',1))
        self.assertEqual((first['evaluation_steps'],first['pending_episode_reserved_steps']),(3,3))
        ledger=(output/'evaluation_ledger_001.jsonl').read_bytes()
        evaluation.evaluate_checkpoint(checkpoint,config,output)
        self.assertEqual((output/'evaluation_ledger_001.jsonl').read_bytes(),ledger)
        self.assertEqual(json.loads((output/'evaluation_audit_002.json').read_text(encoding='utf-8'))['status'],'completed')


if __name__ == '__main__':
    unittest.main()
