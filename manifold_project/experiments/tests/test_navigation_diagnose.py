"""Small CPU checks for frozen-checkpoint diagnostic semantics."""
import csv
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO

from manifold_project.experiments.common.runtime import configure_runtime
configure_runtime(True)
import torch
from manifold_project.experiments.cooperative_navigation.diagnose import main, match_mixture
from manifold_project.experiments.cooperative_navigation.training.runner import Runner
from manifold_project.experiments.cooperative_navigation.training.storage import load_pt, save_pt
from manifold_project.experiments.cooperative_navigation.current_protocol import runner_config
from manifold_project.experiments.common.config import load_config


class DiagnosticTests(unittest.TestCase):
    def test_mixture_hits_kl_and_rejects_unreachable(self):
        mu = torch.tensor([[.2,.3,.5]],dtype=torch.float64)
        proposal = torch.tensor([[.5,.3,.2]],dtype=torch.float64)
        alpha, actual = match_mixture(mu,proposal,lambda v:v.mean(),1e-4)
        self.assertTrue(0 < alpha < 1)
        self.assertAlmostEqual(actual,1e-4,places=10)
        with self.assertRaises(ValueError):
            match_mixture(mu,mu,lambda v:v.mean(),1e-4)

    def test_both_modes_leave_source_frozen(self):
        workspace = Path(__file__).resolve().parents[3]
        with tempfile.TemporaryDirectory(prefix='diagnostic_test_',dir=workspace) as tmp:
            root=Path(tmp)
            # Use the production configuration translator, with tiny networks/windows.
            public=load_config('navigation','smoke')
            public.update(hidden=8,heads=1,relation_layers=0,history=1,horizon=8,
                          evaluation_horizon=8,device='cpu',batch_episodes=2,
                          direction_steps=1,actor_fit_steps=1,critic_steps=1,
                          minibatch_size=32,seed=40,budget=256)
            config=runner_config(public)
            runner=Runner(root/'source',config,'no_checks',40)
            runner.checkpoint();runner.sampler.close()
            source=root/'source/checkpoints/final.pt'
            state=load_pt(source);state['current_method']='AN';save_pt(source,state)
            digest=hashlib.sha256(source.read_bytes()).hexdigest()
            shared=['--checkpoint',str(source),'--device','cpu','--seeds','40',
                    '--validation-episodes','2','--evaluation-episodes','2',
                    '--direction-epochs','1','2','--fit-epochs','2',
                    '--actor-lrs','0.0003','0.0001','--allow-duplicate-openmp']
            with redirect_stdout(StringIO()):
                main(shared+['--mode','direction','--output',str(root/'dry'),'--dry-run'])
            self.assertFalse((root/'dry').exists())
            for mode in ['direction','fit']:
                out=root/mode
                with redirect_stdout(StringIO()):
                    main(shared+['--mode',mode,'--output',str(out),'--fit-trace-every-updates','1'])
                manifest=json.loads((out/'manifest.json').read_text())
                self.assertEqual(manifest['status'],'completed')
                self.assertTrue(manifest['source_actor_critic_frozen'])
                with (out/'diagnostics.csv').open(encoding='utf-8-sig') as f:
                    rows=list(csv.DictReader(f))
                self.assertEqual(len(rows),4)
                with (out/'fit_trace.csv').open(encoding='utf-8-sig') as f:
                    trace=list(csv.DictReader(f))
                self.assertTrue(any(r['scope']=='post_optimizer_step' for r in trace))
                with (out/'direction_scores.csv').open(encoding='utf-8-sig') as f:
                    scores=list(csv.DictReader(f))
                for r in scores:
                    self.assertAlmostEqual(float(r['B_label']),2*float(r['D_label'])-float(r['fisher_energy']))
                if mode=='direction':
                    self.assertEqual(len({r['common_kl'] for r in rows}),1)
                    for r in rows:
                        self.assertLess(abs(float(r['matched_actual_kl'])-float(r['common_kl'])),1e-7)
                else:
                    for r in rows:
                        if r['candidate'].endswith('_best'):
                            self.assertLessEqual(float(r['selected_train_target_kl']),float(r['initial_train_target_kl']))
                            self.assertLessEqual(float(r['selected_train_target_kl']),float(r['final_train_target_kl']))
                with self.assertRaises(FileExistsError):
                    main(shared+['--mode',mode,'--output',str(out)])
            self.assertEqual(hashlib.sha256(source.read_bytes()).hexdigest(),digest)
            with redirect_stdout(StringIO()):
                main(shared+['--mode','direction','--output',str(root/'matrix'),
                             '--sample-sizes','1','2','--direction-updates','2','4','--init-seeds','70'])
            for n in [1,2]:
                leaf=root/'matrix'/f'n{n}_init70'
                with (leaf/'direction_scores.csv').open(encoding='utf-8-sig') as f:
                    scores=list(csv.DictReader(f))
                self.assertEqual({int(r['optimizer_updates']) for r in scores},{2,4})
                self.assertEqual({int(r['init_seed']) for r in scores},{70})
                meta=json.loads((leaf/'manifest.json').read_text())
                self.assertEqual(meta['diagnostic_seeds'],[40])
            self.assertEqual(hashlib.sha256(source.read_bytes()).hexdigest(),digest)


if __name__=='__main__':
    unittest.main()
