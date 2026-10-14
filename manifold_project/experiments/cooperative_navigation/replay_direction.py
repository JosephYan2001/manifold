"""Verify a saved direction snapshot offline; no environment sampling or training."""
import argparse
from pathlib import Path


def replay(path, device='cpu'):
    import torch
    from .training.storage import load_pt
    from .models import Actor,Critic,Direction
    from .training.ppo import value_targets,batch_weighted,valid_values
    from .training.diagnostics import direction_metrics,categorical_kl
    s=load_pt(path)
    c=dict(s['config'],device=device)
    torch.set_num_threads(c['threads'])
    old=Actor(s['input_dim'],c).to(device);old.load_state_dict(s['actor']);old.eval()
    critic=Critic(s['state_dim'],c).to(device);critic.load_state_dict(s['critic']);critic.eval()
    direction=Direction(s['input_dim'],c).to(device);direction.load_state_dict(s['direction']);direction.eval()
    def check(a,b):
        torch.testing.assert_close(torch.as_tensor(a),torch.as_tensor(b),rtol=2e-5,atol=2e-6)
    with torch.no_grad():
        for key,reference in [('batch','train_metrics'),('validation','validation_metrics')]:
            b={k:v.to(device) for k,v in s[key].items()}
            check(valid_values(old(b['x']),b),valid_values(b['mu'],b))
            values,targets,gae=value_targets(b,critic,c)
            advantage=gae if c['direction_label']=='gae' else targets-values
            advantage=advantage.unsqueeze(-1).expand_as(b['actions'])
            check(advantage,b['advantage']);check(targets,b['value_target'])
            metrics=direction_metrics(direction,b,c)
            for k in ('B_label','D_label','fisher_energy','sampled_energy'):
                check(metrics[k],s[reference][k])
            if s['eta'] is not None:
                target=torch.softmax(b['mu'].log()+s['eta']*direction(b['x']),-1)
                check(valid_values(target,b),valid_values(b['target'],b))
        if s['candidate'] is not None:
            candidate=Actor(s['input_dim'],c).to(device);candidate.load_state_dict(s['candidate']);candidate.eval()
            b={k:v.to(device) for k,v in s['batch'].items()}
            actual=float(batch_weighted(categorical_kl(b['target'],candidate(b['x'])),b,c))
            check(actual,s['info']['fit_kl_after_fp64'])
    return dict(snapshot=str(path),round=s['round'],verified=True,environment_steps=0)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--snapshot',type=Path,required=True)
    p.add_argument('--device',default='cpu')
    p.add_argument('--allow-duplicate-openmp',action='store_true')
    args=p.parse_args(argv)
    from ..common.runtime import configure_runtime
    configure_runtime(args.allow_duplicate_openmp)
    import json
    print(json.dumps(replay(args.snapshot,args.device),indent=2))


if __name__=='__main__':
    main()
