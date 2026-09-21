"""Adapt the pinned author implementation to our collection/evaluation protocol.

PPO, models, ValueNorm and GAE return computation live in vendor/mappo. Legacy
windows occupy independent lanes. First-arrival batches pack real transitions,
with terminal masks at internal resets and the actual final state at a cutoff.
"""
import math
from copy import copy
import numpy as np
import torch
from torch import nn
from gymnasium.spaces import Box, Discrete

from ..configs import continuing_task, arrival_task
from ..vendor.mappo.onpolicy.config import get_config
from ..vendor.mappo.onpolicy.algorithms.r_mappo.algorithm.rMAPPOPolicy import R_MAPPOPolicy
from ..vendor.mappo.onpolicy.algorithms.r_mappo.algorithm.r_actor_critic import R_Actor
from ..vendor.mappo.onpolicy.algorithms.r_mappo.r_mappo import R_MAPPO
from ..vendor.mappo.onpolicy.utils.shared_buffer import SharedReplayBuffer

COMMIT = 'de66d7a4b23fac2513f56f96f73b3f5cb96695ac'
ACTOR_KINDS = {name: f'author_{name}_v1' for name in ('mappo', 'ippo')}


def enabled(config, condition):
    # Saved configurations without this field belong to the old local backend.
    return condition in ACTOR_KINDS and config.get(f'{condition}_backend', 'local') == 'author'


def arguments(c, condition='mappo'):
    if condition not in ACTOR_KINDS:
        raise ValueError(f'不支持的作者 PPO 条件: {condition}')
    args = get_config().parse_args([])
    args.algorithm_name = condition
    args.use_centralized_V = condition == 'mappo'
    args.env_name = 'MPE2_' + c.get('task_mode', 'finite_horizon')
    args.use_recurrent_policy = args.use_naive_recurrent_policy = False
    args.hidden_size = c['hidden']
    # Saved author checkpoints predating this option used the upstream ReLU default.
    activation = c.get('ppo_activation', 'relu')
    if activation not in ('relu', 'tanh'):
        raise ValueError('ppo_activation 只支持 relu / tanh')
    args.use_ReLU = activation == 'relu'
    args.episode_length = c['horizon']
    # Buffer lanes represent independent windows, collected sequentially here.
    args.n_rollout_threads = c['train_episodes']
    args.num_mini_batch = math.ceil(c['train_episodes']/c['minibatch_episodes'])
    samples = c['train_episodes']*c['horizon']*c['n_agents']
    if samples % args.num_mini_batch:
        raise ValueError('作者 PPO 要求总 agent 样本数可被小批次数整除，避免丢弃样本')
    if c['ppo_label'] != 'gae' or not c['ppo_normalize_advantage']:
        raise ValueError(f'作者 PPO 保留 GAE 和优势标准化；标签对齐实验请显式使用 {condition}_backend=local')
    args.lr, args.critic_lr = c['actor_lr'], c['critic_lr']
    args.ppo_epoch, args.clip_param = c['ppo_epochs'], c['ppo_clip']
    args.gamma, args.gae_lambda = c['gamma'], c['gae_lambda']
    args.entropy_coef, args.value_loss_coef = c['entropy_coef'], c['value_coef']
    args.max_grad_norm = c['grad_norm']
    # Legacy MAPPO-only keys remain readable in already saved checkpoints.
    args.use_valuenorm = c.get('ppo_value_normalization', c.get('mappo_value_normalization', True))
    args.use_clipped_value_loss = c.get('ppo_value_clipping', c.get('mappo_value_clipping', True))
    args.use_huber_loss = c.get('ppo_huber_loss', c.get('mappo_huber_loss', True))
    # Explicit pre-reset final values + terminal-only masks already encode the
    # continuing objective. Upstream bad_masks=0 would discard the final TD error.
    args.use_proper_time_limits = False
    return args


def observation_space(size):
    return Box(-np.inf, np.inf, shape=(size,), dtype=np.float32)


class AuthorActor(nn.Module):
    """Expose the author's categorical probabilities to the common evaluator.

No probability floor or extra exploration mixture is added. The same R_Actor
instance is optimized through R_MAPPOPolicy. Flattening only adapts batch axes.
    """
    def __init__(self, input_dim, config, network=None, condition='mappo'):
        super().__init__()
        self.network = network if network is not None else R_Actor(
            arguments(config, condition), observation_space(input_dim), Discrete(5))

    def forward(self, x):
        parameter = next(self.parameters())
        x = x.to(device=parameter.device, dtype=parameter.dtype)
        shape = x.shape[:-1]
        features = self.network.base(x.reshape(-1, x.shape[-1]))
        return self.network.act.get_probs(features).reshape(*shape, 5)


class AuthorPPO:
    def __init__(self, input_dim, state_dim, config, condition='mappo'):
        self.c, self.args = config, arguments(config, condition)
        self.condition, self.local = condition, condition == 'ippo'
        self.actor_kind = ACTOR_KINDS[condition]
        self.obs_space = observation_space(input_dim)
        self.state_space = observation_space(input_dim if self.local else state_dim)
        self.action_space = Discrete(5)
        device = torch.device(config['device'])
        self.policy = R_MAPPOPolicy(self.args, self.obs_space, self.state_space, self.action_space, device)
        self.trainer = R_MAPPO(self.args, self.policy, device)
        self.actor = AuthorActor(input_dim, config, self.policy.actor, condition)

    def description(self):
        keys = ('algorithm_name', 'hidden_size', 'layer_N', 'use_ReLU', 'use_feature_normalization',
                'use_orthogonal', 'gain', 'use_recurrent_policy', 'use_naive_recurrent_policy', 'use_centralized_V',
                'lr', 'critic_lr', 'opti_eps', 'weight_decay', 'ppo_epoch', 'num_mini_batch',
                'clip_param', 'gamma', 'gae_lambda', 'use_gae', 'use_valuenorm', 'use_popart',
                'use_clipped_value_loss', 'use_huber_loss', 'huber_delta', 'entropy_coef',
                'value_loss_coef', 'max_grad_norm', 'use_linear_lr_decay', 'use_proper_time_limits')
        return dict(repository='https://github.com/marlbenchmark/on-policy', commit=COMMIT,
                    adapter=self.actor_kind, arguments={k:getattr(self.args, k) for k in keys},
                    critic_input='local_history' if self.local else 'central_state',
                    timeout=('packed_real_transitions; success/deadline_terminal; collection_cutoff_bootstrap'
                             if arrival_task(self.c) else
                             'pre_reset_final_value; independent_window_lanes; terminal_only_mask'),
                    actor_probability='author_categorical_without_beta_mixture')

    def normalizer_state(self):
        norm = self.trainer.value_normalizer
        if norm is None:
            return None
        # The upstream .to(device) in ValueNorm.__init__ can leave these tensors
        # unregistered on CUDA. Save them explicitly, not via module.state_dict().
        return {k:getattr(norm, k).detach().clone() for k in
                ('running_mean', 'running_mean_sq', 'debiasing_term')}

    def restore_normalizer(self, saved):
        norm = self.trainer.value_normalizer
        if (norm is None) != (saved is None):
            raise ValueError('检查点与当前 ValueNorm 配置不一致')
        if norm is not None:
            with torch.no_grad():
                for key in ('running_mean', 'running_mean_sq', 'debiasing_term'):
                    getattr(norm, key).copy_(saved[key])

    @torch.no_grad()
    def values(self, state):
        shape = state.shape[:-1]
        flat = state.reshape(-1, state.shape[-1])
        recurrent = flat.new_zeros((len(flat), self.args.recurrent_N, self.args.hidden_size))
        masks = flat.new_ones((len(flat), 1))
        value = self.policy.get_values(flat, recurrent, masks)
        return value.reshape(*shape, 1)

    def critic_inputs(self, batch):
        if self.local:
            # Each agent keeps its own history/value; never average over agents.
            return batch['x'], batch['final_x']
        # Singleton agent axis broadcasts the shared team value in the buffer.
        return batch['state'].unsqueeze(-2), batch['final_state'].unsqueeze(-2)

    @torch.no_grad()
    def make_buffer(self, batch):
        if arrival_task(self.c):
            return self.make_arrival_buffer(batch)
        c = self.c
        terminal, timeout = batch['terminated'], batch['truncated']
        if (terminal[:, :-1] | timeout[:, :-1]).any() or not (terminal[:, -1] | timeout[:, -1]).all():
            raise ValueError('作者 PPO 适配器要求每条 lane 为一个完整采样窗口，内部不得 reset')
        buffer = SharedReplayBuffer(self.args, c['n_agents'], self.obs_space,
                                    self.state_space, self.action_space)
        to_numpy = lambda x: x.detach().cpu().numpy()
        buffer.obs[:-1] = to_numpy(batch['x'].transpose(0, 1))
        buffer.obs[-1] = to_numpy(batch['final_x'])
        inputs, final_inputs = self.critic_inputs(batch)
        buffer.share_obs[:-1] = to_numpy(inputs.transpose(0, 1))
        buffer.share_obs[-1] = to_numpy(final_inputs)
        buffer.actions[:] = to_numpy(batch['actions'].transpose(0, 1))[..., None]
        buffer.rewards[:] = to_numpy(batch['reward'].transpose(0, 1))[..., None, None]
        boundary = terminal if continuing_task(c) else terminal | timeout
        buffer.masks[1:] = to_numpy((~boundary).transpose(0, 1))[..., None, None]
        # Compute frozen values/log probabilities before any optimizer update.
        buffer.value_preds[:-1] = to_numpy(self.values(inputs).transpose(0, 1))
        final = to_numpy(self.values(final_inputs))
        flat_x = batch['x'].reshape(-1, batch['x'].shape[-1])
        flat_a = batch['actions'].reshape(-1, 1)
        recurrent = flat_x.new_zeros((len(flat_x), self.args.recurrent_N, self.args.hidden_size))
        logp, _ = self.policy.actor.evaluate_actions(flat_x, recurrent, flat_a,
                                                   flat_x.new_ones((len(flat_x), 1)))
        buffer.action_log_probs[:] = to_numpy(logp.reshape(*batch['actions'].shape, 1).transpose(0, 1))
        buffer.compute_returns(final, self.trainer.value_normalizer)
        if not np.isfinite(buffer.returns).all():
            raise FloatingPointError('作者 PPO 回报标签非有限')
        return buffer

    @torch.no_grad()
    def make_arrival_buffer(self, batch):
        """Pack real transitions into one lane; upstream PPO sees no padding.

        Interior resets are true task terminals. Only the last transition may
        be a collection cutoff and bootstrap from its pre-reset observation.
        The upstream return, ValueNorm and optimizer implementations are intact.
        """
        valid = batch['valid']
        real = {key: batch[key][valid] for key in
                ('x', 'actions', 'reward', 'terminated', 'truncated')}
        total = len(real['reward'])
        if not total or real['truncated'][:-1].any():
            raise ValueError('Packed train batch must end each interior episode with a task terminal')
        if total*self.c['n_agents'] % self.args.num_mini_batch:
            raise ValueError('Packed real sample count must be divisible by the PPO minibatch count')
        args = copy(self.args)
        args.episode_length, args.n_rollout_threads = total, 1
        buffer = SharedReplayBuffer(args, self.c['n_agents'], self.obs_space,
                                    self.state_space, self.action_space)
        to_numpy = lambda x: x.detach().cpu().numpy()
        inputs, final_inputs = self.critic_inputs(batch)
        inputs = inputs[valid]
        last = int(torch.nonzero(batch['lengths'] > 0)[-1, 0])
        buffer.obs[:-1, 0] = to_numpy(real['x'])
        buffer.obs[-1, 0] = to_numpy(batch['final_x'][last])
        buffer.share_obs[:-1, 0] = to_numpy(inputs)
        buffer.share_obs[-1, 0] = to_numpy(final_inputs[last])
        buffer.actions[:, 0] = to_numpy(real['actions'])[..., None]
        buffer.rewards[:, 0] = to_numpy(real['reward'])[:, None, None]
        buffer.masks[1:, 0] = to_numpy(~real['terminated'])[:, None, None]
        buffer.value_preds[:-1, 0] = to_numpy(self.values(inputs))
        final = to_numpy(self.values(final_inputs[last:last+1]))
        flat_x = real['x'].reshape(-1, real['x'].shape[-1])
        recurrent = flat_x.new_zeros((len(flat_x), self.args.recurrent_N, self.args.hidden_size))
        logp, _ = self.policy.actor.evaluate_actions(flat_x, recurrent,
                    real['actions'].reshape(-1, 1), flat_x.new_ones((len(flat_x), 1)))
        buffer.action_log_probs[:, 0] = to_numpy(logp.reshape(*real['actions'].shape, 1))
        buffer.compute_returns(final, self.trainer.value_normalizer)
        if not np.isfinite(buffer.returns).all():
            raise FloatingPointError('Nonfinite packed PPO returns')
        return buffer

    def update(self, batch):
        self.trainer.prep_rollout()
        buffer = self.make_buffer(batch)
        target = buffer.returns[:-1].copy()
        self.trainer.prep_training()
        stats = self.trainer.train(buffer)
        stats = {k:float(v.detach()) if isinstance(v, torch.Tensor) else float(v) for k,v in stats.items()}
        if not all(math.isfinite(v) for v in stats.values()) or not all(
                torch.isfinite(p).all() for model in (self.policy.actor, self.policy.critic) for p in model.parameters()):
            raise FloatingPointError('作者 PPO 更新产生非有限值')
        self.trainer.prep_rollout()
        with torch.no_grad():
            inputs, _ = self.critic_inputs(batch)
            packed = arrival_task(self.c)
            predicted = (self.values(inputs[batch['valid']]).unsqueeze(1) if packed else
                         self.values(inputs).transpose(0, 1)).cpu().numpy()
            if self.trainer.value_normalizer is not None:
                predicted = self.trainer.value_normalizer.denormalize(predicted)
            error = target-predicted
            variance = float(np.var(target))
            mse = float(np.mean(np.square(error)))
            x = batch['x'][batch['valid']] if packed else batch['x']
            p = self.actor(x)
            old = batch['mu'][batch['valid']] if packed else batch['mu']
            tiny = torch.finfo(p.dtype).tiny
            kl = float(torch.sum(old*(old.clamp_min(tiny).log()-p.clamp_min(tiny).log()), -1).mean())
            entropy = float(-torch.sum(p*p.clamp_min(tiny).log(), -1).mean())
            chosen = (batch['actions'][batch['valid']] if packed else batch['actions']).unsqueeze(-1)
            ratio = p.gather(-1, chosen)/old.gather(-1, chosen)
            clip_fraction = float(((ratio-1).abs() > self.c['ppo_clip']).float().mean())
        starts = np.r_[0, np.cumsum(batch['lengths'].cpu().numpy())[:-1]] if packed else [0]
        return dict(author_stats=stats, critic_mse=mse,
                    explained_variance=1-float(np.var(error))/variance if variance > 1e-12 else None,
                    train_value_target=float(target[starts].mean()), ppo_kl=kl, policy_entropy=entropy,
                    clip_fraction=clip_fraction,
                    updates=self.args.ppo_epoch*self.args.num_mini_batch)
