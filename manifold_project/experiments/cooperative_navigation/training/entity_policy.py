"""Network interface only; PPO/GAE/ValueNorm remain in the archived author code."""
import torch
from ..models import Actor, Critic, LocalCritic


class EntityPolicy:
    def __init__(self, args, input_dim, state_dim, config, local=False):
        self.device = torch.device(config['device'])
        self.actor = Actor(input_dim, config).to(self.device)
        self.critic = (LocalCritic(input_dim, config) if local else Critic(state_dim, config)).to(self.device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=args.lr,
                                                eps=args.opti_eps, weight_decay=args.weight_decay)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=args.critic_lr,
                                                 eps=args.opti_eps, weight_decay=args.weight_decay)

    def get_values(self, cent_obs, rnn_states=None, masks=None):
        x = torch.as_tensor(cent_obs, device=self.device, dtype=torch.float32)
        return self.critic(x).unsqueeze(-1)

    def evaluate_actions(self, cent_obs, obs, rnn_states_actor, rnn_states_critic,
                         action, masks, available_actions=None, active_masks=None):
        logp, entropy = self.actor.evaluate_actions(obs, rnn_states_actor, action, masks,
                                                   available_actions, active_masks)
        return self.get_values(cent_obs), logp, entropy
