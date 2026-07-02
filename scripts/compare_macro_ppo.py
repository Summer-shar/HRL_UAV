import numpy as np

try:
    from ._bootstrap import bootstrap
except ImportError:
    from _bootstrap import bootstrap

bootstrap()

from hrl_uav.agents.ppo_agent import Actor as PPOActor
from hrl_uav.agents.ppo_agent import Critic as PPOCritic
from hrl_uav.agents.ppo_agent import GAMMA as PPO_GAMMA


def choose_macro_actions(ppo_actors, local_obs_batch, deterministic=False):
    actions = []
    log_probs = []

    for actor_idx, actor in enumerate(ppo_actors):
        local_obs = local_obs_batch[actor_idx]
        if deterministic:
            action = actor.choose_action_deterministic(local_obs)
            actions.append(np.asarray(action, dtype=np.float32))
        else:
            action, log_prob = actor.choose_action(local_obs)
            actions.append(np.asarray(action, dtype=np.float32))
            log_probs.append(np.asarray(log_prob, dtype=np.float32).reshape(1))

    actions = np.asarray(actions, dtype=np.float32)
    if deterministic:
        return actions

    log_probs = np.asarray(log_probs, dtype=np.float32).reshape(len(ppo_actors), 1)
    return actions, log_probs


class MacroPPOTrainer:
    def __init__(
        self,
        local_state_dim,
        global_state_dim,
        num_agents,
        action_dim=2,
        macro_steps_per_episode=30,
        update_episode_window=20,
    ):
        self.num_agents = int(num_agents)
        self.actors = [
            PPOActor(local_state_dim, action_dim)
            for _ in range(self.num_agents)
        ]
        self.critic = PPOCritic(global_state_dim, self.num_agents)
        self.update_threshold = max(
            int(macro_steps_per_episode) * int(update_episode_window),
            1,
        )

        self.buffer_local_s = []
        self.buffer_global_s = []
        self.buffer_a = []
        self.buffer_a_logp = []
        self.buffer_discounted_r = []

        self.episode_local_s = []
        self.episode_global_s = []
        self.episode_a = []
        self.episode_r = []
        self.episode_a_logp = []

        self.actor_losses = []
        self.critic_losses = []

    def start_episode(self):
        self.episode_local_s = []
        self.episode_global_s = []
        self.episode_a = []
        self.episode_r = []
        self.episode_a_logp = []

    def act(self, local_obs_batch, deterministic=False):
        return choose_macro_actions(
            self.actors,
            local_obs_batch,
            deterministic=deterministic,
        )

    def macro_policy(self, env, local_obs_batch, deterministic=False):
        if deterministic:
            return self.act(local_obs_batch, deterministic=True)
        action, _ = self.act(local_obs_batch, deterministic=False)
        return action

    def record_step(
        self,
        local_obs_batch,
        global_state,
        action,
        action_logprob,
        reward_per_uav,
    ):
        self.episode_local_s.append(np.asarray(local_obs_batch, dtype=np.float32).copy())
        self.episode_global_s.append(np.asarray(global_state, dtype=np.float32).copy())
        self.episode_a.append(np.asarray(action, dtype=np.float32).copy())
        self.episode_r.append(np.asarray(reward_per_uav, dtype=np.float32).copy())
        self.episode_a_logp.append(np.asarray(action_logprob, dtype=np.float32).copy())

    def finish_episode(self, next_global_state, force_update=False):
        if self.episode_global_s:
            v_next = self.critic.get_v(next_global_state).detach().cpu().numpy()
            running_v = v_next.reshape(self.num_agents).astype(np.float32)

            ep_discounted_r = []
            for reward in self.episode_r[::-1]:
                running_v = reward + PPO_GAMMA * running_v
                ep_discounted_r.append(running_v.copy())
            ep_discounted_r.reverse()

            self.buffer_local_s.extend(self.episode_local_s)
            self.buffer_global_s.extend(self.episode_global_s)
            self.buffer_a.extend(self.episode_a)
            self.buffer_a_logp.extend(self.episode_a_logp)
            self.buffer_discounted_r.extend(ep_discounted_r)

        self.start_episode()

        if len(self.buffer_global_s) >= self.update_threshold or force_update:
            self.update()

    def update(self):
        if not self.buffer_global_s:
            return

        discounted_r = np.asarray(self.buffer_discounted_r, dtype=np.float32)
        if discounted_r.shape[0] > 1:
            discounted_r = (
                discounted_r - discounted_r.mean(axis=0, keepdims=True)
            ) / (discounted_r.std(axis=0, keepdims=True) + 1e-8)

        bs_global = np.vstack(self.buffer_global_s).astype(np.float32)
        br = discounted_r.astype(np.float32)
        advantage, critic_loss = self.critic.learn(bs_global, br)

        bs_local = np.asarray(self.buffer_local_s, dtype=np.float32)
        ba = np.asarray(self.buffer_a, dtype=np.float32)
        bap = np.asarray(self.buffer_a_logp, dtype=np.float32)

        actor_losses = []
        for agent_idx, actor in enumerate(self.actors):
            actor_bs = bs_local[:, agent_idx, :].astype(np.float32)
            actor_ba = ba[:, agent_idx, :].astype(np.float32)
            actor_bap = bap[:, agent_idx, :].astype(np.float32)
            actor_advantage = advantage[:, agent_idx:agent_idx + 1].astype(np.float32)

            actor_loss = actor.learn(
                actor_bs,
                actor_ba,
                actor_advantage,
                actor_bap,
            )
            actor.update_oldpi()
            actor_losses.append(actor_loss)

        self.actor_losses.append(float(np.mean(actor_losses)) if actor_losses else 0.0)
        self.critic_losses.append(float(critic_loss))

        self.buffer_local_s.clear()
        self.buffer_global_s.clear()
        self.buffer_a.clear()
        self.buffer_a_logp.clear()
        self.buffer_discounted_r.clear()
