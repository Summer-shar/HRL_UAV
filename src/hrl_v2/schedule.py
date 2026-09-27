"""Fixed 600-episode schedule, independent of development performance."""
from dataclasses import dataclass


@dataclass(frozen=True)
class Phase:
    global_episode: int
    stage: int
    cycle: int
    block_episode: int
    layer: str
    alpha: float

    @property
    def actor_enabled(self):
        return self.layer == "lower" and self.block_episode > 2


def phase(episode):
    if not 1 <= episode <= 600:
        raise ValueError("Fixed budget is 600 episodes")
    if episode <= 300:
        return Phase(episode,1,0,episode,"lower",.005 if episode <= 100 else .0025 if episode <= 200 else .0005)
    if episode <= 400:
        return Phase(episode,2,0,episode-300,"upper",.0005)
    cycle, position = divmod(episode-401,20)
    upper = position < 10
    return Phase(episode,3,2*(cycle+1)-int(upper),position%10+1,
                 "upper" if upper else "lower", .0005 if cycle < 5 else 0.)


def expected_counts():
    return dict(episodes=600, physics_steps=5400000, sac_events=35010,
                critic_fits=70020, sac_actor_updates=17010, ppo_batches=40)
