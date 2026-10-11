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


def expected_counts(config=None):
    """Audit retained, unconfirmed schedule numbers; never launch training."""
    from .timing import ClockConfig, default_protocol
    config=default_protocol() if config is None else config
    clocks=ClockConfig.from_config(config)
    episodes=config["schedule"]["total_episodes_per_seed"]
    events=actors=ppo_batches=pending=0
    previous=None
    for number in range(1,episodes+1):
        current=phase(number)
        block=(current.stage,current.cycle,current.layer)
        if current.layer=="lower" and block==previous:
            for _ in range(clocks.lower_steps//config["sac"]["learn_every_routing_frames"]):
                events+=1
                if current.actor_enabled and events%config["sac"]["actor_every_events"]==0:
                    actors+=1
        if current.layer=="upper":
            pending+=1
            if pending==config["ppo"]["batch_episodes"]:
                ppo_batches+=1; pending=0
        previous=block
    return dict(episodes=episodes, physics_steps=episodes*clocks.physical_steps,
                sac_events=events,critic_fits=events*config["sac"]["critic_fits_per_event"],
                sac_actor_updates=actors,ppo_batches=ppo_batches)
