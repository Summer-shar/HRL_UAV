"""Isolated upper-horizon pilot. Production V4 sources are imported, never edited.

The physical cost and SAC time objective remain V4. Only the independently
declared upper discount changes across arms. Full agent serialization is reused
under a new pilot protocol; this is explicit transfer, not V4 continuation.
"""
import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT))
from hrl_uav.industrial_hotspot_evaluation import _load_agent, frozen_fingerprint, _hash_value, atomic_json, file_sha
from hrl_uav.industrial_hotspot_experiment import source_hashes, make_agent
from hrl_uav.industrial_experiment import runtime_setup, counters, network_fingerprint
from hrl_uav.industrial_hotspot_mdp import HotspotIndustrialMDP
from hrl_uav.industrial_hotspot_scene import generate_realization
from hrl_uav.industrial_deployment_training import DeploymentLearningControllers, DeploymentTrainingSchedule
from hrl_uav.industrial_rollout import IndustrialRollout, Transition
from hrl_uav.industrial_deployment_checkpoint import (make_checkpoint, restore_checkpoint,
    save_checkpoint, load_checkpoint, capture_rng, _same)
from hrl_uav.industrial_hotspot_gate import digest, pooled_metrics, paired_rows

SOURCE_RUN=ROOT/'industrial_hotspot_records/load24725_seed1_20260923/training'
PARENT=SOURCE_RUN/'checkpoints/candidate-joint-0600.pt'
SCHEMA='industrial_upper_horizon_pilot_v1'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def now():
    return datetime.now(timezone.utc).isoformat()


def hashed(value):
    h=hashlib.sha256()
    _hash_value(h,value)
    return h.hexdigest()


def append(path,value):
    with Path(path).open('a',encoding='utf-8') as f:
        f.write(json.dumps(value,ensure_ascii=False,allow_nan=False)+'\n')
        f.flush()


def make_protocol(arm):
    if arm not in ('control','long150'):
        raise ValueError('Unknown pilot arm')
    p=read(SOURCE_RUN/'protocol.json')
    # New, explicit experiment contract: no inherited gate or false V4 identity.
    return dict(schema=SCHEMA,purpose='pilot',arm=arm,training_seed=1,
        upper_half_life_s=None if arm=='control' else 150.,
        upper_objective='discounted physical cost with independent upper physical-time discount',
        lower_objective='unchanged V4 discounted physical cost plus SAC entropy',
        parent=dict(path=str(PARENT),sha256=file_sha(PARENT)),
        scene_path=str(SOURCE_RUN/'scene.json'),scene_sha256=digest(read(SOURCE_RUN/'scene.json')),
        learning=copy.deepcopy(p['learning']),runtime=dict(device='cpu',torch_threads=1,max_wall_seconds_per_job=3600),
        budget=dict(episode_seconds=900,joint_episodes=50,joint_development_check_episodes=[25,50],
                    max_training_physical_steps=450000,max_evaluation_physical_steps=216000),
        randomness=dict(layout_base=610000000,arrival_base=630000000,load_base=650000000,
                        routing_base=640000000,training_stride=1000000),
        development_manifest=copy.deepcopy(p['development_manifest']),
        transfer='Full unchanged SAC learning state; newly seeded PPO networks, optimizers, exploration, guard and rollout.',
        quality_rule=dict(cost_ratio_bad=1.10,delay_ratio_bad=1.10,upload_area_ratio_bad=2.,
                          consecutive_bad_nodes=2,automatic_extension=False),
        initial_motion_diagnostic='record_only; one reference is not an optimum or proof of absent benefit',
        episode_semantics=copy.deepcopy(p['episode_semantics']))


def validate_protocol(p):
    if p.get('schema')!=SCHEMA or p.get('purpose') not in ('pilot','functional'):
        raise ValueError('Explicit pilot protocol required')
    if p['arm'] not in ('control','long150') or p['upper_half_life_s'] != (None if p['arm']=='control' else 150.):
        raise ValueError('Arm/time-horizon mismatch')
    original=read(SOURCE_RUN/'protocol.json')['learning']
    allowed={'ppo_update_macros','ppo_epochs'} if p['purpose']=='functional' else set()
    if any(p['learning'].get(k)!=v for k,v in original.items() if k not in allowed) or p['learning'].keys()!=original.keys():
        raise ValueError('Unregistered second learning change')
    if p['purpose']=='pilot' and (p['budget']['episode_seconds']!=900 or p['budget']['joint_episodes']!=50
            or p['budget']['joint_development_check_episodes']!=[25,50]):
        raise ValueError('Pilot is limited to 50 complete episodes and checks at 25/50')


class PilotMDP(HotspotIndustrialMDP):
    def __init__(self,scene,protocol):
        validate_protocol(protocol)
        super().__init__(scene)
        half=protocol['upper_half_life_s']
        self.upper_gamma_frame=self.gamma_frame if half is None else 2**(-self.config['time']['routing_s']/half)
        self.upper_gamma_step=self.upper_gamma_frame**(1/self.env.frame_steps)
        self.gamma_macro=self.upper_gamma_frame**self.frames_per_macro


class PilotRollout(IndustrialRollout):
    """Original collection order with a separate upper physical-cost discount."""
    def run(self,macros,*,truncate=True,collect_metrics=True):
        if type(macros) is not int or macros<1:
            raise ValueError('Positive whole macro count required')
        high,low=[],[]
        start_step=self.mdp.env.step_index
        self.last_physical_costs=[]
        def finish_micro(next_state,is_truncated=False):
            if self._pending_micro is not None:
                state,action,reward,route=self._pending_micro
                low.append(Transition(state,action,reward,next_state.copy(),self.mdp.gamma_frame,
                    float(not is_truncated),is_truncated,route))
                self._pending_micro=None
        for macro in range(macros):
            self._prime_macro()
            reward=0.
            for frame in range(self.mdp.frames_per_macro):
                state=self.mdp.prepare_service()
                finish_micro(state)
                action=np.asarray(self.micro_controller(self.mdp.controller_information())).ravel().copy()
                self.micro_calls+=1
                result=self.mdp.step_frame(action)
                costs=[row['cost'] for row in result['readings']]
                self.last_physical_costs.extend(costs)
                frame_upper=-sum(self.mdp.upper_gamma_step**i*c for i,c in enumerate(costs))
                reward+=self.mdp.upper_gamma_frame**frame*frame_upper
                self._pending_micro=(state,action,result['reward'],result['route'].copy())
            state,action,logprob=self._macro
            is_truncated=truncate and macro==macros-1
            high.append(Transition(state,action,reward,self.mdp.macro_observation().copy(),self.mdp.gamma_macro,
                float(not is_truncated),is_truncated,logprob=logprob))
            self._macro=None
            if self.on_macro_boundary is not None:
                self.on_macro_boundary(high[-1])
        metrics=self.mdp.metrics() if collect_metrics else None
        self._prime_macro()
        finish_micro(self.mdp.prepare_service(),truncate)
        return dict(macro=high,micro=low,metrics=metrics,physical_steps=self.mdp.env.step_index-start_step,
                    next_boundary_primed=True,next_boundary_physical_steps=0)


def source_identity(protocol):
    sources=source_hashes(ROOT)
    sources[str(Path(__file__).resolve().relative_to(ROOT)).replace('\\','/')]=file_sha(__file__)
    return dict(method='ppo_sac',training_seed=protocol['training_seed'],source_sha256=sources)


def build_fresh(protocol,scene):
    validate_protocol(protocol)
    runtime_setup(protocol,protocol['training_seed'])
    return make_agent(protocol,PilotMDP(scene,protocol),protocol['training_seed'],'ppo_sac')


def sac_state(agent):
    return dict(actor=agent.sac_actor.action_net.state_dict(),critic=agent.sac_critic.critic_v.state_dict(),
        target=agent.sac_critic.target_critic_v.state_dict(),log_alpha=agent.entropy.log_alpha.detach(),
        optimizers=[x.state_dict() for x in (agent.sac_actor.optimizer,agent.sac_critic.optimizer,agent.entropy.optimizer)],
        replay=agent.replay.state_dict(),updates=[agent.sac_updates,agent.sac_critic_gradient_steps,
            agent.sac_target_updates,agent.sac_actor_updates],stage_initial_updates=agent.stage_initial_updates,
        last_sac_stats=copy.deepcopy(agent.last_sac_stats))


def make_initial_agent(protocol):
    validate_protocol(protocol)
    saved,donor,sha,sources=_load_agent(protocol['parent']['path'],ROOT)
    if sha!=protocol['parent']['sha256'] or digest(saved['scene'])!=protocol['scene_sha256']:
        raise ValueError('Transfer source identity mismatch')
    if saved['run_state']['schedule']['frames']%protocol['learning']['sac_update_frames']:
        raise ValueError('SAC inherited update phase is not a macro boundary')
    scene=copy.deepcopy(saved['scene'])
    state=sac_state(donor)
    initial_sac_sha=hashed(state)
    agent=build_fresh(protocol,scene)
    initial_ppo_sha=hashed([agent.macro_actor.state_dict(),agent.macro_critic.state_dict()])
    for dest,old in ((agent.sac_actor.action_net,state['actor']),
                     (agent.sac_critic.critic_v,state['critic']),(agent.sac_critic.target_critic_v,state['target'])):
        dest.load_state_dict(old,strict=True)
    with torch.no_grad():
        agent.entropy.log_alpha.copy_(state['log_alpha'])
    agent.entropy.alpha=agent.entropy.log_alpha.exp()
    for opt,old in zip((agent.sac_actor.optimizer,agent.sac_critic.optimizer,agent.entropy.optimizer),state['optimizers']):
        opt.load_state_dict(copy.deepcopy(old))
    agent.replay.load_state_dict(state['replay'],memory_counter=state['replay']['memory_counter'])
    agent.sac_updates,agent.sac_critic_gradient_steps,agent.sac_target_updates,agent.sac_actor_updates=state['updates']
    agent.stage_initial_updates=state['stage_initial_updates']
    agent.last_sac_stats=state['last_sac_stats']
    agent.set_stage('joint')
    if not _same(sac_state(agent),state) or hashed(sac_state(agent))!=initial_sac_sha:
        raise AssertionError('SAC transfer is not exact')
    if agent.ppo_updates or agent.macro_rollout or any(agent.guard_state().values()):
        raise AssertionError('PPO did not reset completely')
    if agent.ppo_actor_optimizer.state or agent.ppo_critic_optimizer.state:
        raise AssertionError('PPO optimizer was inherited')
    receipt=dict(parent_checkpoint=str(PARENT),parent_sha256=sha,source_v4_strict_restore_verified=True,
        initial_sac_sha256=initial_sac_sha,initial_ppo_sha256=initial_ppo_sha,
        initial_global_rng_sha256=hashed(capture_rng()),base_counters=counters(agent),
        base_sac_target_updates=agent.sac_target_updates,
        global_rng_policy='New seed-1 initialization stream, paired across arms; parent SAC replay RNG is preserved.',
        sac_exact_transfer=True,ppo_fresh=True,upper_gamma_macro=agent._macro_gamma,
        lower_gamma_frame=agent.cfg['gamma_per_resource_frame'],
        original_source_sha256=sources,new_protocol_sha256=digest(protocol))
    return agent,scene,receipt


def saved_payload(agent,protocol,scene,state):
    result=make_checkpoint(agent,protocol,scene,source_identity(protocol),state)
    result['pilot_schema']=SCHEMA
    return result


def load_pilot(path):
    saved=load_checkpoint(path)
    if saved.get('pilot_schema')!=SCHEMA:
        raise ValueError('Not an isolated horizon pilot checkpoint')
    protocol,scene=saved['protocol'],saved['scene']
    validate_protocol(protocol)
    agent=build_fresh(protocol,scene)
    state=restore_checkpoint(agent,saved,protocol=protocol,scene=scene,identity=source_identity(protocol))
    return agent,scene,protocol,state


def snapshot(mdp):
    v=mdp.controller_information().base
    positions=np.asarray(v.positions_m)
    distances=[np.linalg.norm(np.asarray(t.position_m)-positions[t.source]) for t in v.terminals]
    return dict(time_s=mdp.env.time_s,position_m=positions.tolist(),
        terminal_mean_distance_m=float(np.mean(distances)),
        unfinished_counts=np.asarray(v.unfinished_counts).tolist(),
        upload_mbits=float(sum(sum(t.upload_bits) for t in v.terminals)/1e6),
        ready_gcycles=float(sum(t.compute_cycles for t in v.ready_tasks)/1e9),
        transfer_mbits=float(np.asarray(v.transfer_bits).sum()/1e6),
        transfer_future_gcycles=float(np.asarray(v.transfer_cycles).sum()/1e9),
        cpu_gcycles=float(np.asarray(v.compute_cycles).sum()/1e9))


def seeds_for(protocol,index):
    r=protocol['randomness']
    return {name+'_seed':r[name+'_base']+protocol['training_seed']*r['training_stride']+index
            for name in ('arrival','layout','load','routing')}


def reset_scene(mdp,seeds,permutation=None):
    realization=generate_realization(mdp.scene_wrapper,seeds['layout_seed'],seeds['load_seed'],profile_permutation=permutation)
    mdp.reset(seeds['arrival_seed'],seeds['layout_seed'],seeds['load_seed'],realization)
    return realization


def evaluate(checkpoint,output):
    output=Path(output)
    output.mkdir(parents=True,exist_ok=False)
    agent,scene,p,state=load_pilot(checkpoint)
    before=frozen_fingerprint(agent)
    identity=source_identity(p)
    rows=[]
    started=time.perf_counter()
    attempted=0
    try:
        with torch.no_grad():
            for scenario in p['development_manifest']['scenarios']:
                for mode in ('hover','moving'):
                    mdp=PilotMDP(scene,p)
                    real=reset_scene(mdp,scenario,scenario['profile_permutation'])
                    controllers=DeploymentLearningControllers(agent,mdp.encoder,deterministic=True,macro_enabled=mode=='moving')
                    curve=[]
                    discounted_common=discounted_upper=0.
                    for i in range(180):
                        if time.perf_counter()-started>900 or attempted>=108000:
                            raise TimeoutError('Bounded pilot evaluation budget reached')
                        action,_=controllers.macro(mdp.controller_information())
                        movement=mdp.start_macro(action)
                        for _ in range(mdp.frames_per_macro):
                            mdp.prepare_service()
                            start=mdp.env.step_index
                            attempted+=mdp.env.frame_steps
                            result=mdp.step_frame(controllers.micro(mdp.controller_information()))
                            discounted_common+=sum(mdp.gamma_step**(start+j)*r['cost'] for j,r in enumerate(result['readings']))
                            discounted_upper+=sum(mdp.upper_gamma_step**(start+j)*r['cost'] for j,r in enumerate(result['readings']))
                        curve.append(dict(**snapshot(mdp),cost_sum=mdp.env.cost_sum,holding_task_s=mdp.env.holding,
                            energy_j=sum(mdp.env.energy.values()),raw_action=np.asarray(action).tolist(),
                            safety_scale=float(movement['safety_scale'])))
                    metrics=mdp.metrics()
                    scale_bits=max(1.,mdp.env.generated_bits)
                    scale_cycles=max(1.,mdp.env.generated_cycles)
                    if any(abs(v)>1e-5+1e-9*(scale_cycles if k.endswith('cycles') else scale_bits)
                           for k,v in metrics['conservation'].items()):
                        raise AssertionError('Task conservation failed')
                    row=dict(status='complete',technical_valid=True,frozen_verified=True,
                        metrics_frozen_before_priming=True,scenario_id=scenario['scenario_id'],scenario_sha=digest(scenario),
                        realization_sha=real['realization_sha256'],event_sha256=metrics['event_sha256'],
                        canonical_event_sha256=metrics['canonical_event_sha256'],profile_permutation=real['profile_by_region'],
                        mode=mode,metrics=metrics,low_unfinished=metrics['priority'][1]['arrived']-metrics['priority'][1]['completed'],
                        discounted_cost_common=discounted_common,discounted_cost_upper=discounted_upper,curve=curve,
                        upload_area_mbit_s_5s=float(5*sum(x['upload_mbits'] for x in curve)))
                    rows.append(row)
                    atomic_json(output/(scenario['scenario_id']+'_'+mode+'.json'),row)
                    atomic_json(output/'progress.json',dict(status='running',trajectories=len(rows),physical_steps=attempted))
        if before!=frozen_fingerprint(agent) or source_identity(p)!=identity:
            raise AssertionError('Frozen evaluation changed learning state or source')
        moving=[r for r in rows if r['mode']=='moving']
        hover=[r for r in rows if r['mode']=='hover']
        pairs=paired_rows(moving,hover)
        if any(a['canonical_event_sha256']!=b['canonical_event_sha256'] for a,b in pairs):
            raise AssertionError('Canonical arrivals not paired')
        mp,hp=pooled_metrics(moving),pooled_metrics(hover)
        cost_ratio=mp['mean_cost_per_s']/hp['mean_cost_per_s']
        delay_ratio=mp['weighted_delay_ms']/hp['weighted_delay_ms']
        upload_ratio=sum(r['upload_area_mbit_s_5s'] for r in moving)/max(1e-12,sum(r['upload_area_mbit_s_5s'] for r in hover))
        q=p['quality_rule']
        bad=cost_ratio>q['cost_ratio_bad'] and delay_ratio>q['delay_ratio_bad'] and upload_ratio>q['upload_area_ratio_bad']
        report=dict(schema=SCHEMA,status='complete',arm=p['arm'],episode=state['completed_episodes'],
            checkpoint=str(checkpoint),checkpoint_sha256=file_sha(checkpoint),protocol_sha256=digest(p),
            identity=identity,frozen_verified=True,physical_steps=attempted,learning_updates=0,replay_writes=0,
            moving_pooled=mp,hover_pooled=hp,cost_ratio=cost_ratio,delay_ratio=delay_ratio,upload_area_ratio=upload_ratio,
            bad_performance_node=bad,rows=rows,elapsed_seconds=time.perf_counter()-started)
        atomic_json(output/'report.json',report)
        atomic_json(output/'progress.json',dict(status='complete',physical_steps=attempted))
        return report
    except BaseException as exc:
        atomic_json(output/'failure.json',dict(error=repr(exc),attempted_physical_steps=attempted))
        raise


def assert_counts(agent,scheduler,receipt):
    b=receipt['base_counters']
    events=scheduler.frames//scheduler.protocol['learning']['sac_update_frames']
    assert agent.sac_updates-b['sac']==events
    assert agent.sac_critic_gradient_steps-b['sac_critic_fits']==events*2
    assert agent.sac_target_updates-receipt['base_sac_target_updates']==events
    assert agent.sac_actor_updates-b['sac_actor']==(b['sac']+events)//2-b['sac']//2
    assert agent.replay.memory_counter-b['replay_writes']==scheduler.frames
    assert scheduler.ppo_samples==scheduler.consumed_ppo_samples+len(agent.macro_rollout)
    assert scheduler.physical_steps==scheduler.frames*5==scheduler.macros*50
    assert agent.ppo_updates==scheduler.macros//scheduler.protocol['learning']['ppo_update_macros']


def train(arm,output,resume=None):
    output=Path(output)
    p=make_protocol(arm) if resume is None else load_checkpoint(resume)['protocol']
    if p['arm']!=arm or p['purpose']!='pilot':
        raise ValueError('Research pilot arm mismatch')
    acceptance=read(HERE/'functional/acceptance.json')
    if acceptance.get('passed') is not True or acceptance.get('pilot_sha256')!=file_sha(__file__):
        raise ValueError('Matching bounded functional acceptance required')
    diagnostic_path=ROOT/'industrial_hotspot_analysis/load24725_initial_motion_20260923/summary.json'
    diagnostic=read(diagnostic_path)
    if diagnostic.get('status')!='complete':
        raise ValueError('Initial-position motion diagnosis must complete before pilot')
    output.mkdir(parents=True,exist_ok=False)
    identity=source_identity(p)
    if resume is None:
        agent,scene,receipt=make_initial_agent(p)
        state=dict(at_stream_boundary=True,completed_episodes=0,completed_seconds=0.,evaluations=[],
            transfer=receipt,wall_deadline_utc_epoch=time.time()+p['runtime']['max_wall_seconds_per_job'],diagnostic_path=str(diagnostic_path),
            diagnostic_sha256=file_sha(diagnostic_path),acceptance_sha256=file_sha(HERE/'functional/acceptance.json'))
    else:
        agent,scene,p,state=load_pilot(resume)
        receipt=state['transfer']
    base=receipt['base_counters']
    atomic_json(output/'protocol.json',p)
    atomic_json(output/'scene.json',scene)
    atomic_json(output/'transfer.json',receipt)
    manifest=dict(schema=SCHEMA,status='running',arm=arm,identity=identity,protocol_sha256=digest(p),
        started_utc=now(),transfer=receipt,source_directory=str(HERE),training_seed=p['training_seed'])
    atomic_json(output/'manifest.json',manifest)
    started=time.perf_counter()
    mdp=PilotMDP(scene,p)
    scheduler=DeploymentTrainingSchedule(agent,'ppo_sac',p,
        on_update=lambda row:append(output/'learning.jsonl',row))
    if resume is not None:
        scheduler.load_state_dict(state['schedule'])
    def remaining():
        value=min(p['runtime']['max_wall_seconds_per_job']-(time.perf_counter()-started),
                  state['wall_deadline_utc_epoch']-time.time())
        if value<=0:
            raise TimeoutError('Pilot wall cap reached; latest complete episode retained')
        return value
    def publish():
        if source_identity(p)!=identity:
            raise RuntimeError('Pilot or production source changed')
        state['schedule']=scheduler.state_dict()
        state['at_stream_boundary']=True
        payload=saved_payload(agent,p,scene,state)
        rec=save_checkpoint(output/'checkpoints/rolling',payload)
        manifest.update(committed_episodes=state['completed_episodes'],completed_seconds=state['completed_seconds'],
            updates=counters(agent),latest_checkpoint=rec,elapsed_seconds=time.perf_counter()-started)
        atomic_json(output/'manifest.json',manifest)
        return rec,payload
    def finish_pending_evaluation():
        pending=state.get('pending_evaluation')
        if pending is None:
            return
        episode=pending['episode']
        node_dir=Path(pending['checkpoint_directory'])
        destination=Path(pending['output_directory'])
        node_path=node_dir/'final.pt'
        if not node_path.exists():
            save_checkpoint(node_dir,saved_payload(agent,p,scene,state),persistent_name='final.pt')
        # Resume may reuse a completed, identity-checked evaluation. A partial
        # attempt is retained and never silently granted a fresh step budget.
        result_path=destination/'report.json'
        before=frozen_fingerprint(agent)
        if not result_path.exists():
            if destination.exists():
                raise RuntimeError('Partial evaluation retained; explicit remaining-budget recovery is required')
            cmd=[sys.executable,'-B',str(Path(__file__).resolve()),'evaluate','--checkpoint',str(node_path),'--output',str(destination)]
            atomic_json(output/'progress.json',dict(status='evaluating',arm=arm,committed_episodes=episode,
                node=episode,elapsed_seconds=time.perf_counter()-started,updates=counters(agent)))
            with (output/f'evaluation-{episode:04d}.stdout.txt').open('x') as stdout,(output/f'evaluation-{episode:04d}.stderr.txt').open('x') as stderr:
                subprocess.run(cmd,cwd=ROOT,stdout=stdout,stderr=stderr,check=True,timeout=min(remaining(),1000),
                    creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
        result=read(result_path)
        if (result['status']!='complete' or result['identity']!=identity or result['protocol_sha256']!=digest(p)
            or result['checkpoint_sha256']!=file_sha(node_path) or frozen_fingerprint(agent)!=before
            or result['episode']!=episode):
            raise AssertionError('Evaluation identity or parent learning state changed')
        if any(x['episode']==episode for x in state['evaluations']):
            raise AssertionError('Duplicate evaluation node')
        state['evaluations'].append(dict(episode=episode,path=str(result_path),sha256=file_sha(result_path),
            cost_ratio=result['cost_ratio'],delay_ratio=result['delay_ratio'],upload_area_ratio=result['upload_area_ratio'],
            bad=result['bad_performance_node']))
        state.pop('pending_evaluation')
        publish()
    try:
        finish_pending_evaluation()
        for index in range(state['completed_episodes'],50):
            remaining()
            ep_started=time.perf_counter()
            scheduler.set_stage('joint',50*180)
            seeds=seeds_for(p,index)
            real=reset_scene(mdp,seeds)
            state['at_stream_boundary']=False
            control=DeploymentLearningControllers(agent,mdp.encoder,deterministic=False)
            trajectory=[]
            def boundary(t):
                trajectory.append(dict(**snapshot(mdp),action=t.action.tolist(),upper_reward=t.reward,
                    cost_sum=mdp.env.cost_sum,holding_task_s=mdp.env.holding,energy_j=sum(mdp.env.energy.values())))
                scheduler.macro_boundary(t)
            rollout=PilotRollout(mdp,control.macro,control.micro,on_macro_boundary=boundary)
            upper_return=0.
            for macro in range(180):
                remaining()
                collected=rollout.run(1,truncate=macro==179,collect_metrics=macro==179)
                scheduler.completed_micro(collected)
                upper_return+=mdp.gamma_macro**macro*collected['macro'][0].reward
                if macro%60==59:
                    atomic_json(output/'progress.json',dict(status='running',arm=arm,committed_episodes=index,
                        active_episode=index+1,episode_simulated_seconds=(macro+1)*5,
                        elapsed_seconds=time.perf_counter()-started,updates=counters(agent)))
            metrics=collected['metrics']
            state.update(at_stream_boundary=True,completed_episodes=index+1,completed_seconds=(index+1)*900)
            assert_counts(agent,scheduler,receipt)
            if scheduler.physical_steps>p['budget']['max_training_physical_steps']:
                raise AssertionError('Pilot training budget exceeded')
            row=dict(episode=index+1,arm=arm,seeds=seeds,realization_sha256=real['realization_sha256'],
                profile_by_region=real['profile_by_region'],metrics=metrics,physical_return=-metrics['cost_sum'],
                upper_discounted_return=upper_return,updates=counters(agent),
                episode_training_seconds=time.perf_counter()-ep_started,trajectory=trajectory)
            state['last_episode']=row
            if index+1 in (25,50):
                state['pending_evaluation']=dict(episode=index+1,
                    checkpoint_directory=str(output/f'checkpoints/episode-{index+1:04d}'),
                    output_directory=str(output/f'evaluations/episode-{index+1:04d}'))
            checkpoint,payload=publish()
            row['checkpoint_sha256']=checkpoint['sha256']
            append(output/'episodes.jsonl',row)
            if index+1==10:
                durations=[json.loads(x)['episode_training_seconds'] for x in (output/'episodes.jsonl').read_text().splitlines()]
                atomic_json(output/'timing_first10.json',dict(training_seconds_per_episode_mean=float(np.mean(durations)),
                    remaining_training_seconds_estimate=float(np.mean(durations)*(50-index-1)),
                    pending_evaluation_trajectories=24,note='Estimate excludes evaluation time and later cost changes.'))
            if index+1 in (25,50):
                finish_pending_evaluation()
            atomic_json(output/'progress.json',dict(status='running',arm=arm,committed_episodes=index+1,
                elapsed_seconds=time.perf_counter()-started,updates=counters(agent),evaluations=state['evaluations']))
        manifest['status']='complete'
        if [x['episode'] for x in state['evaluations']]!=[25,50]:
            raise AssertionError('Required development nodes are incomplete')
        manifest['quality_status']='needs_review' if all(x['bad'] for x in state['evaluations']) else 'review_short_pilot'
        publish()
        atomic_json(output/'progress.json',dict(status='complete',quality_status=manifest['quality_status'],arm=arm,
            committed_episodes=state['completed_episodes'],updates=counters(agent),evaluations=state['evaluations'],
            elapsed_seconds=time.perf_counter()-started,automatic_extension=False))
    except BaseException as exc:
        manifest.update(status='interrupted' if isinstance(exc,(TimeoutError,KeyboardInterrupt)) else 'failed',
            error=repr(exc),elapsed_seconds=time.perf_counter()-started)
        atomic_json(output/'manifest.json',manifest)
        atomic_json(output/'progress.json',dict(status=manifest['status'],arm=arm,committed_episodes=state['completed_episodes'],error=repr(exc)))
        raise


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('phase',choices=['train','evaluate'])
    parser.add_argument('--arm',choices=['control','long150'])
    parser.add_argument('--output',required=True)
    parser.add_argument('--checkpoint')
    args=parser.parse_args()
    if args.phase=='evaluate':
        evaluate(args.checkpoint,args.output)
    else:
        train(args.arm,args.output,args.checkpoint)


if __name__=='__main__':
    main()
