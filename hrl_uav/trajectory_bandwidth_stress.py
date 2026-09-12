"""Same-state 3/6 MHz counterfactual sensitivity, deterministic SAC and no learning."""
import copy
import csv
import gzip
import json
import pickle
from pathlib import Path
import subprocess
import time

import numpy as np
import torch

from .bandwidth import bandwidth_config
from .experiment import append_csv, isolated_rng, write_json
from .local_counterfactual import (
    RUN_HEAD, SCENES, TIMES, HORIZON, EPSILON, C_ATOL, FullSnapshot, ResidualMeter,
    actions, physical_fields, physical_state, semantic_hash, macro_step,
    state_features, branch_metrics, branch_record, save_branch, validate_branch,
)
from .ppo_residual_b import prepare_residual_source
from .ppo_frozen import DEFAULT_SOURCE, SOURCE_DIGESTS
from .priority_grid import file_hash
from .resumption import capture_rng, restore_rng, state_equal
from .scene_prescreen import ROOT, assert_crn_equal, source_hashes
from .trajectory_check import ReadOnlySAC, no_learning, crn_records

REFERENCE = ROOT/'results/b_centered_local_counterfactual_sac101_retry1'
STRESS_SOURCE = ROOT/'results/wireless_model_fixed_rule_check/manifest.json'
REUSED_FILES = ('metrics.json','crn.csv','controls.npz','trajectory.npz','links.csv','frames.csv','routes.csv')


def read_csv(path):
    with Path(path).open(encoding='utf-8-sig',newline='') as handle:
        rows=list(csv.DictReader(handle))
    result=[]
    for row in rows:
        parsed={}
        for key,value in row.items():
            if key.endswith('sha256'):
                parsed[key]=value
            else:
                try:
                    parsed[key]=float(value)
                except ValueError:
                    parsed[key]=value
        result.append(parsed)
    return result


def switch_bandwidth(meter,mhz,fields):
    """Only current bandwidth and its static default matrix change; history stays intact."""
    if mhz not in (3,6) or isinstance(mhz,bool):
        raise ValueError('Only the pre-existing 3/6 MHz conditions are allowed')
    env=meter.env
    if env.wireless_model!='psd_per_uav_budget' or not env.dynamic_u2u_bandwidth:
        raise ValueError('This intervention requires the approved PSD dynamic-sharing model')
    before=physical_state(meter,fields)
    before_hash=semantic_hash(before)
    old_cfg=meter.config
    old_b=env.B_u2u_total
    old_matrix=env.u2u_link_bandwidth_matrix_hz.copy()
    meter.config=bandwidth_config(old_cfg,mhz)
    meter.config['environment']['u2u_bandwidth_mhz']=type(old_cfg['environment']['u2u_bandwidth_mhz'])(mhz)
    env.B_u2u_total=float(mhz)*1e6
    env.u2u_link_bandwidth_matrix_hz=env._normalize_u2u_link_bandwidth_matrix(None)
    # Last-allocation records are historical, not an actual-rate cache, and stay unchanged.
    checked=physical_state(meter,fields)
    checked['mdp']['config']=old_cfg
    checked['env']['B_u2u_total']=old_b
    checked['env']['u2u_link_bandwidth_matrix_hz']=old_matrix
    if semantic_hash(checked)!=before_hash:
        raise AssertionError('Bandwidth intervention changed another physical state/parameter')
    return dict(source_state_physical_sha256=before_hash,
        initial_state_excluding_bandwidth_equal=True,
        changed_fields=['config.environment.u2u_bandwidth_mhz','env.B_u2u_total',
                        'env.u2u_link_bandwidth_matrix_hz (derived static default)'])


def resource_metrics(links):
    active=sum(r['active_steps'] for r in links)
    service_link_steps=sum(r['service_steps'] for r in links)
    return dict(u2u_tx_time_s=sum(r['tx_time_s'] for r in links),
        u2u_served_mbit=sum(r['served_mbit'] for r in links),
        u2u_admitted_mbit=sum(r['admitted_mbit'] for r in links),
        actual_rate_mbps_active=sum(r['rate_mbps_sum'] for r in links)/max(active,1),
        actual_rate_mbps_all_links_service=sum(r['rate_mbps_sum'] for r in links)/max(service_link_steps,1),
        u2u_active_link_seconds=active*.1,
        u2u_tx_time_fraction_of_active=sum(r['tx_time_s'] for r in links)/max(active*.1,1e-12))


def resource_response(links,zero):
    if len(links)!=len(zero):
        raise AssertionError('Link recording grids differ')
    for a,b in zip(links,zero):
        for key in ('time_s','source','destination','service_steps'):
            if a[key]!=b[key]:
                raise AssertionError(f'Unpaired link frame: {key}')
    denominator=sum(r['service_steps'] for r in zero)
    # Actual allocated rates averaged within each 0.5s frame; inactive links count as zero.
    abs_rate=sum(abs(a['rate_mbps_sum']-b['rate_mbps_sum']) for a,b in zip(links,zero))/max(denominator,1)
    return dict(rate_response_abs_mbps_service=abs_rate,
        tx_response_abs_mbit=sum(abs(a['served_mbit']-b['served_mbit']) for a,b in zip(links,zero)),
        tx_time_response_abs_s=sum(abs(a['tx_time_s']-b['tx_time_s']) for a,b in zip(links,zero)),
        active_pattern_response_link_seconds=.1*sum(abs(a['active_steps']-b['active_steps']) for a,b in zip(links,zero)))


def augmented_metrics(values,links,zero_values,zero_links):
    metrics=dict(values,**resource_metrics(links))
    metrics.update(J60=metrics['C60']/600,J60_holding=metrics['C60_holding']/600,J60_energy=metrics['C60_energy']/600)
    reference=dict(zero_values,**resource_metrics(zero_links))
    reference.update(J60=reference['C60']/600,J60_holding=reference['C60_holding']/600,J60_energy=reference['C60_energy']/600)
    keys=('C60','C60_holding','C60_energy','J60','J60_holding','J60_energy','full_energy_j',
          'energy_compute_j','energy_upload_j','energy_u2u_j','energy_propulsion_j',
          'u2u_tx_time_s','u2u_served_mbit','u2u_admitted_mbit','actual_rate_mbps_active',
          'actual_rate_mbps_all_links_service','u2u_active_link_seconds','u2u_tx_time_fraction_of_active',
          'mean_queue_transfer_mbit','mean_queue_compute_mbit','mean_queue_upload_mbit',
          'queue_transfer_mbit','queue_compute_mbit','queue_upload_mbit','queue_total_mbit','unfinished',
          'unfinished_high_count','unfinished_low_count','completed_weighted_delay_ms','completed_high_delay_ms',
          'completed_low_delay_ms','completed_high_count','completed_low_count','ppo_return60')
    metrics.update({'delta_'+k:metrics[k]-reference[k] for k in keys})
    metrics.update(resource_response(links,zero_links))
    return metrics


def verify_reused_geometry(meter,macros,directory):
    with np.load(directory/'controls.npz') as controls:
        for key in ('current_uav','current_centroid','current_load','a_B','sampled_delta','pre_projection',
                    'a_exec','projected','actual_displacement'):
            np.testing.assert_array_equal(np.asarray([r[key] for r in meter.controls]),controls[key],err_msg=key)
    with np.load(directory/'trajectory.npz') as trajectory:
        for key in ('uav','users','centroid'):
            np.testing.assert_array_equal(np.asarray([m['trajectory'][key] for m in macros]),trajectory[key],err_msg=key)


def run(output):
    output=Path(output).resolve()
    if output.exists():
        raise FileExistsError('Will not overwrite or repeat an existing diagnosis')
    previous=json.loads((REFERENCE/'manifest.json').read_text(encoding='utf-8'))
    if previous['status']!='COMPLETE' or previous['completed_branches']!=520 or previous['actual_head']!=RUN_HEAD:
        raise ValueError('Incomplete or wrong 6 MHz reference')
    for path,digest in previous['source_hashes'].items():
        if file_hash(ROOT/path)!=digest:
            raise ValueError(f'Reference source changed: {path}')
    cfg,saved,provenance=prepare_residual_source()
    if cfg!=previous['config'] or cfg['environment']['u2u_bandwidth_mhz']!=6:
        raise ValueError('Source configuration changed')
    stress=json.loads(STRESS_SOURCE.read_text(encoding='utf-8'))['arm_configs']['w1_uniform_b3']['environment']
    for key in ('wireless_model','noise_psd_w_per_hz','u2u_tx_budget_w','u2u_reference_bandwidth_hz',
                'u2u_reference_power_w','compute_during_deployment','pause_service_during_flight'):
        if cfg['environment'][key]!=stress[key]:
            raise ValueError(f'Pre-existing stress condition differs in {key}')
    if stress['u2u_bandwidth_mhz']!=3:
        raise ValueError('Wrong stress bandwidth source')
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    if subprocess.check_output(['git','diff',RUN_HEAD,'--','hrl_uav','scripts','configs','tests'],cwd=ROOT,text=True):
        raise ValueError('Tracked implementation differs from the reference')
    original_protocol=json.loads((REFERENCE/'protocol.json').read_text(encoding='utf-8'))
    if original_protocol['scenes']!=list(SCENES) or original_protocol['branch_times_s']!=list(TIMES) or original_protocol['actions']!=actions():
        raise ValueError('Reference protocol does not match the fixed matrix')
    inputs={p.relative_to(ROOT).as_posix():file_hash(p) for p in
            (REFERENCE/'manifest.json',REFERENCE/'protocol.json',REFERENCE/'per_branch.csv',REFERENCE/'checks.json',STRESS_SOURCE)}
    states=json.loads((REFERENCE/'states.json').read_text(encoding='utf-8'))
    by_state={(s['seed'],s['start_s']):s for s in states}
    for seed in SCENES:
        for t in TIMES:
            directory=REFERENCE/'states'/f'{seed}_t{t}'
            for p in (directory/'physical_snapshot.pkl.gz',directory/'state.json'):
                inputs[p.relative_to(ROOT).as_posix()]=file_hash(p)
            for action in actions():
                for name in REUSED_FILES:
                    p=directory/action['name']/name
                    inputs[p.relative_to(ROOT).as_posix()]=file_hash(p)
    output.mkdir(parents=True)
    hashes=source_hashes()
    protocol=dict(original_protocol,bandwidth_mhz=[6,3],new_branch_count=520,reused_branch_count=520,
        initial_distribution='Saved 6 MHz B states; switch only future U2U bandwidth at the branch boundary',
        SAC_scope='SAC101 trained at6MHz is a frozen stress probe, not a3MHz trained/evaluated algorithm',
        hashing_prefix='Replay B at6MHz to860s solely to restore cumulative CRN hash contexts; exact physical/RNG snapshot matching required',
        sensitivity='Within-bandwidth candidate-minus-zero, then paired difference of those effects across bandwidths',
        rate_resolution='Actual allocated link rates, averaged within0.5s frames; all6 directed links including inactive zeros; service time denominator',
        J60='C60/600, not the mixed-history meter.metrics J or a900s performance estimate',
        controls='Same geometric trajectories across bandwidths for each residual; SAC routing regenerated from own state',
        inference='Development sensitivity only; no significance/generalization/long-run3MHz claims')
    write_json(output/'protocol.json',protocol)
    manifest=dict(status='PREFLIGHT',actual_head=head,
        branch=subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip(),
        worktree=subprocess.check_output(['git','status','--short'],cwd=ROOT,text=True),
        configurations={'b6':cfg,'b3':bandwidth_config(cfg,3)},source_hashes=hashes,reused_file_hashes=inputs,
        source_checkpoint=str(DEFAULT_SOURCE),source_checkpoint_sha256=SOURCE_DIGESTS['checkpoint'],
        provenance=provenance,protocol_sha256=file_hash(output/'protocol.json'),
        training=False,training_seed_count=0,SAC_source_training_seed=101,
        ppo_network_instantiated=False,replay_instantiated=False,learning_updates=0,reserved_scenes_executed=[],completed_states=0)
    write_json(output/'manifest.json',manifest)
    torch.set_num_threads(1)
    sac=ReadOnlySAC(saved)
    del saved
    fields=physical_fields(cfg)
    combined,checks=[],[]
    started=time.perf_counter()
    with isolated_rng(),no_learning(sac):
        for seed in SCENES:
            first=REFERENCE/'states'/f'{seed}_t100'/'physical_snapshot.pkl.gz'
            restore_rng(pickle.loads(gzip.decompress(first.read_bytes()))['rng'])
            prefix=ResidualMeter(cfg)
            prefix.reset(seed,90)
            snapshots,end_rng={},{}
            for i in range(86):
                t=i*10
                if t in TIMES:
                    state_features(prefix)
                    snap=FullSnapshot(prefix,fields)
                    disk=REFERENCE/'states'/f'{seed}_t{t}'/'physical_snapshot.pkl.gz'
                    blob=gzip.decompress(disk.read_bytes())
                    disk_hash=semantic_hash(pickle.loads(blob))
                    if disk_hash!=snap.initial_hash or disk_hash!=by_state[(seed,t)]['initial_state_sha256']:
                        raise AssertionError('Source B replay did not reproduce the saved full snapshot')
                    snap.blob=blob
                    snapshots[t]=snap
                macro_step(prefix,sac,np.zeros(6))
                if (i+1)*10 in [t+60 for t in TIMES]:
                    end_rng[(i+1)*10]=copy.deepcopy(prefix.env.np_random.get_state())
            prefix_hash=semantic_hash(physical_state(prefix,fields))
            for t in TIMES:
                reference=REFERENCE/'states'/f'{seed}_t{t}'
                base6=json.loads((reference/'zero'/'metrics.json').read_text())
                links6_zero=read_csv(reference/'zero'/'links.csv')
                zero3,links3_zero,crn3_zero=None,None,None
                snap=snapshots[t]
                for action in actions():
                    validate_branch(seed,t,action)
                    ref=reference/action['name']
                    old=json.loads((ref/'metrics.json').read_text())
                    if old['initial_state_sha256']!=snap.initial_hash or old['seed']!=seed or old['start_s']!=t or old['action']!=action['name']:
                        raise AssertionError('Reference result has wrong state/action provenance')
                    expanded6=augmented_metrics(old,read_csv(ref/'links.csv'),base6,links6_zero)
                    reused=dict(expanded6,bandwidth_mhz=6,reused=True,source_path=str(ref),
                                source_metrics_sha256=file_hash(ref/'metrics.json'))
                    meter=snap.restore()
                    audit=switch_bandwidth(meter,3,fields)
                    initial_completed=dict(meter.env.completed_priority_count)
                    initial_delays=dict(meter.env.completed_priority_delay_sum)
                    macros=[macro_step(meter,sac,np.asarray(action['delta']) if k==0 else np.zeros(6)) for k in range(HORIZON)]
                    values=branch_metrics(meter,macros,initial_completed,initial_delays)
                    if meter.env._current_time()!=t+60 or meter.reset_calls!=0 or meter.integration_steps!=(t+60)*10:
                        raise AssertionError('Incorrect branch duration/reset')
                    assert_crn_equal(read_csv(ref/'crn.csv'),crn_records(meter))
                    verify_reused_geometry(meter,macros,ref)
                    if not state_equal(meter.env.np_random.get_state(),end_rng[t+60]):
                        raise AssertionError('Cross-bandwidth exogenous environment RNG mismatch')
                    if not state_equal(capture_rng(),pickle.loads(snap.blob)['rng']):
                        raise AssertionError('Deterministic branch changed global RNG')
                    if semantic_hash(physical_state(prefix,fields))!=prefix_hash:
                        raise AssertionError('Clone mutated source B replay')
                    if action['name']=='zero':
                        zero3=values.copy()
                        links3_zero=copy.deepcopy(meter.link_readings)
                        crn3_zero=crn_records(meter)
                    else:
                        assert_crn_equal(crn3_zero,crn_records(meter))
                    expanded3=augmented_metrics(values,meter.link_readings,zero3,links3_zero)
                    record=dict(expanded3,seed=seed,action=action['name'],dimension=action['dimension'],
                                sign=action['sign'],initial_state_sha256=snap.initial_hash)
                    target=output/'states'/f'{seed}_t{t}'/action['name']
                    record.update(bandwidth_mhz=3,reused=False,source_path=str(target),source_metrics_sha256='new_run')
                    save_branch(target,meter,macros,record)
                    append_csv(output/'per_branch.csv',[reused,record])
                    combined.extend((reused,record))
                    checks.append(dict(seed=seed,start_s=t,action=action['name'],
                        initial_state_sha256=snap.initial_hash,**audit,source_snapshot_equal=True,
                        CRN_equal_across_bandwidth=True,CRN_equal_within_bandwidth=True,
                        UAV_trajectory_and_controls_equal=True,environment_rng_equal=True,
                        global_rng_unchanged=True,source_replay_unchanged=True))
                sac.assert_unchanged()
                manifest.update(status='RUNNING',completed_states=len(checks)//13,
                    new_branches=len(checks),reused_branches=len(checks),elapsed_seconds=time.perf_counter()-started)
                write_json(output/'manifest.json',manifest)
                print(f"Completed {len(checks)//13}/40 states, {len(checks)}/520 new3MHz branches: scene={seed},t={t}",flush=True)
            del prefix,snapshots
    if len(combined)!=1040 or len(checks)!=520:
        raise AssertionError('Fixed comparison matrix incomplete')
    if any(file_hash(ROOT/f)!=h for f,h in {**hashes,**inputs}.items()) or file_hash(DEFAULT_SOURCE)!=SOURCE_DIGESTS['checkpoint']:
        raise AssertionError('Source or reused evidence changed during the run')
    write_json(output/'checks.json',checks)
    write_json(output/'states.json',states)
    manifest.update(status='COMPLETE',sac_frozen_checks=sac.assert_unchanged(),source_unchanged=True,
        reused_evidence_unchanged=True,completed_states=40,new_branches=520,reused_branches=520,
        executed_new_branch_macros=3120,source_B_hash_replay_macros=430,elapsed_seconds=time.perf_counter()-started)
    write_json(output/'manifest.json',manifest)
    return output
