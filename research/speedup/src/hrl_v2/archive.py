"""Checkpoint cadence and committed-attempt logs; no learner/RNG operations."""
import json
import os
from pathlib import Path
import re
import struct
import uuid

import torch
from .checkpoint import atomic_json, digest


def checkpoint_interval(value):
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError('checkpoint_every must be a positive integer')
    return value


def should_checkpoint(completed, every, *, pending=None, final=600, boundaries=(300,400)):
    checkpoint_interval(every)
    return completed == 0 or completed % every == 0 or completed in boundaries or completed == final or pending is not None


def publish_checkpoint(root, output, path, completed, attempt_id):
    """Commit the checkpoint pointer and its log owners in one atomic write."""
    root, output, path = Path(root), Path(output), Path(path)
    pointer = output/'latest.json'
    previous = json.loads(pointer.read_text(encoding='utf-8')) if pointer.exists() else {}
    before = previous.get('completed_episode', 0)
    if completed < before:
        raise ValueError('Cannot move the committed boundary backwards')
    owners = dict(previous.get('committed_attempts', {}))
    for number in range(before+1, completed+1): owners[str(number)] = attempt_id
    atomic_json(pointer, dict(path=str(path.relative_to(root)).replace('\\','/'),
        sha256=digest(path), completed_episode=completed, attempt_id=attempt_id,
        committed_attempts=owners))


def canonical_rows(path):
    """Only records owned by each episode's committed attempt, in file order.

    Training metrics have one row per episode; learning logs can have many
    events per episode. Uncommitted episodes and discarded attempts are absent.
    """
    path = Path(path)
    if not path.exists(): return []
    pointer = json.loads((path.parent/'latest.json').read_text(encoding='utf-8'))
    if 'committed_attempts' not in pointer:
        raise ValueError('Canonical logs require checkpoint attempt ownership')
    owners = pointer['committed_attempts']
    result = []
    for line in path.read_text(encoding='utf-8').splitlines():
        try: row = json.loads(line)
        except ValueError: continue  # interrupted final append is not a record
        number = row.get('global_episode', row.get('episode'))
        if number is not None and owners.get(str(number)) == row.get('attempt_id'):
            result.append(row)
    return result


def supersede_episodes(output, completed):
    """Preserve uncommitted episode files under their original attempt IDs."""
    output = Path(output).resolve()
    moved = []
    for source in sorted((output/'episodes').glob('episode-*.json')):
        match = re.fullmatch(r'episode-(\d+)\.json', source.name)
        if match is None or int(match[1]) <= completed: continue
        row = json.loads(source.read_text(encoding='utf-8'))
        owner = row['attempt_id']
        if not isinstance(owner,str) or re.fullmatch(r'[A-Za-z0-9_-]+',owner) is None:
            raise ValueError('Unsafe superseded attempt ID')
        target = output/'episodes'/'superseded'/owner/source.name
        if output not in source.resolve().parents or output not in target.resolve().parents:
            raise ValueError('Superseded episode path escaped run')
        if target.exists(): raise FileExistsError('Never overwrite superseded evidence')
        target.parent.mkdir(parents=True,exist_ok=True)
        source.rename(target)
        moved.append(str(target.relative_to(output)).replace('\\','/'))
    return moved


def recover_metric(output, completed):
    """A committed checkpoint can precede the metric log append."""
    output = Path(output)
    if not completed: return
    path = output/'train_metrics.jsonl'
    if any(r.get('global_episode',r.get('episode')) == completed for r in canonical_rows(path)):
        return
    record = json.loads((output/'episodes'/f'episode-{completed:04d}.json').read_text(encoding='utf-8'))
    owners = json.loads((output/'latest.json').read_text(encoding='utf-8'))['committed_attempts']
    if owners[str(completed)] != record['attempt_id']:
        raise ValueError('Episode file does not belong to the committed attempt')
    row = dict(record['train_metric'], recovered_after_interruption=True)
    with path.open('a',encoding='utf-8',newline='\n') as f:
        f.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')


def _equal_state(a,b):
    if isinstance(a,torch.Tensor) and isinstance(b,torch.Tensor):
        return (a.dtype==b.dtype and a.shape==b.shape and
            torch.equal(a.detach().cpu().contiguous().reshape(-1).view(torch.uint8),
                        b.detach().cpu().contiguous().reshape(-1).view(torch.uint8)))
    if type(a) is not type(b): return False
    if isinstance(a,dict): return a.keys()==b.keys() and all(_equal_state(a[k],b[k]) for k in a)
    if isinstance(a,(tuple,list)): return len(a)==len(b) and all(_equal_state(x,y) for x,y in zip(a,b))
    if isinstance(a,float): return struct.pack('>d',a)==struct.pack('>d',b)
    return a==b


def save_network_snapshot(output, learner, completed, identity):
    """Non-resumable snapshot; no replay, named/global RNG, or state mutation."""
    path = Path(output)/'snapshots'/f'episode-{completed:04d}.pt'
    payload = dict(learner=learner.state_dict(), episode=completed, identity=identity,
                   resumable=False, kind='network_snapshot_not_resumable')
    if path.exists():
        sidecar = json.loads(path.with_suffix('.json').read_text(encoding='utf-8'))
        if digest(path)!=sidecar['sha256'] or not _equal_state(torch.load(path,map_location='cpu',weights_only=False),payload):
            raise ValueError('Existing immutable network snapshot differs from replayed episode')
        return path
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary = path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp')
    try:
        with temporary.open('wb') as f:
            torch.save(payload,f);f.flush();os.fsync(f.fileno())
        os.replace(temporary,path)
    finally:
        if temporary.exists(): temporary.unlink()
    atomic_json(path.with_suffix('.json'),dict(episode=completed,identity=identity,
        sha256=digest(path),resumable=False,kind='network_snapshot_not_resumable'))
    return path
