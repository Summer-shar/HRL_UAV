"""Fixed target x tracking comparison; reuse the read-only SAC diagnostic path."""

import copy
import json
from pathlib import Path
import time
from unittest.mock import patch

import numpy as np

from . import trajectory_check as base
from .experiment import append_csv, write_json
from .geometry import cooperation_aligned_basis
from .scene_prescreen import ROOT, assert_crn_equal, source_hashes
from .wireless_check import read_csv


OLD = ROOT / "results/trajectory_opportunity_sac101"
ATTRIBUTION = ROOT / "results/trajectory_propulsion_attribution_sac101"
SEEDS = (33600, 33601)
RULES = {name: next(c for c in base.candidates() if c["name"] == name) for name in ("zero", "light_xp")}
EXTRA = ("energy_steady_j", "energy_transient_j", "energy_hover_j", "mean_speed_mps", "max_speed_mps",
         "switch_energy_propulsion_j", "switch_energy_steady_j", "switch_energy_transient_j",
         "projection_count", "mean_tracking_error_to_original_target_m")
FIELDS = base.MEASURES + EXTRA
PAIRS = (("B","A"), ("D","C"), ("D","B"), ("D","A"))


def numeric_rows(path):
    rows = read_csv(path)
    for row in rows:
        for key,value in row.items():
            try:
                row[key] = float(value)
            except (TypeError,ValueError):
                pass
    return rows


def tracking_action(current, centroids, loads, offsets, radius, coefficient):
    if coefficient not in (.5, 1.0) or radius != 60:
        raise ValueError("Only coefficient 0.5 (experiment) or 1 (functional check) is allowed")
    current, centroids = np.asarray(current, float), np.asarray(centroids, float)
    if current.shape != (3,2) or centroids.shape != (3,2) or not np.isfinite([current,centroids]).all():
        raise ValueError("Expected finite 3-UAV world coordinates")
    original = base.role_action(loads, offsets, radius).reshape(3,2).astype(float)
    basis, _ = cooperation_aligned_basis(centroids, current, loads)
    basis = basis.astype(float)
    target = np.clip(centroids + radius*np.einsum("nij,nj->ni",basis,original),0,1000)
    interpolated = current + coefficient*(target-current)
    # Solve the actual float32-derived basis, not its approximately orthogonal transpose.
    raw = np.linalg.solve(basis, (interpolated-centroids)[...,None]).squeeze(-1)/radius
    raw = np.where(np.abs(raw-original)<1e-12, original, raw)
    norm = np.linalg.norm(raw,axis=1)
    feasible = raw/np.maximum(norm[:,None],1)
    world = np.clip(centroids + radius*np.einsum("nij,nj->ni",basis,feasible),0,1000)
    detail = dict(original_target=target, interpolated_target=interpolated, feasible_target=world,
                  projected=norm>1, inverse_norm=norm, original_action=original,
                  centroids=centroids, current=current.copy())
    return feasible.ravel(), detail


class TrackingPolicy(base.RolePolicy):
    def __init__(self, sac, candidate):
        super().__init__(sac,candidate)

    def macro_action(self, state, enabled=True, deterministic=False):
        if not enabled or not deterministic:
            raise AssertionError("Tracking diagnosis requires deterministic macro interface")
        self.macro_calls += 1
        env = self.meter.env
        raw, detail = tracking_action(env.uav_coords,env.weighted_centroids,env.zone_expected_offered_load_mbps,
                                      self.candidate["offsets_m"],self.meter.config["waypoint_radius_m"],
                                      self.candidate["tracking_coefficient"])
        self.meter.pending_tracking = detail
        return raw, 0.0


class TrackingMeter(base.OpportunityMeter):
    def __init__(self, config):
        super().__init__(config)
        self.tracking = []
        self.previous_roles = {}

    def start_macro(self, action):
        info = super().start_macro(action)
        detail = self.pending_tracking
        for deployment in self.deployments[-3:]:
            uav = deployment["uav"]
            role = deployment["role"]
            old = self.previous_roles.get(uav)
            category = "initial" if old is None else "role_switch" if old != role else "phase_internal"
            row = dict(macro=deployment["macro"],time_s=deployment["time_s"],uav=uav,role=role,
                       previous_role=old or "none",phase_age=deployment["phase_age"],category=category,
                       projected=bool(detail["projected"][uav]),inverse_action_norm=float(detail["inverse_norm"][uav]),
                       deployment_duration_s=self.env.deployment_dt,speed_mps=float(self.env.current_macro_v[uav]),
                       energy_steady_j=float(info["E_steady_flying_per_uav"][uav]),
                       energy_transient_j=float(info["E_deployment_transient_per_uav"][uav]),
                       energy_hover_j=float(info["E_hover_per_uav"][uav]),
                       energy_propulsion_j=float(info["E_macro_per_uav"][uav]))
            for key in ("original_target","interpolated_target","feasible_target","centroids","current"):
                row[key+"_x_m"],row[key+"_y_m"] = map(float,detail[key][uav])
            np.testing.assert_allclose([deployment["target_x_m"],deployment["target_y_m"]],
                                       detail["feasible_target"][uav],atol=1e-9,rtol=1e-12)
            self.tracking.append(row)
            self.previous_roles[uav] = role
        return info


def enrich(row, deployments):
    for key in ("energy_steady_j","energy_transient_j","energy_hover_j"):
        row[key] = sum(d[key] for d in deployments)
    np.testing.assert_allclose(sum(row[k] for k in EXTRA[:3]),row["energy_propulsion_j"],atol=1e-6,rtol=1e-12)
    row["mean_speed_mps"] = float(np.mean([d["speed_mps"] for d in deployments]))
    row["max_speed_mps"] = max(d["speed_mps"] for d in deployments)
    for key in ("energy_propulsion_j","energy_steady_j","energy_transient_j"):
        row["switch_"+key] = sum(d[key] for d in deployments if d["category"] == "role_switch")
    row["projection_count"] = sum(d.get("projected",False) for d in deployments)
    row["mean_tracking_error_to_original_target_m"] = float(np.mean([d["tracking_error_m"] for d in deployments]))


def evaluate(cfg, sac, target, coefficient, seed, macros, directory, label):
    if seed not in SEEDS or macros not in (2,90) or (macros==90 and (coefficient!=.5 or label not in ("B","D"))):
        raise ValueError("Only four declared 900s evaluations and short functional checks are authorized")
    candidate = dict(RULES[target],name=label,tracking_coefficient=coefficient)
    with patch.object(base,"RolePolicy",TrackingPolicy), patch.object(base,"OpportunityMeter",TrackingMeter):
        row, meter = base.evaluate_rule(cfg,sac,candidate,seed,macros,directory,"tracking_check")
    for d,t in zip(meter.deployments,meter.tracking):
        t.update(end_x_m=d["after_x_m"],end_y_m=d["after_y_m"],distance_m=d["distance_m"],
                 tracking_error_m=float(np.linalg.norm(np.array([d["after_x_m"],d["after_y_m"]])-
                                                       np.array([t["original_target_x_m"],t["original_target_y_m"]]))))
        np.testing.assert_allclose(t["distance_m"],t["speed_mps"]*t["deployment_duration_s"],atol=1e-8,rtol=1e-12)
    enrich(row,meter.tracking)
    row.update(reused=False, source_path=str(directory))
    append_csv(directory/"tracking.csv",meter.tracking)
    write_json(directory/"metrics.json",row)
    return row,meter


def pairing(left, right, same=False):
    a,b=Path(left["source_path"]),Path(right["source_path"])
    assert_crn_equal(read_csv(a/"crn.csv"),read_csv(b/"crn.csv"))
    with np.load(a/"trajectory.npz") as x,np.load(b/"trajectory.npz") as y:
        for key in ("users","centroid") + (("uav","raw_action","executed_route") if same else ()):
            np.testing.assert_array_equal(x[key],y[key])
        if left["policy"]=="A" and right["policy"] in ("B","D"):
            distances=np.linalg.norm(y["uav"]-x["uav"],axis=-1)
            right["mean_position_delta_vs_zero_m"]=float(distances.mean())
            right["max_position_delta_vs_zero_m"]=float(distances.max())
    if same:
        for key in base.MEASURES:
            np.testing.assert_allclose(left[key],right[key],atol=1e-9,rtol=1e-12,err_msg=key)
    return {"pair":right["policy"]+"-"+left["policy"],"seed":right["seed"],"crn_equal":True,
            **{"delta_"+k:right[k]-left[k] for k in FIELDS if k in left and k in right}}


def reuse(cfg):
    old=base.load_json(OLD/"manifest.json")
    corrected=base.load_json(OLD/"analysis_manifest.json")
    attr=base.load_json(ATTRIBUTION/"manifest.json")
    if old["actual_head"]!=base.REVIEWED_HEAD or old["config"]!=cfg or old["source_digests"]!=base.SOURCE_DIGESTS:
        raise ValueError("Existing A/C source/configuration mismatch")
    for name,digest in old["source_hashes_before"].items():
        expected=corrected["postrun_diagnostic_source_hashes"].get(name,digest)
        if base.file_hash(ROOT/name)!=expected:
            raise ValueError(f"Prior physical/statistical source mismatch: {name}")
    for name,digest in attr["input_sha256"].items():
        if base.file_hash(ROOT/name)!=digest:
            raise ValueError(f"Old energy-component provenance mismatch: {name}")
    saved=numeric_rows(OLD/"per_seed_corrected.csv")
    energy=numeric_rows(ATTRIBUTION/"per_deployment.csv")
    hashes={}
    for path in (OLD/"manifest.json",OLD/"analysis_manifest.json",OLD/"per_seed_corrected.csv",
                 OLD/"deployments_corrected.csv",ATTRIBUTION/"manifest.json",ATTRIBUTION/"per_deployment.csv"):
        hashes[str(path.relative_to(ROOT))]=base.file_hash(path)
    rows=[]
    for label,name in (("A","zero"),("C","light_xp")):
        for seed in SEEDS:
            row=copy.deepcopy(next(r for r in saved if r["policy"]==name and r["seed"]==seed))
            directory=OLD/"exploration"/name/str(seed)
            for path in directory.iterdir():
                if path.is_file(): hashes[str(path.relative_to(ROOT))]=base.file_hash(path)
            raw=base.load_json(directory/"metrics.json")
            for key in base.MEASURES:
                if key!="fleet_distance_m": np.testing.assert_allclose(row[key],raw[key],atol=1e-12,rtol=1e-12)
            ds=[]
            for d in energy:
                if d["policy"]==name and d["seed"]==seed:
                    ds.append(dict(speed_mps=d["speed_mps"],energy_steady_j=d["steady_j"],
                                   energy_transient_j=d["transient_j"],energy_hover_j=d["hover_j"],
                                   energy_propulsion_j=d["propulsion_j"],category=d["category"],
                                   tracking_error_m=d["target_remaining_m"]))
            assert len(ds)==270
            enrich(row,ds)
            row.update(policy=label,seed=seed,reused=True,source_path=str(directory))
            rows.append(row)
    checks=[pairing(next(r for r in rows if r["policy"]=="A" and r["seed"]==s),
                    next(r for r in rows if r["policy"]=="C" and r["seed"]==s)) for s in SEEDS]
    return rows,hashes,checks


def prepare(output):
    cfg,saved,_=base.source_bundle()
    old,hashes,checks=reuse(cfg)
    if base.git("rev-parse","HEAD")!=base.REVIEWED_HEAD or base.git("diff",base.REVIEWED_HEAD,"--","hrl_uav","scripts","configs"):
        raise ValueError("Current tracked implementation differs from the reviewed baseline")
    output.mkdir(parents=True,exist_ok=False)
    protocol=dict(controllers={"A":"original zero (reused)","B":"world interpolation 0.5 to current centroid",
                               "C":"original light +20 local x (reused)","D":"world interpolation 0.5 to original current light +20 target"},
                  coefficients=[.5],functional_coefficient=1,seeds=SEEDS,duration_s=900,
                  new_900s_runs=4,reused_900s_runs=4,comparisons=PAIRS,
                  projection="solve original current local basis; radially project inverse action to unit disk; apply original world clipping, movement cap and safety",
                  original_target="same current A/C target, not C raw action multiplied by 0.5",
                  training=False,replay=False,deterministic_sac=True,complete_tail=False,
                  forbidden_scenes=list(range(33700,33710)),no_other_scenes=True)
    write_json(output/"protocol.json",protocol)
    append_csv(output/"reused_per_seed.csv",old)
    write_json(output/"reuse_checks.json",checks)
    m=dict(status="PREPARED",actual_head=base.git("rev-parse","HEAD"),branch=base.git("branch","--show-current"),
           worktree_before=base.git("status","--short"),config=cfg,source_digests=base.SOURCE_DIGESTS,
           source_checkpoint=str(base.DEFAULT_SOURCE),source_hashes_before=source_hashes(),reuse_file_hashes=hashes,
           protocol_sha256=base.file_hash(output/"protocol.json"),new_900s_runs=0,training=False,replay=False,
           source_checkpoint_updates=saved["updates"],started_unix=time.time())
    write_json(output/"manifest.json",m)
    print(json.dumps({"status":"PREPARED","source_checkpoint_sha256":base.SOURCE_DIGESTS["checkpoint"],"reuse":4}),flush=True)


def assert_manifest(output,status):
    m=base.load_json(output/"manifest.json")
    if m["status"]!=status or m["source_hashes_before"]!=source_hashes() or m["protocol_sha256"]!=base.file_hash(output/"protocol.json"):
        raise ValueError("Frozen status/source/protocol mismatch")
    for name,h in m["reuse_file_hashes"].items():
        if base.file_hash(ROOT/name)!=h: raise ValueError(f"Reused input changed: {name}")
    return m


def smoke(output):
    m=assert_manifest(output,"PREPARED")
    cfg,saved,_=base.source_bundle()
    sac=base.ReadOnlySAC(saved)
    del saved
    checks=[]
    references={}
    for label,target in (("A","zero"),("C","light_xp")):
        directory=output/"smoke"/(label+"_native")
        row,_=base.evaluate_rule(cfg,sac,RULES[target],33600,2,directory,"smoke")
        row.update(policy=label,source_path=str(directory))
        references[label]=row
        same,_=evaluate(cfg,sac,target,1,33600,2,output/"smoke"/(label+"_coefficient1"),label+"_coefficient1")
        checks.append(pairing(row,same,same=True))
    for label,target,reference in (("B","zero","A"),("D","light_xp","C")):
        row,meter=evaluate(cfg,sac,target,.5,33600,2,output/"smoke"/label,label)
        checks.append(pairing(references[reference],row))
        assert all(not r["projected"] for r in meter.tracking)
    write_json(output/"smoke"/"report.json",dict(status="SMOKE_COMPLETE",checks=checks,frozen=sac.assert_unchanged(),runs=6))
    m.update(status="SMOKE_COMPLETE",smoke_runs=6)
    write_json(output/"manifest.json",m)
    print(json.dumps({"status":"SMOKE_COMPLETE","runs":6,"coefficient1_exact":True}),flush=True)


def report(output,rows,pairs,means):
    def table(lines,headers,values):
        lines.extend(["", "| "+" | ".join(headers)+" |", "| "+" | ".join("---" for _ in headers)+" |"])
        lines.extend("| "+" | ".join(f"{v:.6f}" if isinstance(v,(float,np.floating)) else str(v) for v in r)+" |" for r in values)
    lines=["# 目标位置 × 追踪方式：固定系数0.5最小对照", "",
           "A=原动态质心；B=渐进质心；C=原轻载角色(+20,0)m；D=渐进轻载角色(+20,0)m。",
           "B、D先生成各自当前原目标r，在世界坐标取q+0.5(r-q)，再按当前局部基反算动作并投影到原单位圆。不是将C的20m偏移改为10m。",
           "仅新增B、D各33600、33601共4次900秒真实闭环评价；A、C四条记录及推进分项复用已核验结果。",
           "运行基准b4908d3e78d4c807013c8ff6991637dbae546129；指定SAC101 window2000及哈希核验通过。",
           "物理、奖励、动作映射、部署时序不变；确定性冻结SAC按每条轨迹自己的状态生成路由，无学习更新、replay写入/采样。",
           "", "## 两场景均值", "",
           "J为原平均物理代价；J驻留/能量已包含原权重和归一化。完成时延只对已完成父任务统计。"]
    table(lines,["控制器","J","J驻留","J能量","加权ms","高优ms","低优ms","功率W"],
          [[r["policy"],*[r[k] for k in ("J","J_holding","J_energy","task_delay_weighted_ms","task_delay_high_ms","task_delay_low_ms","power_w")]] for r in means])
    lines += ["", "## 配对差", "", "全部是前者减后者；J和时延负值更低。仅两个已使用开发场景，不宣称显著性、泛化或最优。"]
    for title,items in (("逐场景",pairs),("两场景均值",[dict(pair=p,seed="mean",**{k:float(np.mean([r[k] for r in pairs if r['pair']==p])) for k in pairs[0] if k.startswith('delta_')}) for p in ("B-A","D-C","D-B","D-A")])):
        lines += ["", "### "+title]
        table(lines,["比较","场景","ΔJ","ΔJ驻留","ΔJ能量","Δ加权ms","Δ高优ms","Δ低优ms","Δ功率W"],
              [[r["pair"],r["seed"],*[r[k] for k in ("delta_J","delta_J_holding","delta_J_energy","delta_task_delay_weighted_ms","delta_task_delay_high_ms","delta_task_delay_low_ms","delta_power_w")]] for r in items])
    lines += ["", "## 完成与积压", "", "平均队列为0.5秒帧末采样，期末队列为900秒。未完成不等于超时。任务数量小数为两场景平均。"]
    table(lines,["控制器","高优完成%","低优完成%","高优未完成","低优未完成","平均上传","平均计算","平均转发","期末上传","期末计算","期末转发"],
          [[r["policy"],r["task_completion_high_rate"]*100,r["task_completion_low_rate"]*100,
            *[r[k] for k in ("task_unfinished_high_count","task_unfinished_low_count","mean_queue_upload_mbit","mean_queue_compute_mbit","mean_queue_transfer_mbit","queue_upload_mbit","queue_compute_mbit","queue_transfer_mbit")]] for r in means])
    lines += ["", "## 运动和能量", "",
              "三机总路程是900秒全部部署腿之和；速度均值按每个周期/每UAV部署等权（含零位移）。分项单位J。角色切换不含初始部署。",
              "A/C分项复用此前按已保存位移和运行原公式重算且对账通过的记录；B/D分项直接记录本轮env.macro_step返回值。"]
    table(lines,["控制器","总路程m","均速m/s","最大速度","稳态J","附加J","悬停J","切换当次推进J","切换稳态J","切换附加J","投影次数"],
          [[r["policy"],*[r[k] for k in ("fleet_distance_m","mean_speed_mps","max_speed_mps","energy_steady_j","energy_transient_j","energy_hover_j","switch_energy_propulsion_j","switch_energy_steady_j","switch_energy_transient_j","projection_count")]] for r in means])
    lines += ["", "投影逐次记录在B/D各场景tracking.csv中，含原目标、世界插值目标、可行目标、投影标志、原始动作范数、实际起止、速度及原能量分项。",
              "所有900秒结果使用真实通信/队列演进；未沿用旧驻留收益替换能量估算新J。逐场景能量、各阶段积压、后段斜率、实际U2U吞吐/速率及全部所需差值见CSV。",
              "", "## 验证和限制", "",
              "短功能检查6次20秒；同一插值实现系数1复现原A/C的UAV、路由、任务结果和能量。系数0.5走原start_macro，保持原1秒部署/9秒悬停、CPU及通信暂停协议。",
              "同种子的逐任务出生、物理步用户移动、负载、业务质心配对一致；不要求UAV轨迹/队列/路由相同。学习状态保持不变，经验池未实例化，无PPO。",
              "这是2×2固定规则开发对照，不是PPO训练或新的最优轨迹搜索；未运行其他场景、保留集或其他系数。",
              "既有未提交文件、配置、生产代码和历史结果保持不变；本轮不提交推送。", "",
              "文件：manifest.json、protocol.json、reused_per_seed.csv、per_seed.csv、summary.csv、paired_deltas.csv、paired_mean.csv、pairing_checks.json；evaluations下保存B/D真实轨迹、队列、路由、链路、部署与投影记录。"]
    (output/"result.md").write_text("\n".join(lines)+"\n",encoding="utf-8")


def run(output):
    m=assert_manifest(output,"SMOKE_COMPLETE")
    cfg,saved,_=base.source_bundle()
    sac=base.ReadOnlySAC(saved)
    del saved
    rows=numeric_rows(output/"reused_per_seed.csv")
    checks=[]
    for label,target in (("B","zero"),("D","light_xp")):
        for seed in SEEDS:
            row,_=evaluate(cfg,sac,target,.5,seed,90,output/"evaluations"/label/str(seed),label)
            reference=next(r for r in rows if r["policy"]=="A" and r["seed"]==seed)
            checks.append(pairing(reference,row))
            write_json(Path(row["source_path"])/"metrics.json",row)
            rows.append(row)
            append_csv(output/"new_per_seed.csv",[row])
            m["new_900s_runs"]+=1
            write_json(output/"manifest.json",m)
            print(json.dumps({"completed":m["new_900s_runs"],"of":4,"policy":label,"seed":seed,"J":row["J"],"delay_ms":row["task_delay_weighted_ms"]}),flush=True)
    pairs=[]
    for new,old in PAIRS:
        for seed in SEEDS:
            a=next(r for r in rows if r["policy"]==old and r["seed"]==seed)
            b=next(r for r in rows if r["policy"]==new and r["seed"]==seed)
            pairs.append(pairing(a,b))
    means=[dict(policy=label,scene_count=2,**{k:float(np.mean([r[k] for r in rows if r['policy']==label])) for k in FIELDS}) for label in "ABCD"]
    paired_means=[dict(pair=p,scene_count=2,**{k:float(np.mean([r[k] for r in pairs if r['pair']==p])) for k in pairs[0] if k.startswith('delta_')}) for p in ("B-A","D-C","D-B","D-A")]
    append_csv(output/"per_seed.csv",[{k:r[k] for k in ("policy","seed","reused","source_path",*FIELDS)} for r in sorted(rows,key=lambda r:(r['policy'],r['seed']))])
    append_csv(output/"summary.csv",means)
    append_csv(output/"paired_deltas.csv",pairs)
    append_csv(output/"paired_mean.csv",paired_means)
    write_json(output/"pairing_checks.json",checks)
    assert_manifest(output,"SMOKE_COMPLETE")
    m.update(status="COMPLETE",source_hashes_after=source_hashes(),frozen=sac.assert_unchanged(),
             elapsed_s=time.time()-m["started_unix"],reused_files_unchanged=True,reserved_scenes_run=[],
             ppo_updates=0,sac_update_increment=0,replay_reads=0,replay_writes=0)
    write_json(output/"manifest.json",m)
    report(output,rows,pairs,means)
    print(json.dumps({"status":"COMPLETE","new_900s_runs":4,"reused":4}),flush=True)
