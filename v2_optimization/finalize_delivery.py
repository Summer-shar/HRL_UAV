# -*- coding: utf-8 -*-
"""Assemble the bounded evidence and a reviewable patch; never trains."""
import difflib
import hashlib
import json
from pathlib import Path
import subprocess

from seed_budget import budget

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"hrl_v2/validation"
WORK=ROOT/"v2_optimization"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def save(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")


def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT)


def main():
    before=read(WORK/"baseline_final.json")
    after=read(WORK/"optimized_final.json")
    validation=read(OUT/"validation.json")
    cpu=read(OUT/"optimization_equivalence_cpu.json")
    cuda=read(OUT/"optimization_equivalence_cuda.json")
    assert before["status"]==after["status"]=="complete"
    assert before["fixture_sha256"]==after["fixture_sha256"]
    cold=read(OUT/"throughput_cold_inference.json")
    assert cold["status"]=="complete" and cold["fixture_sha256"]==after["fixture_sha256"]
    # Preserve original repeated-window timings and explicitly label the
    # cold-graph supplement used for online inference and the seed budget.
    after["original_window_measurements"]={k:after["measurements"][k] for k in cold["measurements"]}
    after["measurements"].update(cold["measurements"])
    after["online_inference_supplement"]={k:v for k,v in cold.items() if k!="measurements"}
    assert validation["status"]=="passed" and cpu["passed"] and cuda["passed"]
    for relative in ("hrl_v2/configs/v2_1_0.json", "hrl_v2/configs/physical_scene.json"):
        assert (ROOT/relative).read_bytes()==git("show", f"137d2f1:{relative}")
    # Validation records bind the production files, including the benchmark.
    for relative, expected in validation["identity"]["files"].items():
        assert hashlib.sha256((ROOT/relative).read_bytes()).hexdigest()==expected, relative
    save(OUT/"throughput_baseline.json",before)
    save(OUT/"throughput_optimized.json",after)
    io=read(OUT/"throughput_extra_io.json")
    budgets=dict(baseline=budget(before,extra_io=io),
                 optimized_micro16=budget(after,extra_io=io),
                 optimized_micro32=budget(after,32,io))
    save(OUT/"throughput_budget.json",budgets)
    b,o=before["measurements"],after["measurements"]
    def seconds(data,key): return data[key]["seconds"]
    rows=[("双评价拟合，有效 batch128 / micro16","critic128_micro16"),
          ("下层动作更新，有效 batch128 / micro16","actor128_micro16"),
          ("PPO 首个150样本动作＋value步骤","ppo150_actual_first_minibatch"),
          ("PPO 完整900样本、6 epoch","ppo900_complete_batch"),
          ("PPO 整批 KL900","ppo_kl900"),
          ("普通状态下层在线推理","ordinary_lower_inference"),
          ("拥堵状态下层在线推理","congested_lower_inference"),
          ("普通状态上层动作推理","ordinary_upper_inference"),
          ("拥堵状态上层动作推理","congested_upper_inference"),
          ("四条1800帧历史前向＋完整反传","four_full_1800_history_forward_backward"),
          ("两份原始图特征构建","fixed_graph_features_two_reports"),
          ("16状态预处理后的图动作网络","graph_and_actor16_history_precomputed"),
          ("128样本报告／历史解包","resolve_128"),
          ("1800帧历史解压","decompress_full_history"),
          ("历史归一化＋H2D","history_normalize_and_h2d"),
          ("纯历史H2D","history_h2d_only"),
          ("16状态分布logits D2H","distribution_logits_d2h16"),
          ("CPU float64 分布采样16状态","cpu_float64_distribution_sample16"),
          ("CPU float64 代理损失16状态（无网络）","cpu_float64_surrogate16_without_network"),
          ("32份报告压缩","report_compression32"),
          ("含Adam的空回放检查点保存","checkpoint_empty_replay_save"),
          ("含4条独立episode检查点保存","checkpoint_four_episodes_save"),
          ("4条episode检查点加载","checkpoint_four_episodes_load"),
          ("完整9000步脚本采集，含压缩","scripted_collection_9000")]
    lines=["## 最终有界测量", "",
           "设备、软件版本、源文件指纹、fixture 指纹及原始重复值保存在 `throughput_baseline.json` 和 `throughput_optimized.json`。两版使用相同的5条独立脚本轨迹和采样序列，串行执行；SAC每项预热1次、测量3次取中位数，完整PPO各实测1批。普通/拥堵推理使用连续状态窗口，并单测参数更新后的历史重建。优化版窗口每次重复前清空固定图LRU，避免旧报告跨重复命中；补测及原窗口数据分别留存在 `throughput_cold_inference.json` 和汇总JSON，预算采用清空后的数据。基线单状态入口本来就逐次构图。", "",
           "以下单位均为秒；负的缩短比例表示该项实测变慢，分项计时不重复加到端到端预算。", "",
           "| 项目 | 137d2f1 | 优化后 | 耗时缩短 |", "|---|---:|---:|---:|"]
    for label,key in rows:
        old,new=seconds(b,key),seconds(o,key)
        lines.append(f"| {label} | {old:.6f} | {new:.6f} | {(1-new/old)*100:.1f}% |")
    lines += [f"| 开发评价9000步物理／报告／指标（不含神经推理） | {seconds(b,'scripted_collection_9000'):.6f} | {seconds(o,'streaming_evaluation_9000'):.6f} | {(1-seconds(o,'streaming_evaluation_9000')/seconds(b,'scripted_collection_9000'))*100:.1f}% |", "",
        f"优化版16状态的固定图索引／设备张量准备实测 {seconds(o,'graph_index_device_prepare16_cold'):.6f} 秒，同批复用 {seconds(o,'graph_index_device_prepare16_reuse'):.6f} 秒。这两项是优化内部的分项，不能替代整个网络更新时间。", "",
        f"两版完整 PPO 的实际计数：基线 `{b['ppo900_complete_batch']['counts']}`；优化后 `{o['ppo900_complete_batch']['counts']}`。辅助动作由当前策略在脚本轨迹上生成，因此这是计算验证 fixture，不是用于科研结论的 on-policy 训练批次。", "",
        "### microbatch 调优", "", "| 设置 | critic128 秒 | actor128 秒 | critic 峰值分配 MiB | actor 峰值分配 MiB |", "|---|---:|---:|---:|---:|"]
    for label,data,mb in (("基线16",b,16),("优化16",o,16),("优化32候选",o,32)):
        c,a=data[f"critic128_micro{mb}"],data[f"actor128_micro{mb}"]
        lines.append(f"| {label} | {c['seconds']:.3f} | {a['seconds']:.3f} | {c['peak_allocated_bytes']/2**20:.1f} | {a['peak_allocated_bytes']/2**20:.1f} |")
    lines += ["", "默认仍为 microbatch16。32仅作为已测工程候选：改变分组会改变采样调用顺序，当前逐流一致性对照采用16，未自动切换。两者有效batch都为128，PPO优化小批次仍为150。未执行batch64/50秒备选配置。", "",
              "CPU float64 分布和传输单项实测已列出；它们的单次耗时远小于完整网络更新，不能据此把分布预设为最大瓶颈。本次保留CPU参考后端。图构建本身未改物理特征；收益来自索引／固定张量复用，原始构建单次成本没有消失。", "",
              "### 最终一致性", "",
              f"完整有界套件 **{validation['tests']} 项通过**，失败和错误均为0；额外CUDA基线集成对照 **{cuda['tests_run']} 项通过**。日志见 `tests.txt`、`equivalence_cuda_tests.txt`；误差和容差见 `optimization_equivalence_cpu.json`、`optimization_equivalence_cuda.json`。", "",
              "| 设备 | 最大参数绝对差 | 最大梯度绝对差 | 最大Adam状态绝对差 |", "|---|---:|---:|---:|"]
    for data in (cpu,cuda):
        def maximum(suffix):
            return max((v for test in data["measurements"].values() for k,v in test.items() if k.endswith(suffix)),default=0.)
        lines.append(f"| {data['device']} | {maximum('_parameter_max_abs'):.3g} | {maximum('_gradient_max_abs'):.3g} | {maximum('_state_max_abs'):.3g} |")
    lines += ["", "数值比较验证输出、梯度、Adam及执行语义的容差等价，不声称长期优化轨迹逐位相同。没有改数值后端。", "",
              "### 完整 seed 墙钟预算", "",
              "`throughput_budget.json` 给出每一项数量、测量系数、训练采样、SAC、PPO、开发评价、检查点、日志和JSON IO。下面主列用36次动作／value步骤和6次KL的无早停外推；同时列出用完整PPO实测替换该项的预算。它们都是条件外推，普通/拥堵两列是上层工作量敏感性，不能当作置信区间或实际600轮墙钟。", "",
              "| 设置 | 训练＋开发：普通／拥堵小时 | 换用完整PPO实测：普通／拥堵小时 | 含5–15%余量的规划跨度 | 可选正式测试小时 |", "|---|---:|---:|---:|---:|"]
    for label,key in (("基线16","baseline"),("优化16，实际默认","optimized_micro16"),("优化32，仅候选","optimized_micro32")):
        lo,hi=budgets[key]["envelopes"].values()
        lines.append(f"| {label} | {lo['training_plus_development_hours']:.1f} / {hi['training_plus_development_hours']:.1f} | {lo['hours_using_measured_fixture_PPO_KL_behavior']:.1f} / {hi['hours_using_measured_fixture_PPO_KL_behavior']:.1f} | {lo['planning_hours_with_5_to_15_percent_reserve'][0]:.1f}–{hi['planning_hours_with_5_to_15_percent_reserve'][1]:.1f} h | {lo['optional_primary_test_hours']:.1f}–{hi['optional_primary_test_hours']:.1f} |")
    lines += ["", "检查点保存包含fsync及内部哈希，并补计608次latest身份哈希和7次评价身份哈希；哈希吞吐见 `throughput_extra_io.json`。0/4条episode检查点成本线性外推至55条回放及待完成PPO记录，累计占用按原日程逐轮计算。目标网络软更新、源码指纹、启动、垃圾回收、磁盘波动及未单列快照开销使用明确的5–15%规划余量；这不是统计误差带。", "",
              "采集项已含物理、报告和压缩；评价另加实际神经推理，避免遗漏采样或把独立分项再次加算。正式测试160条episode单列，其他检查点／确定性补充需另计。研究参数和有效更新预算未改，研究训练episode为0。", "",
              "### 复现与补丁", "",
              "基准入口为 `hrl_v2/benchmark_optimization.py`，测试入口为 `python -B -m hrl_v2.validate --skip-benchmark`；本轮完整记录脚本为 `v2_optimization/finish_validation.py`。`v2_optimization/seed_budget.py` 仅读取JSON外推，不启动训练。独立fixture生成器和SHA256记录随补丁提供，72MB本机fixture不嵌入补丁。", "",
              "基线源码必须来自Git `137d2f1`。此前重复轨迹、图缓存修复前及并发干扰的测量均不用于本表。交付补丁为 `v2_optimization/throughput_vs_137d2f1.patch`，未提交或推送。", ""]
    report=OUT/"throughput_report.md"
    prefix=report.read_text(encoding="utf-8").split("## 最终有界测量")[0].rstrip()
    report.write_text(prefix+"\n\n"+"\n".join(lines), encoding="utf-8")
    for name in ("tests.txt","equivalence_cuda_tests.txt","throughput_report.md"):
        path=OUT/name
        path.write_bytes(path.read_bytes().replace(b"\r\n",b"\n"))

    tracked=git("diff","--binary","137d2f1","--","hrl_v2").decode("utf-8")
    new=git("ls-files","--others","--exclude-standard","--","hrl_v2").decode("utf-8").splitlines()
    new += ["v2_optimization/"+name for name in ("make_independent_fixture.py","seed_budget.py","finish_validation.py","cold_inference.py","finalize_delivery.py")]
    pieces=[tracked]
    for relative in sorted(new):
        content=(ROOT/relative).read_text(encoding="utf-8").splitlines(keepends=True)
        pieces.append(f"diff --git a/{relative} b/{relative}\nnew file mode 100644\n")
        pieces.extend(difflib.unified_diff([],content,fromfile="/dev/null",tofile="b/"+relative))
    patch=WORK/"throughput_vs_137d2f1.patch"
    with patch.open("w",encoding="utf-8",newline="\n") as output:
        output.write("".join(pieces))
    check=WORK/"patch_check"
    for relative in git("ls-tree","-r","--name-only","137d2f1","--","hrl_v2").decode().splitlines():
        target=check/relative
        target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes(git("show",f"137d2f1:{relative}"))
    git("apply","--check","--directory=v2_optimization/patch_check",str(patch))
    git("diff","--check","--","hrl_v2")
    print(json.dumps(dict(tests=validation["tests"],cuda_tests=cuda["tests_run"],
                         patch=str(patch),patch_bytes=patch.stat().st_size,
                         budgets={key:{label:row["training_plus_development_hours"] for label,row in value["envelopes"].items()} for key,value in budgets.items()}),ensure_ascii=False))


if __name__=="__main__": main()
