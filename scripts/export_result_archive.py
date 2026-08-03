from __future__ import annotations

import csv
import hashlib
import json
import shutil
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from urllib.parse import quote


ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
ARCHIVE_DIR = ROOT / "result_archive"
CONFIGS_DIR = ROOT / "configs"
INCLUDED_EXTENSIONS = {".png", ".pdf", ".svg", ".csv", ".json", ".log", ".txt", ".md"}
EXCLUDED_EXTENSIONS = {".pt", ".npz", ".npy", ".pth", ".pkl"}
CURRENT_VERSION = "tcom_dynamic_u2u_sparse_v23_target_noise"
SOFT_POWER_NOTES = {
    "tcom_fixed_region_soft_power_v2": "软功率固定区域基线。",
    "tcom_fixed_region_soft_power_v3": "动作与功率反馈的第二轮基线修订。",
    "tcom_fixed_region_soft_power_v4_stick": "采用 stick-breaking 微观动作编码。",
    "tcom_fixed_region_soft_power_v5_alpha01": "调整 SAC 温度系数到 alpha=0.1。",
    "tcom_fixed_region_soft_power_v6_macro_signal": "强化宏观层可见的跨层反馈信号。",
    "tcom_fixed_region_soft_power_v7_centroid_waypoint": "将宏观动作改为相对业务质心的 waypoint。",
    "tcom_fixed_region_soft_power_v8_slow_dual": "降低能耗对偶变量更新速度。",
    "tcom_fixed_region_soft_power_v9_frame_power_mean": "按资源帧平均功率更新约束统计。",
    "tcom_fixed_region_soft_power_v10_long_term_credit": "加强部署间隙与长期队列的跨周期信用传递。",
    "tcom_fixed_region_soft_power_v11_full_action_coverage": "扩大宏观动作覆盖范围。",
    "tcom_fixed_region_soft_power_v12_clean_warmup": "清理 warm-up 数据与正式训练的边界。",
    "tcom_fixed_region_soft_power_v13_split_exploration": "拆分宏观与微观探索过程。",
    "tcom_fixed_region_soft_power_v14_staged_joint": "采用分阶段联合训练。",
    "tcom_fixed_region_soft_power_v15_two_timescale": "明确大小时间尺度更新协议。",
    "tcom_fixed_region_soft_power_v16_delayed_actors": "延迟 TD3/SAC Actor 更新以稳定 Critic。",
    "tcom_fixed_region_soft_power_v17_hinge_penalty": "使用功率违约 hinge penalty。",
    "tcom_fixed_region_soft_power_v18_conservative_joint_sac": "降低联合阶段 SAC 更新强度。",
    "tcom_fixed_region_soft_power_v19_balanced_penalty": "重新平衡时延与功率违约惩罚。",
}


@dataclass
class VersionSummary:
    version: str
    family: str
    status: str
    note: str
    last_modified: str
    archived_files: int
    images: int
    tables: int
    logs: int
    archived_size_mb: float
    omitted_checkpoints: int
    omitted_arrays: int
    omitted_size_mb: float


def version_metadata(version: str) -> tuple[str, str, str]:
    if version == CURRENT_VERSION:
        return (
            "动态用户与 U2U 协作",
            "当前诊断版本",
            "v23：缩小 TD3 target noise 并加入动作覆盖、位置参考和反事实诊断；宏观轨迹仍出现有害漂移。",
        )
    if version == "tcom_dynamic_u2u_sparse_v22_long1000":
        return ("动态用户与 U2U 协作", "历史长训练", "v22：1000 轮动态 U2U 训练，用于确认长期漂移并非训练轮数不足。")
    if version == "tcom_dynamic_u2u_sparse_v21_long600":
        return ("动态用户与 U2U 协作", "历史基线", "v21：稀疏 U2U 动态场景基线及微观预训练来源。")
    if version == "tcom_dynamic_u2u_v20_long600":
        return ("动态用户与 U2U 协作", "历史基线", "v20：动态 U2U 通信模型的首个长训练版本。")
    if version.startswith("tcom_fixed_region_soft_power_v"):
        return (
            "固定区域与软功率机制",
            "迭代诊断",
            SOFT_POWER_NOTES.get(version, "软功率与两时间尺度训练的历史迭代。")
            + " 各版本配置不同，只用于追踪设计演化。",
        )
    if version.startswith("tcom_fixed_region_reward"):
        return ("宏观奖励诊断", "诊断/消融", "固定区域下的宏观奖励信号、惩罚与冻结策略消融。")
    if version.startswith("tcom_communication_convergence"):
        return ("通信模型收敛验证", "历史验证", "切换到参考论文风格 U2U 动态带宽后的通信模型与收敛测试。")
    if version == "joint_hrl_v1":
        return (
            "早期动态用户 HRL",
            "历史轨迹参考",
            "早期曾出现 UAV-3 靠近 UAV-1 的清晰轨迹；协议较旧且缺少当前口径下的配对验证，不作为最终结论。",
        )
    if version.startswith("joint_hrl_v") or version == "joint_hrl_30step":
        return ("早期动态用户 HRL", "历史实验", "早期联合 TD3-SAC 训练与宏观周期设置迭代。")
    if version.startswith("joint_hrl_static"):
        return ("静态用户 HRL", "历史诊断", "静态用户条件下的 waypoint、动作参数化与宏观学习稳定性诊断。")
    if version == "diagnostics":
        return ("跨版本诊断", "诊断数据", "宏观奖励、反事实分叉、动作覆盖和动态质心位置扫描。")
    if version == "dynamic_load_position_audit":
        return ("位置价值审计", "诊断数据", "动态负载条件下的位置敏感度与参考策略审计。")
    if version in {"oracle_smoke", "position_reference_audit_smoke"}:
        return ("工具冒烟测试", "Smoke", "位置搜索或参考点审计工具的快速正确性测试。")
    return ("其他实验", "历史/辅助", "保留的历史实验结果；请结合目录内指标与对应配置解读。")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def markdown_link(path: Path, label: str | None = None) -> str:
    relative = path.as_posix()
    return f"[{label or path.name}]({quote(relative, safe='/._-')})"


def select_preview(image_paths: list[Path]) -> Path | None:
    priorities = [
        "trajectory_final.png",
        "trajectory_best.png",
        "training_convergence.png",
        "micro_convergence_complete.png",
    ]
    for name in priorities:
        for path in image_paths:
            if path.name == name:
                return path
    return image_paths[0] if image_paths else None


def write_version_readme(
    target_dir: Path,
    summary: VersionSummary,
    copied_paths: list[Path],
) -> None:
    images = sorted(path.relative_to(target_dir) for path in copied_paths if path.suffix.lower() in {".png", ".pdf", ".svg"})
    tables = sorted(path.relative_to(target_dir) for path in copied_paths if path.suffix.lower() in {".csv", ".json"})
    logs = sorted(path.relative_to(target_dir) for path in copied_paths if path.suffix.lower() in {".log", ".txt", ".md"})
    preview = select_preview([target_dir / path for path in images if path.suffix.lower() == ".png"])

    lines = [
        f"# {summary.version}",
        "",
        f"- **实验族**：{summary.family}",
        f"- **状态**：{summary.status}",
        f"- **说明**：{summary.note}",
        f"- **归档内容**：{summary.archived_files} 个文件，{summary.archived_size_mb:.2f} MB",
        f"- **未归档大文件**：{summary.omitted_checkpoints} 个检查点、{summary.omitted_arrays} 个数组，共 {summary.omitted_size_mb:.2f} MB",
        "",
        "> 不同版本的场景、奖励、时隙或通信模型可能不同。除非总 README 明确说明，否则不要直接比较不同版本的绝对数值。",
        "",
    ]
    if preview is not None:
        relative_preview = preview.relative_to(target_dir).as_posix()
        lines.extend(["## 预览", "", f"![result preview]({quote(relative_preview, safe='/._-')})", ""])

    for heading, paths in (("图片", images), ("指标数据", tables), ("日志", logs)):
        lines.extend([f"## {heading}", ""])
        if paths:
            lines.extend(f"- {markdown_link(path)}" for path in paths)
        else:
            lines.append("- 无")
        lines.append("")

    (target_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")


def archive_version(source_dir: Path) -> tuple[VersionSummary, list[dict[str, object]]]:
    version = source_dir.name
    family, status, note = version_metadata(version)
    target_dir = ARCHIVE_DIR / version
    target_dir.mkdir(parents=True, exist_ok=True)

    all_files = [path for path in source_dir.rglob("*") if path.is_file()]
    selected = [path for path in all_files if path.suffix.lower() in INCLUDED_EXTENSIONS]
    omitted = [path for path in all_files if path.suffix.lower() in EXCLUDED_EXTENSIONS]
    copied_paths: list[Path] = []
    manifest_rows: list[dict[str, object]] = []

    for source_path in selected:
        relative = source_path.relative_to(source_dir)
        target_path = target_dir / relative
        target_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, target_path)
        copied_paths.append(target_path)
        manifest_rows.append(
            {
                "version": version,
                "archive_path": target_path.relative_to(ARCHIVE_DIR).as_posix(),
                "source_path": source_path.relative_to(ROOT).as_posix(),
                "extension": source_path.suffix.lower(),
                "size_bytes": source_path.stat().st_size,
                "sha256": sha256(target_path),
            }
        )

    image_count = sum(path.suffix.lower() in {".png", ".pdf", ".svg"} for path in selected)
    table_count = sum(path.suffix.lower() in {".csv", ".json"} for path in selected)
    log_count = sum(path.suffix.lower() in {".log", ".txt", ".md"} for path in selected)
    latest_mtime = max((path.stat().st_mtime for path in all_files), default=source_dir.stat().st_mtime)
    summary = VersionSummary(
        version=version,
        family=family,
        status=status,
        note=note,
        last_modified=datetime.fromtimestamp(latest_mtime).strftime("%Y-%m-%d %H:%M"),
        archived_files=len(selected),
        images=image_count,
        tables=table_count,
        logs=log_count,
        archived_size_mb=round(sum(path.stat().st_size for path in selected) / (1024 * 1024), 2),
        omitted_checkpoints=sum(path.suffix.lower() in {".pt", ".pth"} for path in omitted),
        omitted_arrays=sum(path.suffix.lower() in {".npz", ".npy", ".pkl"} for path in omitted),
        omitted_size_mb=round(sum(path.stat().st_size for path in omitted) / (1024 * 1024), 2),
    )
    write_version_readme(target_dir, summary, copied_paths)
    return summary, manifest_rows


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_root_readme(summaries: list[VersionSummary], commit: str) -> None:
    total_archived = sum(item.archived_size_mb for item in summaries)
    total_omitted = sum(item.omitted_size_mb for item in summaries)
    rows = []
    for item in summaries:
        rows.append(
            "| "
            + " | ".join(
                [
                    f"[{item.version}]({quote(item.version, safe='._-')}/)",
                    item.family,
                    item.status,
                    str(item.images),
                    str(item.tables),
                    f"{item.archived_size_mb:.2f}",
                    item.note.replace("|", "/"),
                ]
            )
            + " |"
        )

    lines = [
        "# HRL-UAV 全版本实验结果归档",
        "",
        f"本目录整合了 `results/` 中现存的 **{len(summaries)} 个顶层实验版本**。导出基于提交 `{commit[:12]}`。",
        "",
        "## 归档范围",
        "",
        "- 已归档：全部 PNG/PDF/SVG 图、CSV/JSON 指标和文本日志。",
        "- 未归档：`.pt` 模型检查点、`.npz/.npy` 原始数组及其他大二进制文件。",
        f"- 已归档体积约 **{total_archived:.2f} MB**；本地未归档大文件约 **{total_omitted / 1024:.2f} GB**。",
        "- `archive_manifest.csv` 记录每个归档文件的来源、大小和 SHA-256；`versions.csv` 是版本总表。",
        "",
        "> 重要：这些版本是研发过程记录，不是同一实验协议下的正式横向对比。场景、奖励、动作编码、训练轮数和通信模型在版本间发生过变化。",
        "",
        "## 当前结论",
        "",
        "- 当前诊断版本为 `tcom_dynamic_u2u_sparse_v23_target_noise`。",
        "- 微观 SAC 能正常完成协作转发；宏观 TD3 在 400 轮中出现 U2G 距离增大和时延恶化，尚未形成最终可用轨迹。",
        "- 20-seed 位置参考审计表明：动态跟踪相对固定锚点降低约 40.5 ms，但动态质心、上行参考点和联合位置搜索之间差异不足 1 ms。",
        "- `joint_hrl_v1` 中 UAV-3 靠近 UAV-1 的轨迹是历史上最清晰的协作轨迹，但协议较旧，不能替代当前口径下的正式验证。",
        "",
        "## 推荐阅读顺序",
        "",
        "1. [`tcom_dynamic_u2u_sparse_v23_target_noise`](tcom_dynamic_u2u_sparse_v23_target_noise/)：当前训练与轨迹漂移。",
        "2. [`diagnostics`](diagnostics/)：反事实、奖励与位置扫描诊断。",
        "3. [`joint_hrl_v1`](joint_hrl_v1/)：历史上较清晰的协作轨迹与早期收敛图。",
        "4. [`tcom_dynamic_u2u_v20_long600`](tcom_dynamic_u2u_v20_long600/) 到 v23：当前通信模型的演化链。",
        "5. `tcom_fixed_region_soft_power_v2` 到 v19：固定区域与功率机制的历史迭代，仅用于追踪研发过程。",
        "",
        "## 版本索引",
        "",
        "| 版本 | 实验族 | 状态 | 图片 | CSV/JSON | MB | 说明 |",
        "|---|---|---:|---:|---:|---:|---|",
        *rows,
        "",
        "## 重新生成",
        "",
        "```powershell",
        "D:\\Anaconda3\\envs\\pytorch-gpu\\python.exe scripts\\export_result_archive.py",
        "```",
        "",
        "如需在另一台机器继续训练，应从 GitHub 获取代码与本归档，再单独复制本地 `.pt` 检查点；Git 仓库不承担模型权重存储。",
    ]
    (ARCHIVE_DIR / "README.md").write_text("\n".join(lines), encoding="utf-8")


def copy_configs() -> None:
    target = ARCHIVE_DIR / "_configs"
    target.mkdir(parents=True, exist_ok=True)
    for config in sorted(CONFIGS_DIR.glob("*.json")):
        shutil.copy2(config, target / config.name)


def main() -> None:
    if not RESULTS_DIR.is_dir():
        raise FileNotFoundError(f"Missing results directory: {RESULTS_DIR}")

    archive_resolved = ARCHIVE_DIR.resolve()
    if archive_resolved.parent != ROOT.resolve() or archive_resolved.name != "result_archive":
        raise RuntimeError(f"Refusing to replace unexpected archive path: {archive_resolved}")
    if ARCHIVE_DIR.exists():
        shutil.rmtree(ARCHIVE_DIR)
    ARCHIVE_DIR.mkdir(parents=True)

    summaries: list[VersionSummary] = []
    manifest: list[dict[str, object]] = []
    for source_dir in sorted(path for path in RESULTS_DIR.iterdir() if path.is_dir()):
        summary, rows = archive_version(source_dir)
        summaries.append(summary)
        manifest.extend(rows)

    copy_configs()
    commit = git_commit()
    write_root_readme(summaries, commit)
    write_csv(
        ARCHIVE_DIR / "versions.csv",
        [asdict(item) for item in summaries],
        list(asdict(summaries[0]).keys()),
    )
    write_csv(
        ARCHIVE_DIR / "archive_manifest.csv",
        manifest,
        ["version", "archive_path", "source_path", "extension", "size_bytes", "sha256"],
    )
    summary_payload = {
        "source_commit": commit,
        "version_count": len(summaries),
        "archived_file_count": len(manifest),
        "archived_size_mb": round(sum(item.archived_size_mb for item in summaries), 2),
        "omitted_large_file_size_mb": round(sum(item.omitted_size_mb for item in summaries), 2),
        "included_extensions": sorted(INCLUDED_EXTENSIONS),
        "excluded_extensions": sorted(EXCLUDED_EXTENSIONS),
    }
    (ARCHIVE_DIR / "archive_summary.json").write_text(
        json.dumps(summary_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary_payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
