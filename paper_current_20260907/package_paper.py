"""Generate a Markdown reading copy, audit inputs and package the paper.

Only writes inside this paper directory and its two named sibling ZIP files.
Does not execute any simulation or modify source configs.
"""
from pathlib import Path
import hashlib
import json
import re
import shutil
import subprocess
import zipfile
from collections import Counter
from pypdf import PdfReader

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
TEX = ROOT / "overleaf"
VERIFY = ROOT / "verification"
VERIFY.mkdir(exist_ok=True)
EVIDENCE = ROOT / "source_evidence"
EVIDENCE.mkdir(exist_ok=True)

main = (TEX / "main.tex").read_text(encoding="utf-8")
order = re.findall(r"\\input\{([^}]+)\}", main)
texts = [(name, (TEX / (name + ".tex")).read_text(encoding="utf-8")) for name in order]
combined = "\n\n".join(text for _, text in texts)
all_tex = main + "\n" + combined
labels = re.findall(r"\\label\{([^}]+)\}", combined)
duplicates = [x for x,n in Counter(labels).items() if n > 1]
refs = re.findall(r"\\(?:eqref|ref)\{([^}]+)\}", combined)
unresolved = sorted(set(refs) - set(labels))
bibkeys = re.findall(r"\\bibitem\{([^}]+)\}", combined)
cites = [k for block in re.findall(r"\\cite\{([^}]+)\}", combined) for k in block.split(",")]
unknown_cites = sorted(set(cites) - set(bibkeys))
unused_cites = sorted(set(bibkeys) - set(cites))
first_citations = list(dict.fromkeys(cites))
assert not duplicates, duplicates
assert not unresolved, unresolved
assert not unknown_cites, unknown_cites
assert not unused_cites, unused_cites
assert first_citations == bibkeys, (first_citations, bibkeys)

# Check environment nesting, not just equal counts.
stack = []
for match in re.finditer(r"\\(begin|end)\{([^}]+)\}", all_tex):
    action, name = match.groups()
    if action == "begin":
        stack.append(name)
    else:
        assert stack and stack[-1] == name, (stack, name)
        stack.pop()
assert not stack, stack
for name, text in texts:
    level = 0
    for token in re.findall(r"\\.|[{}]", re.sub(r"(?m)(?<!\\)%.*$", "", text)):
        if token == "{": level += 1
        if token == "}": level -= 1
        assert level >= 0, name
    assert level == 0, (name, level)
    assert "\ufffd" not in text and "\x00" not in text, name

fig_names = re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", combined)
assert len(fig_names) == 5, fig_names
pdf_checks = []
for name in fig_names:
    path = TEX / name
    assert path.is_file(), path
    reader = PdfReader(path)
    page = reader.pages[0]
    xobjects = page.get("/Resources", {}).get("/XObject", {})
    xobjects = xobjects.get_object() if hasattr(xobjects, "get_object") else xobjects
    images = [key for key, val in xobjects.items() if val.get_object().get("/Subtype") == "/Image"]
    assert not images, (name, images)
    assert len(reader.pages) == 1
    assert abs(float(page.mediabox.width) * 25.4 / 72 - 170) < .02
    pdf_checks.append({"file": name, "pages": 1, "width_mm": 170, "raster_images": len(images)})
assert len(PdfReader(ROOT / "figures_review.pdf").pages) == 5

# Build one global numbering map shared by text, math, figures and algorithms.
numbering = {}
for env in ["equation", "figure\\*", "algorithm\\*?"]:
    matches = list(re.finditer(r"\\begin\{(" + env + r")\}.*?\\end\{\1\}", combined, flags=re.S))
    for i, match in enumerate(matches, 1):
        for key in re.findall(r"\\label\{([^}]+)\}", match.group()):
            numbering[key] = str(i)
section_index = 0
for name, content in texts:
    for m in re.finditer(r"\\section\{[^}]+\}(?:\s*\\label\{([^}]+)\})?", content):
        section_index += 1
        if m.group(1): numbering[m.group(1)] = str(section_index)
assert set(refs) <= set(numbering), set(refs)-set(numbering)
bibmap = {key: str(i+1) for i,key in enumerate(bibkeys)}

def md_math_macros(text):
    text = re.sub(r"\\(?:vct|mat)\b", r"\\boldsymbol", text)
    text = re.sub(r"\\E\b", r"\\mathbb{E}", text)
    text = re.sub(r"\\R\b", r"\\mathbb{R}", text)
    text = re.sub(r"\\dd\b", r"\\mathrm{d}", text)
    text = re.sub(r"\\clip\b", r"\\operatorname{clip}", text)
    text = re.sub(r"\\diag\b", r"\\operatorname{diag}", text)
    text = re.sub(r"\\argmin\b", r"\\operatorname*{arg\\,min}", text)
    text = re.sub(r"\\argmax\b", r"\\operatorname*{arg\\,max}", text)
    text = re.sub(r"\\ind\{([^}]+)\}", lambda m:r"\mathbb{1}_{\{"+m[1]+r"\}}", text)
    return text

def convert_figure(match):
    block = match.group()
    path = re.search(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", block)[1]
    key = re.search(r"\\label\{([^}]+)\}", block)[1]
    caption = re.search(r"\\caption\{([^}]+)\}", block)[1]
    png = "overleaf/" + str(Path(path).with_suffix(".png")).replace("\\", "/")
    return "\n\\par\\includegraphics{" + png + "}\n\\par 图" + numbering[key] + "　" + caption + "\n\\par\n"

def convert_algorithm(match):
    block = match.group()
    key = re.search(r"\\label\{([^}]+)\}", block)[1]
    title = re.search(r"\\caption\{([^}]+)\}", block)[1]
    out = [r"\par 算法" + numbering[key] + "　" + title + r"\par", r"\begin{enumerate}"]
    for raw in block.splitlines():
        line = raw.strip()
        if line.startswith(r"\State "):
            out.append(r"\item " + line[len(r"\State "):])
        elif line.startswith(r"\Require "):
            out.append(r"\item 输入：" + line[len(r"\Require "):])
        elif line.startswith(r"\Ensure "):
            out.append(r"\item 输出：" + line[len(r"\Ensure "):])
        elif line.startswith(r"\For{"):
            out.append(r"\item 对" + line[5:-1] + "，执行以下循环：")
        elif line == r"\EndFor":
            out.append(r"\item 结束该循环。")
    out.append(r"\end{enumerate}")
    return "\n".join(out)

md_tex = combined
md_tex = re.sub(r"\\Journal(?:ChineseTitle|EnglishTitle|ChineseAuthor|EnglishAuthor|Affiliation|Abstract)\b", "", md_tex)
md_tex = re.sub(r"\\begin\{figure\*\}.*?\\end\{figure\*\}", convert_figure, md_tex, flags=re.S)
md_tex = re.sub(r"\\begin\{algorithm\*?\}.*?\\end\{algorithm\*?\}", convert_algorithm, md_tex, flags=re.S)
def number_equation(match):
    body = match[1]
    key = re.search(r"\\label\{([^}]+)\}", body)[1]
    body = re.sub(r"\\label\{[^}]+\}", "", body)
    return "\\[\n" + body.strip() + r"\tag{" + numbering[key] + "}\n\\]"
md_tex = re.sub(r"\\begin\{equation\}(.*?)\\end\{equation\}", number_equation, md_tex, flags=re.S)
md_tex = re.sub(r"\\eqref\{([^}]+)\}", lambda m:"(" + numbering[m[1]] + ")", md_tex)
md_tex = re.sub(r"\\ref\{([^}]+)\}", lambda m:numbering[m[1]], md_tex)
md_tex = re.sub(r"\\cite\{([^}]+)\}", lambda m:"[" + ",".join(bibmap[k] for k in m[1].split(",")) + "]", md_tex)
md_tex = re.sub(r"\\label\{[^}]+\}", "", md_tex)
md_tex = re.sub(r"\\begin\{thebibliography\}\{[^}]+\}", r"\\section*{参考文献}", md_tex)
md_tex = md_tex.replace(r"\end{thebibliography}", "")
md_tex = re.sub(r"\\bibitem\{([^}]+)\}", lambda m:r"\par \textnormal{["+bibmap[m[1]]+"]} ", md_tex)
md_tex = md_math_macros(md_tex)
combined_path = VERIFY / "reading_copy.tex"
combined_path.write_text(md_tex, encoding="utf-8")
pandoc = shutil.which("pandoc") or ("D:/pandoc.exe" if Path("D:/pandoc.exe").exists() else None)
assert pandoc, "Pandoc is required for Markdown generation"
proc = subprocess.run([pandoc, str(combined_path), "-f", "latex", "-t", "markdown+tex_math_dollars-raw_tex-fenced_divs-header_attributes-smart", "--wrap=none", "-o", str(ROOT/"manuscript.md")],
                      cwd=str(ROOT), capture_output=True, text=True, encoding="utf-8")
(VERIFY/"pandoc.log").write_text(proc.stdout + proc.stderr, encoding="utf-8")
assert proc.returncode == 0, proc.stderr
markdown = (ROOT/"manuscript.md").read_text(encoding="utf-8")
markdown = re.sub(r"</?div[^>]*>", "", markdown)
markdown = re.sub(r"\[(\\\[\d+\\\])\]\{\.nodecor\}", lambda m:m[1], markdown)
markdown = re.sub(r"\\\[([A-Za-z0-9/, \-]+)\\\]", lambda m:"["+m[1]+"]", markdown)
markdown = re.sub(r"\$\$(.*?)\$\$", lambda m:"\n\n$$\n"+m[1].strip()+"\n$$\n\n", markdown, flags=re.S)
markdown = re.sub(r"\n{3,}", "\n\n", markdown)
# Plain headings work in Typora as well as generic Markdown viewers.
section_no = subsection_no = 0
new_lines = []
for line in markdown.splitlines():
    if line.startswith("# "):
        subsection_no = 0
        if line[2:] != "参考文献":
            section_no += 1
            line = "## " + str(section_no) + " " + line[2:]
        else:
            line = "## 参考文献"
    elif line.startswith("## "):
        subsection_no += 1
        line = "### " + str(section_no) + "." + str(subsection_no) + " " + line[3:]
    new_lines.append(line)
markdown = "\n".join(new_lines).strip() + "\n"
markdown = "# " + markdown
markdown = markdown.replace("\n作者姓名待填写", "\n> 当前模型作者讨论稿；联合训练性能及正式对比仍待补充。\n\n作者姓名待填写", 1)
for i, name in enumerate(fig_names,1):
    stem = Path(name).stem
    markdown = markdown.replace("![image](overleaf/figures/"+stem+".png)", "![图"+str(i)+"](overleaf/figures/"+stem+".png)")
(ROOT/"manuscript.md").write_text(markdown,encoding="utf-8")
assert markdown.count("overleaf/figures/") == 5
assert markdown.count(r"\tag{") == 55
assert not re.search(r"\\(?:vct|mat|input|eqref)\b", markdown)
assert "<div" not in markdown and "{.nodecor}" not in markdown
assert markdown.startswith("# 多无人机协同计算的双时间尺度调度")
assert len(re.findall(r"(?m)^\[\d+\] ",markdown)) == len(bibkeys)
for link in re.findall(r"!\[[^\]]*\]\(([^ )]+)", markdown):
    assert (ROOT/link).is_file(), link

source_files = [
    "hrl_uav/env.py", "hrl_uav/mdp.py", "hrl_uav/geometry.py",
    "hrl_uav/routing.py", "hrl_uav/learning.py", "hrl_uav/experiment.py",
    "hrl_uav/agents/sac_agent.py", "configs/micro_budget300.json",
    "configs/micro_budget600.json",
]
source_hashes = {}
for rel in source_files:
    source = PROJECT / rel
    data = source.read_bytes()
    source_hashes[rel] = hashlib.sha256(data).hexdigest()
    if source.suffix == ".json":
        (EVIDENCE/source.name).write_bytes(data)
(EVIDENCE/"source_hashes.json").write_text(json.dumps(source_hashes, indent=2), encoding="utf-8")

front = dict(texts)["sections/frontmatter"]
abstract = front.split(r"\noindent\textbf{摘\quad 要：}",1)[1].split(r"\vspace",1)[0]
abstract_han = len(re.findall(r"[\u4e00-\u9fff]", abstract))
audit = {
    "date": "2026-09-07",
    "manuscript_han_characters": len(re.findall(r"[\u4e00-\u9fff]", combined)),
    "abstract_han_characters": abstract_han,
    "numbered_equations": len(re.findall(r"\\begin\{equation\}", combined)),
    "figures": len(fig_names), "algorithms": 2, "references": len(bibkeys),
    "duplicate_labels": duplicates, "unresolved_references": unresolved,
    "unknown_citations": unknown_cites, "unused_citations": unused_cites,
    "citation_order_matches_bibliography": first_citations == bibkeys,
    "environment_nesting": "passed", "brace_balance": "passed",
    "source_encoding": "UTF-8; no replacement characters",
    "pdf_assets": pdf_checks,
    "full_xelatex_compile": "NOT_RUN: no local compiler",
    "experiment_results": "not fabricated; formal joint comparison pending",
}
assert abstract_han <= 200, abstract_han
(VERIFY/"audit.json").write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding="utf-8")

overleaf_zip = PROJECT / "paper_current_20260907_overleaf.zip"
full_zip = PROJECT / "paper_current_20260907_complete.zip"
for output, root, thin in [(overleaf_zip,TEX,True),(full_zip,ROOT,False)]:
    with zipfile.ZipFile(output,"w",zipfile.ZIP_DEFLATED,compresslevel=9) as archive:
        for path in sorted(root.rglob("*")):
            if not path.is_file() or "__pycache__" in path.parts:
                continue
            if path.parent == VERIFY and path.name in ("reading_copy.tex","pandoc.log"):
                continue
            if thin and path.suffix.lower() in (".png",".svg"):
                continue
            archive.write(path,path.relative_to(root).as_posix())
    with zipfile.ZipFile(output) as archive:
        assert archive.testzip() is None
        if thin: assert "main.tex" in archive.namelist()
print(json.dumps(audit,ensure_ascii=False,indent=2))
print("Overleaf ZIP:",overleaf_zip,overleaf_zip.stat().st_size,"bytes")
print("Full ZIP:",full_zip,full_zip.stat().st_size,"bytes")
