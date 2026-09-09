"""Create a font proof and two source archives, leaving the user's PDFs intact."""
from pathlib import Path
import hashlib
import json
import re
import zipfile
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Paragraph
from reportlab.lib.styles import ParagraphStyle
from pypdf import PdfReader

ROOT=Path(__file__).resolve().parent
TEX=ROOT/"overleaf"
OUT=ROOT/"output"/"pdf"
OUT.mkdir(parents=True,exist_ok=True)
assets={
    "Song":"simsun.ttc","Hei":"simhei.ttf","Kai":"STKAITI.TTF",
    "TimesR":"times.ttf","TimesB":"timesbd.ttf",
    "TimesI":"timesi.ttf","TimesBI":"timesbi.ttf",
}
for alias,file in assets.items():
    pdfmetrics.registerFont(TTFont(alias,str(TEX/"fonts"/file),subfontIndex=0))
pdfmetrics.registerFontFamily("Song",normal="Song",bold="Hei",italic="Kai",boldItalic="Hei")
pdfmetrics.registerFontFamily("TimesR",normal="TimesR",bold="TimesB",italic="TimesI",boldItalic="TimesBI")

path=OUT/"font_match_proof.pdf"
c=canvas.Canvas(str(path),pagesize=(595.28,841.89))
c.setTitle("前两页字体核样（非整篇LaTeX编译件）")
c.setAuthor("Author font proof")
def center(text,font,size,y):
    c.setFont(font,size);c.drawCentredString(297.64,y,text)
def label(text,y):
    c.setFont("Song",8);c.setFillGray(.4);c.drawString(57,y,text);c.setFillGray(0)
def paragraph(text,font,size,leading,top,width=481.9,x=56.69):
    p=Paragraph(text,ParagraphStyle("proof",fontName=font,fontSize=size,leading=leading,
                wordWrap="CJK" if font in ("Song","Hei","Kai") else None,
                firstLineIndent=2*size if size==10.5 else 0))
    _,height=p.wrap(width,800)
    p.drawOn(c,x,top-height)
    return top-height
label("前两页字体核样：已匹配实际字体；本页不是完整论文的XeLaTeX编译结果",808)
c.setLineWidth(.35);c.line(56.69,796,538.59,796)
label("中文主标题：SimHei（黑体）16磅，不额外伪加粗",765)
center("多无人机协同计算的双时间尺度调度","Hei",16,731)
label("中文作者：STKaiti（华文楷体）10.5磅；单位：SimSun（宋体）7.5磅",699)
center("作者姓名待填写","Kai",10.5,677)
center("（作者单位全称，省市 邮政编码待填写）","Song",7.5,659)
label("中文摘要：SimSun（宋体）9磅，行距14.67磅",629)
y=paragraph("<b>摘　要：</b>针对动态热点下多无人机计算负载失衡及部署中断问题，建立包含上传、转发、计算和分片同步完成的任务级模型。大时间尺度调整部署位置，小时间尺度调整任务路由，并用共同物理代价描述任务等待与能量消耗。","Song",9,14.67,610)
label("英文标题：Times New Roman Bold，16磅",y-29)
center("Two-Timescale Scheduling for","TimesB",16,y-56)
center("Cooperative Multi-UAV Computing","TimesB",16,y-77)
label("正文标题：SimHei（黑体）12磅；序号：Times New Roman Bold",y-113)
c.setFont("TimesB",12);c.drawString(56.69,y-137,"1")
c.setFont("Hei",12);c.drawString(79,y-137,"引言")
label("正文：SimSun＋Times New Roman，10.5磅；行距16磅",y-169)
body_lines=[
    "移动边缘计算将服务部署在用户附近，",
    "减少任务处理对远端云平台的依赖。",
    "无人机能够调整服务位置，为移动用户",
    "群提供计算接入。热点迁移可能造成",
    "局部负载过高，而其他无人机仍有空闲",
    "算力，需要通过机间协作进行调度。",
]
c.setFont("Song",10.5)
for column_x in (56.69,307.56):
    for line_no,line_text in enumerate(body_lines):
        c.drawString(column_x+(21 if line_no==0 else 0),y-198-line_no*16,line_text)
y2=y-198-(len(body_lines)-1)*16
paragraph("Mobile edge computing (MEC), unmanned aerial vehicle (UAV), proximal policy optimization (PPO), and soft actor-critic (SAC).","TimesR",10.5,16,y2-18)
label("数学字体保持XITS Math；本次未改动算法、公式含义、实验配置或结果。",65)
c.save()
reader=PdfReader(path)
assert len(reader.pages)==1
actual={str(v.get_object().get("/BaseFont")) for v in reader.pages[0]["/Resources"]["/Font"].get_object().values()}
for expected in ("SimHei","SimSun","STKaiti","TimesNewRomanPSMT","TimesNewRomanPS-BoldMT"):
    assert any(expected in f for f in actual),(expected,actual)

main=(TEX/"main.tex").read_text(encoding="utf-8")
style=(TEX/"communication-journal.sty").read_text(encoding="utf-8")
assert "fontset=none" in main and "fontset=fandol" not in main
assert r"\linespread{1}" in style and r"\fontsize{10.5bp}{16bp}" in style
binding=(TEX/"journal-fonts.tex").read_text(encoding="utf-8")
for file in assets.values():
    assert file in binding and (TEX/"fonts"/file).exists()
content_files=list((TEX/"sections").glob("*.tex"))
alltext="\n".join(p.read_text(encoding="utf-8") for p in content_files)
labels=set(re.findall(r"\\label\{([^}]+)\}",alltext))
refs=set(re.findall(r"\\(?:eqref|ref)\{([^}]+)\}",alltext))
assert refs<=labels,refs-labels
audit={
    "scope":"Fonts, sizes and leading only; original PDFs unchanged",
    "reference_fonts":["SimHei","SimSun","STKaiti","Times New Roman"],
    "old_body_leading_pdf_pt":19.69,"new_body_leading_pdf_pt":16,
    "font_sha256":{f:hashlib.sha256((TEX/"fonts"/f).read_bytes()).hexdigest() for f in assets.values()},
    "proof_embedded_fonts":sorted(actual),
    "full_xelatex_compile":"NOT_RUN",
}
(ROOT/"verification"/"font_fix_audit.json").write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding="utf-8")
for name,root,thin in [
    ("paper_current_20260907_overleaf_fontfix.zip",TEX,True),
    ("paper_current_20260907_complete_fontfix.zip",ROOT,False),
]:
    output=ROOT.parent/name
    with zipfile.ZipFile(output,"w",zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for f in sorted(root.rglob("*")):
            if not f.is_file() or "__pycache__" in f.parts or "font_fix_backup" in f.parts:continue
            if f.name in ("reading_copy.tex","pandoc.log"):continue
            if thin and f.suffix.lower() in (".png",".svg"):continue
            z.write(f,f.relative_to(root).as_posix())
    with zipfile.ZipFile(output) as z:
        assert z.testzip() is None
        if thin: assert "journal-fonts.tex" in z.namelist() and "fonts/simsun.ttc" in z.namelist()
    print(output.name,round(output.stat().st_size/1024/1024,2),"MiB")
print(json.dumps(audit,ensure_ascii=False))
