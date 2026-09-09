from pathlib import Path
import json
from collections import defaultdict
from pypdf import PdfReader

ROOT = Path(__file__).resolve().parent
paths = {
    "ours": Path("D:/Chrome/paper_current_20260907_complete.pdf"),
    "reference": Path("C:/Users/PC/Desktop/基于多时间尺度协同的无蜂窝RAN切片资源分配算法_夏玮玮.pdf"),
}
report = {}
for name,path in paths.items():
    reader = PdfReader(path)
    pages = []
    for page in reader.pages[:2]:
        items=[]
        def visit(text, cm, tm, font, size):
            if text.strip():
                items.append({
                    "text": text.strip(), "font": str(font.get("/BaseFont")) if font else None,
                    "size": round(size,3), "tm": [round(x,3) for x in tm],
                    "cm": [round(x,3) for x in cm],
                })
        page.extract_text(visitor_text=visit)
        counts=defaultdict(int)
        examples=defaultdict(list)
        for item in items:
            key=(item["font"],item["size"],tuple(item["cm"][:4]),tuple(item["tm"][:4]))
            counts[key]+=len(item["text"])
            if sum(len(t) for t in examples[key])<90:
                examples[key].append(item["text"])
        fonts = sorted(set(item["font"] for item in items if item["font"]))
        pages.append({"page_size":list(page.mediabox),"fonts":fonts,"text":items,
                      "groups":[{"key":k,"count":n,"examples":examples[k]} for k,n in sorted(counts.items(),key=lambda x:-x[1])]})
    report[name]=pages
(ROOT/"font_audit.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
for name,pages in report.items():
    print(name)
    for i,page in enumerate(pages,1):
        print("PAGE",i,"SIZE",page["page_size"])
        print(json.dumps(page["fonts"],ensure_ascii=False))
        for group in page["groups"][:14]:
            print(json.dumps(group,ensure_ascii=False))
        print("START",json.dumps(page["text"][:13],ensure_ascii=False))
