from collections import Counter
from pathlib import Path
import json
import pdfplumber
import sys
sys.path.append("D:/Anaconda3/envs/pytorch-gpu/lib/site-packages")
from fontTools.ttLib import TTFont, TTCollection

paths = [Path("D:/Chrome/paper_current_20260907_complete.pdf"),
         Path("C:/Users/PC/Desktop/基于多时间尺度协同的无蜂窝RAN切片资源分配算法_夏玮玮.pdf")]
for path in paths:
    print(path.name)
    with pdfplumber.open(path) as doc:
        for i,page in enumerate(doc.pages[:2],1):
            groups=Counter((c["fontname"],round(c["size"],2)) for c in page.chars)
            print("PAGE",i,groups.most_common(12))
            for font,size in list(groups)[:18]:
                chars=[c for c in page.chars if c["fontname"]==font and round(c["size"],2)==size]
                tops=sorted(set(round(c["top"],1) for c in chars))
                if len(chars)>40 or size>=12:
                    print(font,size,"sample","".join(c["text"] for c in chars)[:45],
                          "tops",tops[:14])
print("LOCAL FONT FLAGS")
for name in ["simsun.ttc","simhei.ttf","STKAITI.TTF","times.ttf","timesbd.ttf","timesi.ttf","timesbi.ttf"]:
    path=Path("C:/Windows/Fonts")/name
    font=TTFont(path,fontNumber=0,lazy=True)
    print(name, font["name"].getDebugName(1), font["name"].getDebugName(6),
          "fsType",font["OS/2"].fsType)
    font.close()
