# 前两页字体修正

本次修改仅涉及字体、字号及行距，不改变论文模型、公式含义或实验数据。原排版文件备份在 font_fix_backup/，用户上传的PDF未被覆盖。

## 根据两份实际PDF确认的差异

| 项目 | 用户编译的上一版 | 师兄论文前两页 | 本次设置 |
|---|---|---|---|
| 中文主标题 | FandolHei Regular，约17.93 PDF磅 | SimHei，16磅 | SimHei，16bp |
| 中文作者 | FandolSong，约11.96磅 | STKaiti，10.5磅 | 华文楷体，10.5bp |
| 单位文字 | FandolSong，约10.46磅 | SimSun，7.5磅 | 宋体，7.5bp |
| 中文摘要 | FandolSong，约10.46磅 | SimSun，9磅 | 宋体，9bp |
| 英文标题 | TeX Gyre Termes Bold，约14.94磅 | Times New Roman Bold，16磅 | Times New Roman粗体，16bp |
| 正文中文 | FandolSong，约10.46磅 | SimSun，10.5磅 | 宋体，10.5bp |
| 正文英文 | TeX Gyre Termes | Times New Roman | Times New Roman |
| 正文行距 | 约19.69磅 | 约16磅 | 显式16bp |

字体名称、字号来自PDF字体对象和逐字符布局信息，不是根据截图外观猜测。正文行距原先写成15.2pt，但叠加了ctexart默认的1.3倍行距，实际约为19.69 PDF磅；本次显式将行距倍率设为1。

bp是1/72英寸，等于PDF/PostScript磅。TeX的pt是1/72.27英寸，因此直接写16pt并不严格等于PDF中的16磅。本次按实际PDF测量值设置。

## 本次修改

- main.tex：取消 fontset=fandol，改为 fontset=none。
- journal-fonts.tex：新增显式字体文件绑定及首页各类文字的字号命令。
- communication-journal.sty：加载上述字体，修正标题层级和正文行距。
- sections/frontmatter.tex：分别使用中文标题、楷体作者、单位、摘要、英文标题等样式。
- fonts/：随私人写作工程带入本机字体，避免只写“SimSun”而Overleaf无法找到。

数学字体XITS Math未改动。师兄第二页主体是引言，没有足够公式用来确定全部数学字形的精确替换；本次不再凭外观任意切换数学字体。

## 使用与核验边界

使用带 fontfix 后缀的Overleaf压缩包，完整上传，选择XeLaTeX。不要只覆盖样式文件而漏传 journal-fonts.tex 或 fonts/。已有旧工程的用户可整体上传新ZIP，避免缓存旧字体设置。

完整论文的内容、作者占位信息和A4纸张大小保持原设定。参考PDF的出版裁切页高约285 mm，而本工程为297 mm；两篇内容长度也不同。因此，本次对齐字体及行距，不宣称两篇每个换行、页高和段落坐标完全一致。

本机仍无XeLaTeX，未将字体核样冒充整篇论文编译结果。font_match_proof.pdf是采用同一组实际字体生成的字形核样，用于检查标题、作者、摘要和正文的字体层级；正式分页需以Overleaf重新编译结果为准。
