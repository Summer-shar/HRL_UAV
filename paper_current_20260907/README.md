# 当前模型论文与学术插图完整包

字体修正更新：前两页现已按参考PDF改为SimHei标题、SimSun正文、STKaiti作者名及Times New Roman英文，并修正字号和正文行距。请使用带 fontfix 后缀的新压缩包；字体说明见 font_fix_notes.md。本次只改排版，Markdown正文内容不变。

整理日期：2026-09-07。题目暂定为“多无人机协同计算的双时间尺度调度”，不超过20个汉字。

这是当前 PPO＋Direct SAC 模型的作者讨论稿，不是旧 v49/v67 结果的重新包装，也不是已经完成全部实验的投稿终稿。系统模型、目标、分层观测、时间折扣、算法、可行性与复杂度已写入正文；实验部分提供真实设置、指标和验证方案，没有编造待完成的联合训练成绩。

## 直接使用

- 上传单独提供的 paper_current_20260907_overleaf.zip 到 Overleaf，选择 XeLaTeX，主文件选择 main.tex，再编译。
- 若只解压了本完整包，上传 overleaf 文件夹里面的内容。不要只上传 main.tex。
- 阅读文字与公式：打开 manuscript.md。建议使用支持 MathJax 的 Markdown 编辑器，并开启数学公式。
- 直接看图：打开 figures_review.pdf，共5页，每页一张图；所有图已插入相应 LaTeX 正文，不是“占位图”。

整篇论文在本机没有完成 XeLaTeX 编译，因为没有检测到 TeX 编译器。交付前已进行文件、引用、环境配对、UTF-8及矢量图检查；图集PDF已渲染后逐页检查。不能将静态检查当作整篇论文的编译通过证明。若 Overleaf 首次编译出现排版溢出，优先检查单栏长公式，不要通过整体缩小正文字号处理。

## 单独修改哪里

main.tex 只有装配指令，不是内容丢失。分章节文件可以分别修改，不用每次重新发送完整包。

- 标题、作者、摘要：overleaf/sections/frontmatter.tex
- 引言：overleaf/sections/01_introduction.tex
- 场景、通信、计算模型：overleaf/sections/02_system_model.tex
- 优化问题、MDP、观测、奖励：overleaf/sections/03_problem_mdp.tex
- PPO、SAC、伪代码、复杂度：overleaf/sections/04_algorithm.tex
- 仿真设置、指标及待完成的验证方案：overleaf/sections/05_evaluation_protocol.tex
- 结束语：overleaf/sections/06_conclusion.tex
- 参考文献：overleaf/sections/references.tex
- 字体、栏宽、字号、间距：overleaf/communication-journal.sty
- 图片：overleaf/figures/，正文引用其中的PDF矢量版。

参考文献直接使用 thebibliography，不需要 references.bbl，不依赖 BibTeX。不要再添加 \input{references.bbl}。数学字体使用 unicode-math 与 XITS Math（提供回退），没有加载 newtxmath。Overleaf 必须选 XeLaTeX，不能选 pdfLaTeX。

## 五张图分别讲什么

图1：网络中有哪些对象，任务从哪上传，为什么允许机间协作。

图2：父任务如何经过完整上传、一次拆分、本地或远端计算，最后同步完成。

图3：10 s 大周期、0.5 s 资源帧、0.1 s 物理步及跨部署边界经验的关系。顶部时间轴按真实长度绘制。

图4：PPO与Direct SAC分别看什么、输出什么，如何通过同一个物理系统关联。

图5：SAC的原始分数、执行路由、评论家输入与熵项分别位于哪里。

参考论文的原图没有复制进包。图中对象与流程均按本项目重新绘制，借鉴的是黑白线稿、浅灰分组、短标签和箭头层级。为适配《通信学报》黑白印刷要求，采用中文标注；这比之前通用“顶会架构图”提示词中的全英文和彩色要求优先。

## 图源与再编辑

figure_source/draw_figures.py 是绘图源代码，使用 Python、NumPy 和 Matplotlib，不运行训练，不读取实验结果来生成虚构曲线。

每张图提供原生PDF、可编辑文字SVG、文字转轮廓SVG、240 dpi PNG。PDF及SVG均为实际矢量线条，不是将位图嵌入PDF。带 outlined 后缀的SVG适合跨机器显示；普通SVG保留文字便于修改，但接收方需有相应字体。

图宽170 mm，文字最小7.5 pt；中文优先华文中宋，英文Times New Roman。没有放入没有数据支撑的性能柱状图、收敛曲线或真实轨迹。

在本项目环境重画：

    .\.venv\Scripts\python.exe paper_current_20260907\figure_source\draw_figures.py

package_paper.py 用于从LaTeX生成Markdown、做静态核验并重新打包，需要Python、pypdf及Pandoc。它只生成本论文目录和同名ZIP，不修改模型代码或训练配置。字体修正版使用 build_font_fix.py 单独打包，保留原ZIP不覆盖。

## 投稿前仍需补充

真实作者、单位、通信作者、基金、中图分类号；正式联合训练结果；独立训练重复与未被开发过程反复使用的测试场景；核心消融与适配基线；参考文献扩充及最终版面核对。

当前摘要不虚构性能结论，末节也不宣称收敛。正文中的阶段完成状态提示、首页讨论稿标记需要在相应证据完整后再调整，不应先删除提示再把未完成稿当成实验证明。

具体代码对应关系和已归档诊断数据见 author_notes.md；它不作为论文正文引用。
