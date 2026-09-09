# Overleaf使用

主文件：main.tex。编译器：XeLaTeX。

完整上传本目录的内容，不能仅上传主文件。各章在 sections/ 单独编辑，图片在 figures/，字体与版式在 communication-journal.sty。

图片已用 \includegraphics 引入；参考文献已内置，不需要 references.bbl，不需要联网下载图片或调用绘图程序，不需要 shell-escape。

该版式是面向《通信学报》的作者工作排版，不是编辑部提供的官方LaTeX类。正文10.5 PDF磅、16磅行距、双栏，图片170 mm通栏。字体已按参考PDF核对：中文标题SimHei、正文SimSun、中文作者STKaiti、英文Times New Roman。字体文件在 fonts/，绑定在 journal-fonts.tex，必须一并上传。数学仍使用XITS Math，不同时加载newtxmath。

首页独立，引言从下一页开始；作者与单位尚待真实填写，不为填满首页而虚构署名或扩写空泛摘要。整篇尚未在本机用XeLaTeX编译，建议在Overleaf完成编译后逐页核对长公式、浮动图位置与页数。

稿件依据当前67维PPO观测、68维SAC观测、18维直接分数及执行路由评论家编写。实验成绩尚不完整，正文未填入旧版联合结果。
