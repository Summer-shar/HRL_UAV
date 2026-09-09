"""Original journal-style vector diagrams; no experiment or training is run.

Run with Python + matplotlib + numpy. Chinese font falls back to SimSun.
Coordinates are millimetres; all figure lettering is at least 7.5 points.
"""
from pathlib import Path
import json
import math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.patches import Rectangle, Ellipse, Circle, FancyArrowPatch, Polygon
from matplotlib.backends.backend_pdf import PdfPages

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'overleaf' / 'figures'
OUT.mkdir(parents=True, exist_ok=True)
for name in ['STZHONGS.TTF', 'simsun.ttc', 'times.ttf']:
    path = Path('C:/Windows/Fonts') / name
    if path.exists():
        fm.fontManager.addfont(str(path))
matplotlib.rcParams.update({
    'font.family': ['Times New Roman', 'STZhongsong', 'SimSun', 'DejaVu Sans'],
    'font.size': 7.5, 'axes.unicode_minus': False, 'pdf.fonttype': 42,
    'svg.fonttype': 'none', 'hatch.linewidth': .45, 'savefig.facecolor': 'white',
})
INK = '#202020'
figures = []
texts = []

def canvas(height):
    f = plt.figure(figsize=(170/25.4, height/25.4), facecolor='white')
    ax = f.add_axes([0,0,1,1], xlim=(0,170), ylim=(0,height))
    ax.set_aspect('equal'); ax.axis('off')
    return f, ax

def label(ax,x,y,s,size=7.5,align='center',weight='normal',**kw):
    t = ax.text(x,y,s,fontsize=size,ha=align,va='center',color=INK,
                linespacing=1.35,weight=weight,**kw)
    texts.append(t)
    return t

def box(ax,x,y,w,h,s='',fill='white',dash=False,hatch=None):
    p=Rectangle((x,y),w,h,facecolor=fill,edgecolor=INK,lw=.65,
                linestyle=(0,(4,2)) if dash else '-',hatch=hatch)
    ax.add_patch(p)
    if s: label(ax,x+w/2,y+h/2,s)
    return p

def arrow(ax,points,dash=False,two=False,lw=.75):
    for p,q in zip(points[:-2],points[1:-1]):
        ax.plot([p[0],q[0]],[p[1],q[1]],color=INK,lw=lw,ls=(0,(3,2)) if dash else '-')
    ax.add_patch(FancyArrowPatch(points[-2],points[-1],
        arrowstyle='<->' if two else '-|>',mutation_scale=7,
        linewidth=lw,color=INK,linestyle=(0,(3,2)) if dash else '-',shrinkA=0,shrinkB=0))

def line(ax,points,dash=False,lw=.6):
    ax.plot(*zip(*points),color=INK,lw=lw,ls=(0,(3,2)) if dash else '-')

def person(ax,x,y,filled=False):
    ax.add_patch(Circle((x,y+1.8),.8,fc='0.45' if filled else 'white',ec=INK,lw=.55))
    ax.add_patch(Ellipse((x,y),2.5,2.5,fc='0.75' if filled else 'white',ec=INK,lw=.55))

def drone(ax,x,y):
    for dx,dy in [(-3,-1.8),(-3,1.8),(3,-1.8),(3,1.8)]:
        line(ax,[(x,y),(x+dx,y+dy)],lw=.8)
        ax.add_patch(Ellipse((x+dx,y+dy),3.2,1.3,fc='white',ec=INK,lw=.65))
    ax.add_patch(Ellipse((x,y),2.8,1.8,fc='.8',ec=INK,lw=.65))

def queue(ax,x,y,n,label_text):
    label(ax,x-1.5,y+1.5,label_text,align='right')
    for i in range(5): box(ax,x+2.6*i,y,2.6,3,fill='.65' if i<n else 'white')

def save(f,name,title):
    f.canvas.draw()
    f.savefig(OUT/(name+'.pdf'))
    f.savefig(OUT/(name+'.svg'))
    f.savefig(OUT/(name+'_outlined.svg'),backend='svg')
    with matplotlib.rc_context({'svg.fonttype':'path'}):
        f.savefig(OUT/(name+'_outlined.svg'))
    f.savefig(OUT/(name+'.png'),dpi=240)
    figures.append((f,name,title))

# 1. Physical scene. Icons are original line drawings, not lifted from reference.
f,a=canvas(103)
label(a,85,97,'机间协作：6 MHz 总带宽，由当前非空有向链路共享',weight='bold')
for x,n,nh,nl in [(30,1,3,5),(85,2,2,3),(140,3,1,2)]:
    drone(a,x,80); label(a,x,73,f'无人机 {n}')
    box(a,x-19,51,38,16,fill='.97')
    queue(a,x-6,60,nh,'高优'); queue(a,x-6,54,nl,'低优')
    label(a,x,46,'本地计算队列')
    a.add_patch(Ellipse((x,22),43,24,fc='.95',ec=INK,lw=.65,ls=(0,(4,2))))
    for dx,dy,hi in [(-10,1,True),(-3,5,False),(6,1,True),(11,-5,False),(-5,-5,False)]:
        person(a,x+dx,22+dy,False)
    label(a,x,7,f'用户群 {n}（固定源关联）')
    arrow(a,[(x,35),(x,43)])
    label(a,x+3,39,'上传',align='left')
arrow(a,[(36,80),(78,80)],two=True)
arrow(a,[(92,80),(134,80)],two=True)
arrow(a,[(30,84),(30,90),(140,90),(140,84)])
arrow(a,[(30,90),(30,84)])
arrow(a,[(53,20),(62,20)],dash=True)
arrow(a,[(108,20),(117,20)],dash=True)
label(a,57.5,28,'热点迁移')
label(a,112.5,28,'热点迁移')
label(a,85,1.7,'实线：任务传输　　虚线：场景或业务变化；示意位置不代表训练轨迹')
save(f,'fig1_system','多无人机协同计算场景')

# 2. Parent task and branch synchronization.
f,a=canvas(83)
label(a,85,77,'父任务只拆分一次；所有非零分片完成后，父任务才完成',weight='bold')
box(a,2,36,22,13,'任务生成\n高优 / 低优')
box(a,32,36,24,13,'上传队列\n完整上传')
box(a,64,36,24,13,'一次路由\n分片生成',fill='.92')
arrow(a,[(24,42.5),(32,42.5)]); arrow(a,[(56,42.5),(64,42.5)])
box(a,100,58,30,12,'源无人机计算')
box(a,100,35,30,12,'转发队列\n完整机间传输')
box(a,100,12,30,12,'目的无人机计算')
arrow(a,[(88,42.5),(93,42.5),(93,64),(100,64)])
arrow(a,[(88,42.5),(100,41)])
arrow(a,[(115,35),(115,24)])
label(a,110,29.5,'下一物理步',align='right')
box(a,141,35,27,15,'全部分片完成\n父任务完成',fill='.9')
arrow(a,[(130,64),(135,64),(135,46),(141,46)])
arrow(a,[(130,18),(135,18),(135,38),(141,38)])
label(a,28,61,'等待与上行传输')
label(a,76,57,'当前帧路由')
label(a,47,18,'后续路由动作仅影响新拆分任务\n已分配分片按原路径继续服务')
label(a,85,3.5,'上传、排队、转发和计算均计入任务驻留；高优任务先获得服务')
save(f,'fig2_taskflow','父任务处理路径与分片同步完成')

# 3. Time geometry uses a genuinely proportional 20-second top strip.
f,a=canvas(88)
label(a,85,86,'大周期 10 s：部署 1 s + 服务 9 s',weight='bold')
x0=17; scale=7
for k in range(2):
    base=x0+k*10*scale
    for j in range(20):
        box(a,base+j*.5*scale,58,.5*scale,9,
            fill='.82' if j<2 else 'white',hatch='///' if j<2 else None)
    label(a,base+5*scale,72,f'第 t{ "+1" if k else "" } 个大周期')
    arrow(a,[(base,78),(base,67)])
    label(a,base,80,'PPO',align='left')
for t in [0,1,10,11,20]: label(a,x0+t*scale,54,str(t))
label(a,85,50,'时间/s')
label(a,49,46,'服务帧 0.5 s：SAC 决策一次')
label(a,125,46,'每帧含 5 个 0.1 s 物理步')
for x,s,h in [(10,'周期末服务帧',False),(48,'部署帧 1',True),(86,'部署帧 2',True),(124,'下一服务帧',False)]:
    box(a,x,26,32,11,s,fill='.85' if h else 'white',hatch='///' if h else None)
    label(a,x+16,22,'0.5 s')
for x in [42,80,118]: arrow(a,[(x,31.5),(x+6,31.5)])
line(a,[(10,18),(10,14),(124,14),(124,18)],dash=True)
label(a,67,10,'一条跨边界经验：回报累加三帧，后继折扣为 γ³')
label(a,85,3,'部署期间：任务到达、用户移动继续；上传、转发与计算暂停')
save(f,'fig3_timescale','双时间尺度时序与部署边界转移')

# 4. Two observations from one system; separate experience stores.
f,a=canvas(110)
box(a,4,68,162,36,dash=True)
label(a,8,100,'大时间尺度：每 10 s 决策',align='left',weight='bold')
box(a,9,77,32,14,'部署观测\n67 维')
box(a,50,77,27,14,'PPO\n6 维分数',fill='.91')
box(a,87,77,35,14,'协作方向与\n可行位移映射')
box(a,132,77,29,14,'部署位置\n服务期保持')
for x,y in [(41,50),(77,87),(122,132)]:arrow(a,[(x,84),(y,84)])
box(a,4,26,162,36,dash=True)
label(a,8,58,'小时间尺度：每 0.5 s 服务帧决策',align='left',weight='bold')
box(a,9,35,32,14,'路由观测\n68 维')
box(a,50,35,27,14,'Direct SAC\n18 维分数',fill='.91')
box(a,87,35,35,14,'逐行单纯形投影\n6 行合法比例')
box(a,132,35,29,14,'任务路由\n帧内保持')
for x,y in [(41,50),(77,87),(122,132)]:arrow(a,[(x,42),(y,42)])
arrow(a,[(146.5,77),(146.5,62)])
label(a,115,65,'位置改变链路条件')
box(a,9,5,152,14,'同一个物理系统：用户、任务、队列、链路与计算服务',fill='.95')
arrow(a,[(146.5,35),(146.5,19)])
arrow(a,[(9,12),(1.5,12),(1.5,84),(9,84)],dash=True)
arrow(a,[(1.5,42),(9,42)],dash=True)
label(a,85,23,'等待与能耗按决策间隔聚合，分别形成两层经验')
save(f,'fig4_framework','PPO 与 Direct SAC 的双时间尺度协同框架')

# 5. Distinguish raw score entropy and executed-route critic.
f,a=canvas(103)
label(a,85,97,'SAC：原始分数采样，执行路由评价',weight='bold')
box(a,4,71,27,14,'路由观测\n68 维')
box(a,40,71,29,14,'Actor\n高斯 + tanh',fill='.92')
box(a,79,71,29,14,'连续分数 z\n18 维')
box(a,119,71,46,14,'逐行欧氏投影\n路由 ρ = P(z)')
for x,y in [(31,40),(69,79),(108,119)]:arrow(a,[(x,78),(y,78)])
box(a,120,40,45,15,'任务级环境\n执行、回报、后继观测')
arrow(a,[(142,71),(142,55)])
box(a,65,40,40,15,'经验回放\n观测、分数、路由、折扣')
arrow(a,[(120,47.5),(105,47.5)])
box(a,7,40,43,15,'双 Critic\n输入观测与执行路由',fill='.92')
arrow(a,[(65,47.5),(50,47.5)])
box(a,8,9,58,15,'Actor 更新\n价值梯度经过路由投影')
box(a,99,9,65,15,'熵与温度更新\n使用原始分数 z 的对数概率')
arrow(a,[(29,40),(29,24)])
arrow(a,[(66,16.5),(85,16.5),(85,31),(54.5,31),(54.5,71)],dash=True)
arrow(a,[(93.5,71),(93.5,61),(168,61),(168,16.5),(164,16.5)],dash=True)
label(a,85,3,'实线：前向与经验流　　虚线：更新信号；投影允许精确零比例')
save(f,'fig5_sac','执行路由表征下的 SAC 采样与更新')

# A compact booklet makes all figures reviewable without compiling LaTeX.
with PdfPages(ROOT/'figures_review.pdf') as pdf:
    for f,name,title in figures:
        pdf.savefig(f)

# Machine-check lettering sizes and bounds; visual checks remain necessary.
out_of_bounds=[]
for t in texts:
    b=t.get_window_extent(t.figure.canvas.get_renderer())
    fb=t.figure.bbox
    if b.x0<fb.x0-1 or b.x1>fb.x1+1 or b.y0<fb.y0-1 or b.y1>fb.y1+1:
        out_of_bounds.append(t.get_text())
report={'figures':len(figures),'min_font_pt':min(t.get_fontsize() for t in texts),
        'text_out_of_canvas':out_of_bounds,'width_mm':170,
        'style':'grayscale, Chinese labels, original vector artwork',
        'figure_files':[n for _,n,_ in figures]}
(ROOT/'figure_source'/'figure_check.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(report,ensure_ascii=False))
if out_of_bounds: raise RuntimeError('Some labels extend beyond the figure canvas')
