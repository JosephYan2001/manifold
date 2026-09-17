"""Suite-level panels only; importing this module does not import pyplot."""
from pathlib import Path
import numpy as np
from .reporting import read_csv, bootstrap

LABELS = {'ours':'完整方法','sampled':'采样平方','mappo':'MAPPO','ippo':'IPPO',
          'no_direction_check':'去方向检查','no_return_check':'去回报检查','fit_quarter':'1/4 拟合'}


def setup():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    available = {f.name for f in font_manager.fontManager.ttflist}
    fonts = [name for name in ('Microsoft YaHei','SimHei','Noto Sans CJK SC','DejaVu Sans') if name in available]
    plt.rcParams['font.sans-serif'] = fonts
    plt.rcParams['axes.unicode_minus'] = False
    return plt


def plot_suite(directory):
    directory = Path(directory)
    rows = [r for r in read_csv(directory/'training_results.csv') if r['status']=='complete']
    if not rows:
        return
    plt = setup()
    fig, axes = plt.subplots(3,3,figsize=(18,13),layout='constrained')
    axes = axes.ravel()
    curves = read_csv(directory/'learning_curves.csv')
    conditions = list(dict.fromkeys(r['condition'] for r in rows))
    for ax,metric,title in zip(axes[:3],('J','coverage','collision_pairs'),('独立源评价：折扣回报','终点覆盖率','每步唯一碰撞对数')):
        for c in conditions:
            group = [r for r in curves if r['condition']==c]
            xs = sorted({int(r['budget_checkpoint']) for r in group})
            stats = [bootstrap([r[metric] for r in group if int(r['budget_checkpoint'])==x]) for x in xs]
            line, = ax.step(xs,[s['mean'] for s in stats],where='post',label=LABELS[c])
            if stats and stats[0]['count']>1:
                ax.fill_between(xs,[s['ci_low'] for s in stats],[s['ci_high'] for s in stats],step='post',alpha=.1,color=line.get_color())
        ax.set(title=title,xlabel='累计源团队步')
    def bars(ax,metric,title):
        for i,c in enumerate(conditions):
            values = [float(r[metric]) for r in rows if r['condition']==c and r.get(metric,'')!='']
            if values:
                s = bootstrap(values)
                ax.bar(i,s['mean'],alpha=.65)
                ax.scatter(np.full(len(values),i)+np.linspace(-.08,.08,len(values)),values,color='black',s=12)
                if len(values)>1:
                    ax.errorbar(i,s['mean'],yerr=[[s['mean']-s['ci_low']],[s['ci_high']-s['mean']]],color='black',capsize=3)
        ax.set(xticks=range(len(conditions)),xticklabels=[LABELS[c] for c in conditions],title=title)
        ax.tick_params(axis='x',rotation=30,labelsize=8)
    bars(axes[3],'AUC','源回报 AUC/B（点为训练种子）')
    bars(axes[4],'J','final 回报及种子区间')
    bottom = np.zeros(len(conditions))
    for key,label in [('train','采集训练'),('direction_check','方向检查'),('return_old','旧策略回报检查'),('return_candidate','候选回报检查'),('interrupted_uncommitted','中断未提交')]:
        heights = [np.mean([float(r.get('cost_'+key) or 0) for r in rows if r['condition']==c]) for c in conditions]
        if any(heights):
            axes[5].bar(range(len(conditions)),heights,bottom=bottom,label=label)
            bottom += heights
    axes[5].set(title='分项源交互成本',xticks=range(len(conditions)),xticklabels=[LABELS[c] for c in conditions])
    axes[5].tick_params(axis='x',rotation=30,labelsize=8)
    bars(axes[6],'train_seconds','训练墙钟（秒，排除独立评价）')
    bars(axes[7],'optimizer_steps','优化器更新次数')
    bars(axes[8],'fit_kl_after','每轮最后候选拟合 KL 均值')
    for ax in axes:
        ax.grid(alpha=.15)
    axes[0].legend(fontsize=8,ncol=2)
    axes[5].legend(fontsize=8)
    fig.suptitle('合作导航：源训练总览；区间为训练种子 bootstrap，单种子无区间')
    fig.savefig(directory/'overview.png',dpi=160)
    plt.close(fig)


def plot_transfer(directory):
    directory = Path(directory)
    rows = [r for r in read_csv(directory/'transfer_results.csv') if r['status']=='complete']
    if not rows:
        return
    plt = setup()
    fig,axes = plt.subplots(2,3,figsize=(16,9),layout='constrained')
    for ax,metric,title in zip(axes.ravel(),('J','distance','coverage','all_covered','collision_pairs','collisions_per_agent'),
                               ('折扣团队回报（同规模比较）','平均终点距离','终点覆盖率','全覆盖率','每步唯一碰撞对数','每机器人碰撞数')):
        for c in dict.fromkeys(r['condition'] for r in rows):
            group = [r for r in rows if r['condition']==c]
            ns = sorted({int(r['n_agents']) for r in group})
            stats = [bootstrap([r[metric] for r in group if int(r['n_agents'])==n]) for n in ns]
            line, = ax.plot(ns,[s['mean'] for s in stats],marker='o',label=LABELS[c])
            if stats[0]['count']>1:
                ax.fill_between(ns,[s['ci_low'] for s in stats],[s['ci_high'] for s in stats],alpha=.1,color=line.get_color())
        ax.set(title=title,xlabel='机器人/地标数',xticks=[3,4,6,8])
        ax.grid(alpha=.15)
    axes[0,0].legend()
    fig.suptitle('final 策略冻结迁移；N=4 复用源 final 评价')
    fig.savefig(directory/'transfer_overview.png',dpi=160)
    plt.close(fig)
