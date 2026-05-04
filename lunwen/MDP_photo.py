from graphviz import Digraph
import os

# 创建绘图对象
dot = Digraph(format='png', comment='Heterogeneous MDP Alignment')
dot.attr(dpi='300') 
dot.attr(rankdir='LR', splines='spline', nodesep='1.0', ranksep='1.2')

# ================= Global settings =================
# 注意：使用 HTML 标签时，fontname 通常设为 Times-Italic 以符合学术规范
dot.attr('node', shape='box', style='filled, rounded', fontname='Times-Italic', fontsize='14')
dot.attr('edge', fontname='Times-Italic', fontsize='12', color='#333333')

# ================= Domain A (Franka) =================
with dot.subgraph(name='cluster_A') as A:
    # 标题使用 HTML 模式，支持下标 M_A
    A.attr(label='<<B>Domain A: Franka MDP </B><I>M<SUB>A</SUB></I>>', 
           style='dashed', color='#1f77b4', fontname='Times-Bold', fontsize='16')
    
    # 状态 s_A 和 动作 a_A
    A.node('sA', '<<I>s<SUB>A</SUB></I> &isin; R<SUP>14</SUP>>', fillcolor='#c6dbef', height='1.0')
    A.node('aA', '<<I>a<SUB>A</SUB></I> &isin; R<SUP>7</SUP>>', fillcolor='#9ecae1', height='1.0')
    
    # 编码器 F_A 和 G_A
    A.node('FA', '<<I>Encoder F<SUB>A</SUB></I>>', fillcolor='#6baed6')
    A.node('GA', '<<I>Encoder G<SUB>A</SUB></I>>', fillcolor='#4292c6')
    
    A.edge('sA', 'FA')
    A.edge('sA', 'GA')
    A.edge('aA', 'GA')

# ================= Domain B (UR) =================
with dot.subgraph(name='cluster_B') as B:
    B.attr(label='<<B>Domain B: UR MDP </B><I>M<SUB>B</SUB></I>>', 
           style='dashed', color='#d62728', fontname='Times-Bold', fontsize='16')
    
    B.node('sB', '<<I>s<SUB>B</SUB></I> &isin; R<SUP>12</SUP>>', fillcolor='#fcbba1', height='1.0')
    B.node('aB', '<<I>a<SUB>B</SUB></I> &isin; R<SUP>6</SUP>>', fillcolor='#fc9272', height='1.0')
    
    B.node('FB', '<<I>Encoder F<SUB>B</SUB></I>>', fillcolor='#fb6a4a')
    B.node('GB', '<<I>Encoder G<SUB>B</SUB></I>>', fillcolor='#ef3b2c')
    
    B.edge('sB', 'FB')
    B.edge('sB', 'GB')
    B.edge('aB', 'GB')

# ================= Latent Space (Shared) =================
with dot.subgraph(name='cluster_latent') as L:
    L.attr(label='<<B>Shared Latent Manifold</B>>', 
           style='dashed', color='#8c564b', fontname='Times-Bold', fontsize='16')
    
    L.node('y', '<<I>Latent State y</I> &isin; R<SUP>d</SUP>>', fillcolor='#dadaeb', height='1.2')
    L.node('u', '<<I>Latent Action u</I> &isin; R<SUP>m</SUP>>', fillcolor='#bcbddc', height='1.2')

# ================= Alignment Edges =================
dot.edge('FA', 'y', color='black', penwidth='2.0')
dot.edge('FB', 'y', color='black', penwidth='2.0')
dot.edge('GA', 'u', color='black', penwidth='2.0')
dot.edge('GB', 'u', color='black', penwidth='2.0')

# 保存路径
save_dir = '/home/nng/koopman_project/lunwen'
save_file = os.path.join(save_dir, 'MDP_alignment_professional')

# 渲染输出
dot.render(save_file, format='png', cleanup=True)
dot.render(save_file, format='pdf', cleanup=True)

print(f"✅ 完成！专业下标版 MDP 对齐图已保存至: {save_file}.png")