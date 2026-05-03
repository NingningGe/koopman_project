from graphviz import Digraph

dot = Digraph('Koopman_Architecture')

# 极致紧凑的全局设置
# dpi='300' 保证矢量转图时的超高清晰度
dot.attr(rankdir='LR', splines='spline', nodesep='0.35', ranksep='0.4', dpi='300')

# =======================
# 辅助函数: 画向量矩阵块
# =======================
def vector(label, color):
    return f'''<
    <TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0">
        <TR><TD BGCOLOR="{color}" WIDTH="20" HEIGHT="10"></TD></TR>
        <TR><TD BGCOLOR="{color}" WIDTH="20" HEIGHT="10"></TD></TR>
        <TR><TD BGCOLOR="{color}" WIDTH="20" HEIGHT="10"></TD></TR>
        <TR><TD>{label}</TD></TR>
    </TABLE>
    >'''

# =======================
# Stage 1: 输入层
# =======================
dot.node('a_t', vector('<I>a<SUB>t</SUB></I>', 'paleturquoise'), shape='none')
dot.node('s_t', vector('<I>s<SUB>t</SUB></I>', 'lightblue'), shape='none')

# 强制将 a_t 和 s_t 上下对齐
with dot.subgraph() as s:
    s.attr(rank='same')
    s.node('a_t')
    s.node('s_t')

# =======================
# Stage 2: 编码器层
# =======================
dot.node('sum1', '<<I>⊕</I>>', shape='circle', width='0.3', fixedsize='true')
dot.node('G', 'Action Encoder\nG', shape='box', style='rounded')
dot.node('F', 'State Encoder\nF', shape='box', style='rounded')

dot.node('U_t', vector('<I>U<SUB>t</SUB></I>', 'pink'), shape='none')
dot.node('Y_t', vector('<I>Y<SUB>t</SUB></I>', 'lightgray'), shape='none')

# 连接
dot.edge('a_t', 'sum1')
dot.edge('s_t', 'sum1')
dot.edge('s_t', 'F')

dot.edge('sum1', 'G')
dot.edge('F', 'Y_t')
dot.edge('G', 'U_t')

# =======================
# Stage 3: Lifting (提升网络)
# =======================
dot.node('lift', '<Lifting Net<BR/><I>g_θ</I>>', shape='box', style='rounded')
dot.node('gY', vector('<I>g_θ(Y<SUB>t</SUB>)</I>', 'mediumpurple'), shape='none')

# Koopman 状态簇 (虚线框)
with dot.subgraph(name='cluster_Z') as c:
    c.attr(label='<<I>Koopman State Z<SUB>t</SUB> =<BR/>[Y<SUB>t</SUB>; g_θ(Y<SUB>t</SUB>)]</I>>', style='dashed', margin='10')
    c.node('Z_top', vector('<I>Y<SUB>t</SUB></I>', 'lightgray'), shape='none')
    c.node('Z_bottom', vector('<I>g_θ(Y<SUB>t</SUB>)</I>', 'mediumpurple'), shape='none')

dot.edge('Y_t', 'Z_top')
dot.edge('Y_t', 'lift')
dot.edge('lift', 'gY')
dot.edge('gY', 'Z_bottom')

# =======================
# Stage 4: Koopman 动力学
# =======================
dot.node('B', 'Koopman\nMatrix B', shape='box')
dot.node('A', 'Koopman\nMatrix A', shape='box')
dot.node('sum2', '<<I>⊕</I>>', shape='circle', width='0.3', fixedsize='true')

dot.node('Z_t1', vector('<I>Z<SUB>t+1</SUB></I>', 'lightblue'), shape='none')

dot.edge('U_t', 'B')
dot.edge('Z_top', 'A')
dot.edge('Z_bottom', 'A')

dot.edge('B', 'sum2')
dot.edge('A', 'sum2')
dot.edge('sum2', 'Z_t1', label='<<I>Z<SUB>t+1</SUB> = A Z<SUB>t</SUB><BR/>+ B U<SUB>t</SUB></I>>')

# 强制矩阵 A 和 B 上下对齐
with dot.subgraph() as s:
    s.attr(rank='same')
    s.node('B')
    s.node('A')

# =======================
# Stage 5: U型折叠与解码器 (核心优化区)
# =======================
dot.node('Z_tk', vector('<I>Z<SUB>t+K</SUB></I>', 'lightblue'), shape='none')
dot.node('extract', '<<I>Y = CZ</I>>', shape='box')
dot.node('Finv', '<Decoder<BR/><I>F<SUP>-1</SUP></I>>', shape='box', style='rounded')
dot.node('s_t1', vector('<I>s<SUB>t+1</SUB></I>', 'lightblue'), shape='none')

# 【分支1】继续向右演化
# 这里顺便修复了之前 K-step 换行失效的 Bug
dot.edge('Z_t1', 'Z_tk', style='dashed', label='<<I>K-step<BR/>evolution</I>>')

# 【分支2】垂直向下折叠提取 Y
with dot.subgraph() as s:
    s.attr(rank='same')
    s.node('Z_t1')
    s.node('extract')
dot.edge('Z_t1', 'extract')

# 【分支3】向左“倒车”完成解码 (利用 dir='back' 反转箭头视觉方向)
dot.edge('Finv', 'extract', dir='back')
dot.edge('s_t1', 'Finv', dir='back')

# =======================
# 渲染与输出
# =======================
output_path = '/home/nng/koopman_project/lunwen/koopman_architecture'

# 输出矢量图
dot.render(output_path, format='pdf', view=False)
dot.render(output_path, format='svg', view=False)

print(f"✅ 搞定！【U型折叠版】4:3架构图已生成！")