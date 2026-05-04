from graphviz import Digraph
import os

dot = Digraph('Koopman_Unrolled_Loss')

# 设置论文级 DPI 和布局属性
dot.attr(rankdir='LR', splines='spline', nodesep='0.6', ranksep='0.8', dpi='300')

# =======================
# 辅助函数: 绘制专业矩阵/向量块
# =======================
def vector(label, color):
    return f'''<
    <TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0">
        <TR><TD BGCOLOR="{color}" WIDTH="15" HEIGHT="8"></TD></TR>
        <TR><TD BGCOLOR="{color}" WIDTH="15" HEIGHT="8"></TD></TR>
        <TR><TD>{label}</TD></TR>
    </TABLE>
    >'''

# =======================
# Stage 1: Ground Truth (观测/真实值层)
# =======================
# 使用绿色系表示真实观测状态 x
dot.node('x_t', vector('<I>x<SUB>t</SUB></I>', '#e5f5e0'), shape='none')
dot.node('x_t1', vector('<I>x<SUB>t+1</SUB></I>', '#e5f5e0'), shape='none')
dot.node('x_t2', vector('<I>x<SUB>t+2</SUB></I>', '#e5f5e0'), shape='none')

# 使用橙色系表示外部控制输入 u
dot.node('u_t', vector('<I>u<SUB>t</SUB></I>', '#fee0d2'), shape='none')
dot.node('u_t1', vector('<I>u<SUB>t+1</SUB></I>', '#fee0d2'), shape='none')

# 确保观测状态水平对齐
with dot.subgraph() as s:
    s.attr(rank='same')
    s.node('x_t'); s.node('x_t1'); s.node('x_t2')

# =======================
# Stage 2: Koopman Latent Space (潜在演化层)
# =======================
# 编码器 F (蓝色系)
dot.node('F', '<Encoder<BR/><I>F</I>>', shape='box', style='filled,rounded', fillcolor='#bdd7e7')

# 潜在状态 y (浅蓝色)
dot.node('y_t', vector('<I>y<SUB>t</SUB></I>', '#deebf7'), shape='none')
dot.node('y_t1_hat', vector('<I>y&#770;<SUB>t+1</SUB></I>', '#deebf7'), shape='none')
dot.node('y_t2_hat', vector('<I>y&#770;<SUB>t+2</SUB></I>', '#deebf7'), shape='none')

# 线性转移算子 (A, B)
dot.node('K1', '<<I>A, B</I>>', shape='circle', width='0.5', style='filled', fillcolor='#9ecae1')
dot.node('K2', '<<I>A, B</I>>', shape='circle', width='0.5', style='filled', fillcolor='#9ecae1')

# 时序演化连接
dot.edge('x_t', 'F')
dot.edge('F', 'y_t')
dot.edge('y_t', 'K1')
dot.edge('u_t', 'K1')
dot.edge('K1', 'y_t1_hat')
dot.edge('y_t1_hat', 'K2')
dot.edge('u_t1', 'K2')
dot.edge('K2', 'y_t2_hat')

# =======================
# Stage 3: Decoding & Reconstruction (解码与重构层)
# =======================
# 解码器 F^-1
dot.node('Finv', '<Decoder<BR/><I>F<SUP>-1</SUP></I>>', shape='box', style='filled,rounded', fillcolor='#c7e9c0')

# 预测出的观测状态 x_hat
dot.node('x_t_hat', vector('<I>x&#770;<SUB>t</SUB></I>', '#edf8e9'), shape='none')
dot.node('x_t1_hat', vector('<I>x&#770;<SUB>t+1</SUB></I>', '#edf8e9'), shape='none')
dot.node('x_t2_hat', vector('<I>x&#770;<SUB>t+2</SUB></I>', '#edf8e9'), shape='none')

# 解码路径
dot.edge('y_t', 'Finv')
dot.edge('y_t1_hat', 'Finv')
dot.edge('y_t2_hat', 'Finv')
dot.edge('Finv', 'x_t_hat')
dot.edge('Finv', 'x_t1_hat')
dot.edge('Finv', 'x_t2_hat')

# =======================
# Stage 4: Loss Functions (三类关键损失标注)
# =======================
# 1. 重构损失 L_rec (红色虚线)
dot.edge('x_t_hat', 'x_t', dir='both', style='dashed', color='red', 
         label='<<FONT COLOR=\"red\"><I>L<SUB>rec</SUB></I></FONT>>')

# 2. 预测损失 L_pred (红色虚线)
dot.edge('x_t1_hat', 'x_t1', dir='both', style='dashed', color='red', 
         label='<<FONT COLOR=\"red\"><I>L<SUB>pred</SUB></I></FONT>>')
dot.edge('x_t2_hat', 'x_t2', dir='both', style='dashed', color='red')

# 3. 线性演化损失 L_lin (潜在空间蓝色虚线)
dot.node('y_t1_true', vector('<I>F(x<SUB>t+1</SUB>)</I>', '#f0f0f0'), shape='none')
dot.edge('x_t1', 'y_t1_true', style='dotted')
dot.edge('y_t1_hat', 'y_t1_true', dir='both', style='dashed', color='blue', 
         label='<<FONT COLOR=\"blue\"><I>L<SUB>lin</SUB></I></FONT>>')

# =======================
# 渲染与输出设置
# =======================
output_dir = '/home/nng/koopman_project/lunwen'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
output_path = os.path.join(output_dir, 'koopman_time_unrolled_professional')

dot.render(output_path, format='png', cleanup=True)
dot.render(output_path, format='pdf', cleanup=True)

print(f"✅ 专业版【时序展开损失图】已生成至: {output_path}.png")