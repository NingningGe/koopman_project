from graphviz import Digraph
import os

dot = Digraph('GAN_Cycle_Alignment_V2')

# 全局紧凑设置
dot.attr(rankdir='LR', splines='spline', nodesep='0.6', ranksep='0.5', dpi='300')

# =======================
# 辅助函数: 画向量矩阵块
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
# Stage 1: Cycle Consistency Path (A -> yA -> sB' -> yA' -> sA')
# =======================
with dot.subgraph(name='cluster_top') as T:
    T.attr(label='<<B>Cycle Consistency Flow (Source Domain A)</B>>', 
           fontsize='14', style='dashed', color='#756bb1')
    
    # 1. 初始状态 sA
    T.node('sA', vector('<I>s<SUB>A</SUB></I>', '#c6dbef'), shape='none')
    
    # 2. 冻结的源域编码器 FA
    T.node('FA', '<Encoder<BR/><I>F<SUB>A</SUB></I><BR/><FONT POINT-SIZE="10">(Frozen)</FONT>>', 
           shape='box', style='filled,rounded', fillcolor='#bdbdbd')
    
    # 3. 源域潜在表示 yA
    T.node('yA', vector('<I>y<SUB>A</SUB></I>', '#e0e0e0'), shape='none')
    
    # 4. 可训练的目标域逆映射 FB^-1 (Decoder B)
    T.node('FB_inv', '<Decoder<BR/><I>F<SUB>B</SUB><SUP>-1</SUP></I><BR/><FONT POINT-SIZE="10">(Trainable)</FONT>>', 
           shape='box', style='filled,dashed,rounded', fillcolor='#fdd0a2')
    
    # 5. 生成的目标域状态 sB'
    T.node('sB_p', vector('<I>s\'<SUB>B</SUB></I>', '#fee6ce'), shape='none')
    
    # 6. 可训练的目标域编码器 FB
    T.node('FB_cycle', '<Encoder<BR/><I>F<SUB>B</SUB></I>>', 
           shape='box', style='filled,dashed,rounded', fillcolor='#fdae6b')

    # 7. 新增：循环后的潜在表示 yA'
    T.node('yA_p', vector('<I>y\'<SUB>A</SUB></I>', '#e0e0e0'), shape='none')

    # 8. 新增：冻结的源域解码器 FA^-1
    T.node('FA_inv', '<Decoder<BR/><I>F<SUB>A</SUB><SUP>-1</SUP></I><BR/><FONT POINT-SIZE="10">(Frozen)</FONT>>', 
           shape='box', style='filled,rounded', fillcolor='#bdbdbd')
    
    # 9. 最终重构状态 sA'
    T.node('sA_hat', vector('<I>s\'<SUB>A</SUB></I>', '#deebf7'), shape='none')
    
    # 顺序连接
    T.edge('sA', 'FA')
    T.edge('FA', 'yA')
    T.edge('yA', 'FB_inv')
    T.edge('FB_inv', 'sB_p')
    T.edge('sB_p', 'FB_cycle')
    T.edge('FB_cycle', 'yA_p')
    T.edge('yA_p', 'FA_inv')
    T.edge('FA_inv', 'sA_hat')

# =======================
# Stage 2: Adversarial Alignment Path (Target Domain B)
# =======================
with dot.subgraph(name='cluster_bottom') as B:
    B.attr(label='<<B>Target Projection (Domain B)</B>>', 
           fontsize='14', style='dashed', color='#e6550d')
    
    B.node('sB', vector('<I>s<SUB>B</SUB></I>', '#fdd0a2'), shape='none')
    B.node('FB_main', '<Encoder<BR/><I>F<SUB>B</SUB></I>>', 
           shape='box', style='filled,dashed,rounded', fillcolor='#fdae6b')
    B.node('yB', vector('<I>y<SUB>B</SUB></I>', '#eeeeee'), shape='none')
    
    B.edge('sB', 'FB_main')
    B.edge('FB_main', 'yB')

# =======================
# Stage 3: Discriminator & Loss
# =======================
dot.node('Disc', '<<I>Discriminator D</I>>', shape='diamond', style='filled', fillcolor='#fb9a99')

# 判别器输入：真实 yA vs 伪造 yB
dot.edge('yA', 'Disc', label='Real')
dot.edge('yB', 'Disc', label='Fake')

# 循环一致性损失 (连接初始 sA 和最终重构 sA')
dot.edge('sA', 'sA_hat', dir='both', style='dashed', color='red', constraint='false',
         label='<<FONT COLOR="red"><I>L<SUB>cyc</SUB></I></FONT>>')

# 对抗损失反馈
dot.edge('Disc', 'FB_main', style='dashed', color='red', constraint='false',
         label='<<FONT COLOR="red"><I>L<SUB>adv</SUB></I></FONT>>')

# 布局对齐
dot.edge('sA', 'sB', style='invis')

# 保存
output_dir = '/home/nng/koopman_project/lunwen'
if not os.path.exists(output_dir): os.makedirs(output_dir)
save_file = os.path.join(output_dir, 'GAN_Cycle_Alignment_V2')

dot.render(save_file, format='png', cleanup=True)
print(f"✅ 更新完成！文件已保存至: {save_file}.png")