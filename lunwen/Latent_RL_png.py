#修改成了latex版本， https://www.overleaf.com/上直接画



# \documentclass[tikz,border=15pt]{standalone}
# \usetikzlibrary{shapes.geometric, arrows.meta, positioning, calc, shadows, backgrounds, fit}

# \begin{document}

# \begin{tikzpicture}[
#     font=\Large\sffamily, % 全局基础字体调大至 Large
#     >=Stealth,
#     % 基础样式定义
#     base/.style={draw, thick, align=center, minimum height=1.5cm, minimum width=3cm, rounded corners=4pt, fill=white, drop shadow={opacity=0.2}},
#     agent/.style={base, fill=blue!10},
#     env/.style={base, fill=gray!10, minimum width=4.5cm},
#     enc/.style={base, fill=green!10},
#     reward/.style={draw, thick, diamond, aspect=2.5, fill=red!10, align=center, drop shadow={opacity=0.2}, minimum width=4cm},
#     % 线条样式
#     mainarrow/.style={->, line width=1.5pt}, % 线条加粗
#     feedback/.style={->, line width=1.5pt, dashed, red},
#     loopline/.style={->, line width=1.5pt, blue!70!black},
#     % 连线文字样式
#     var/.style={font=\Large\bfseries\itshape} % 专门用于连线上变量的样式：大号、加粗、斜体
# ]

#     % --- 1. 左侧模块: TD3 Agent ---
#     \node[agent] (actor) {Actor\\$\pi(a|y)$};
#     \node[agent, below=2cm of actor] (critic) {Critic\\$Q(y,a)$};
    
#     % 背景大框 (TD3 Agent)
#     \begin{scope}[on background layer]
#         \node[draw, blue!50, line width=1pt, dashed, rounded corners=8pt, inner sep=20pt, 
#               fit=(actor) (critic), label={[blue, font=\huge\bfseries]above:TD3 Agent}] (agent_box) {};
#     \end{scope}

#     % --- 2. 右侧模块: Latent Dynamics ---
#     \node[base, fill=orange!10, right=5cm of actor] (encoder_g) {Action Encoder\\$G$};
#     \node[env, right=2cm of encoder_g] (koopman) {Koopman Environment\\$y_{t+1}^{raw} = A y_t + B u_t$};

#     % --- 3. 底部回路: Manifold Projection ---
#     \node[enc, below=4.5cm of koopman] (decoder_f) {State Decoder\\$F^{-1}$};
#     \node[reward, left=3.5cm of decoder_f] (reward_node) {Reward \& Physical\\Clipping};
#     \node[enc, left=3.5cm of reward_node] (encoder_f) {State Encoder\\$F$};

#     % --- 4. 连线逻辑 (横平竖直) ---

#     % 控制流 [左 -> 右]
#     \draw[mainarrow] (actor.east) -- (encoder_g.west) node[midway, above, var] {$a_t$};
#     \draw[mainarrow] (encoder_g.east) -- (koopman.west) node[midway, above, var] {$u_t$};
    
#     % 动作反馈给 Critic
#     \draw[->, line width=1pt, dashed, gray] ($(actor.east)!0.5!(encoder_g.west)$) |- (critic.east);

#     % 演化流 [右 -> 下]
#     \draw[mainarrow] (koopman.south) -- (decoder_f.north) node[midway, right, var] {$y_{t+1}^{raw}$};

#     % 反馈流 [下侧自右向左]
#     \draw[mainarrow] (decoder_f.west) -- (reward_node.east) node[midway, above, var] {$\hat{s}_{t+1}$};
#     \draw[mainarrow] (reward_node.west) -- (encoder_f.east) node[midway, above, var] {clipped $s$};

#     % rt 连线修正 (加粗变量)
#     \draw[feedback] (reward_node.north) -- (reward_node.north |- critic.east) -- (critic.east)
#         node[pos=0.1, right, var, text=red] {$r_t$};

#     % yt 连线修正 (加粗变量)
#     \draw[loopline] (encoder_f.west) -- ++(-2.5cm, 0) |- (agent_box.west)
#         node[pos=0.75, left, var, text=blue!70!black] {$y_{t+1}$};

# \end{tikzpicture}
# \end{document}




























# from graphviz import Digraph
# import os

# # 创建绘图对象
# dot = Digraph('Latent_RL_Rectangle_Loop')

# # ================= 1. 全局布局锁定 (关键设置) =================
# dot.attr(dpi='300')
# dot.attr(rankdir='LR')
# # ortho 强制所有线条为直角，类似 TikZ 的 |- 效果
# dot.attr(splines='ortho')
# # 调大间距，确保绕回线条有足够空间不与框重合
# dot.attr(nodesep='1.2', ranksep='1.5')

# # 全局节点与字体放大
# dot.attr('node', shape='box', style='filled,rounded', 
#          fontname='Times-Italic', fontsize='20', # 字体调大至 20pt
#          margin='0.3,0.2', height='1.2') # 框框加大

# dot.attr('edge', fontname='Times-Italic', fontsize='16', penwidth='2.0')

# # ================= 2. 辅助函数：彩色矩阵块 (18pt) =================
# def vector(label, color):
#     return f'''<
#     <TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0">
#         <TR><TD BGCOLOR="{color}" WIDTH="30" HEIGHT="15"></TD></TR>
#         <TR><TD BGCOLOR="{color}" WIDTH="30" HEIGHT="15"></TD></TR>
#         <TR><TD><FONT POINT-SIZE="18">{label}</FONT></TD></TR> 
#     </TABLE>
#     >'''

# # ================= 3. 节点定义 =================

# # --- 左侧：TD3 Agent ---
# with dot.subgraph(name='cluster_agent') as A:
#     A.attr(label='<<B>TD3 Agent Controller</B>>', fontsize='22', 
#            fontname='Times-Bold', style='dashed', color='#1f77b4')
#     # 使用 group 属性强制垂直对齐
#     A.node('actor', '<<I>Actor &pi;(a|y)</I>>', fillcolor='#e8f0fe', group='left')
#     A.node('critic', '<<I>Critic Q(y,a)</I>>', fillcolor='#e8f0fe', group='left')

# # --- 右侧：Latent Dynamics ---
# dot.node('G', '<Action Encoder<BR/><I>G</I>>', fillcolor='#fef7e0', group='right')
# dot.node('koop', '<<B>Koopman Environment</B><BR/><I>y<SUB>t+1</SUB><SUP>raw</SUP> = A y<SUB>t</SUB> + B u<SUB>t</SUB></I>>', 
#          fillcolor='#f1f3f4', width='3.5', group='right')

# # --- 底部：Manifold Projection (投影回路) ---
# # 定义底部节点的组，确保它们在同一水平线上
# dot.node('decoder', '<State Decoder<BR/><I>F<SUP>-1</SUP></I>>', fillcolor='#e6f4ea')
# dot.node('reward', '<Reward &amp; Physical<BR/>Clipping>', shape='diamond', fillcolor='#fce8e6')
# dot.node('encoder', '<State Encoder<BR/><I>F</I>>', fillcolor='#e6f4ea')

# # --- 信号与中间变量 ---
# dot.node('st_hat', vector('<I>s&#770;<SUB>t+1</SUB></I>', '#e6f4ea'), shape='none')
# dot.node('y_raw', vector('<I>y<SUB>t+1</SUB><SUP>raw</SUP></I>', '#f1f3f4'), shape='none')
# dot.node('y_next', vector('<I>y<SUB>t+1</SUB></I>', '#e8f0fe'), shape='none')

# # ================= 4. 连线逻辑 (逆时针强制布局) =================

# # [顶部：左 -> 右] 动作输出
# dot.edge('actor', 'G', label='<<I>a<SUB>t</SUB></I>>')
# dot.edge('G', 'koop', label='<<I>u<SUB>t</SUB></I>>')

# # [动作反馈给 Critic]
# dot.edge('G', 'critic', style='dashed', color='gray', constraint='false')

# # [右侧：下行] 动力学演化进入投影
# # 使用 group 保持 koop, y_raw, decoder 垂直对齐
# dot.edge('koop', 'y_raw')
# dot.edge('y_raw', 'decoder')

# # [底部：右 -> 左] 投影链
# dot.edge('decoder', 'st_hat')
# dot.edge('st_hat', 'reward')
# dot.edge('reward', 'encoder', label='clipped s')

# # [奖励反馈]
# dot.edge('reward', 'critic', style='dashed', color='red', label='<<FONT COLOR="red"><I>r<SUB>t</SUB></I></FONT>>')

# # [左侧绕回：下 -> 上] 闭环
# dot.edge('encoder', 'y_next')
# # 重点：constraint='false' 配合 rank 强制 y_next 绕回 actor
# dot.edge('y_next', 'actor', label='<<I>y<SUB>t+1</SUB></I>>', color='#1a73e8', penwidth='2.5')

# # ================= 5. 布局微调 (对齐约束) =================
# # 强制让左右两列对齐
# with dot.subgraph() as s:
#     s.attr(rank='same')
#     s.node('actor'); s.node('G')

# with dot.subgraph() as s:
#     s.attr(rank='same')
#     s.node('y_next'); s.node('encoder'); s.node('decoder'); s.node('y_raw')

# # 渲染
# output_dir = '/home/nng/koopman_project/lunwen'
# if not os.path.exists(output_dir): os.makedirs(output_dir)
# save_path = os.path.join(output_dir, 'Latent_RL_Rectangle_Final')

# dot.render(save_path, format='png', cleanup=True)
# dot.render(save_path, format='pdf', cleanup=True)

# print(f"✅ 布局已锁定！专业矩形闭环图已生成至: {save_path}.png")