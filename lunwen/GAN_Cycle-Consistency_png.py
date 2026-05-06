
#修改成了latex版本， https://www.overleaf.com/上直接画


\documentclass[tikz,border=15pt]{standalone}

% 引入排版专家常用的 TikZ 库
\usetikzlibrary{shapes.geometric, arrows.meta, positioning, calc, shadows, backgrounds, fit}

\begin{document}

% =========================
% 定义学术论文标准颜色
% =========================
\definecolor{domainA}{RGB}{232, 240, 254} % 浅蓝色 (Source Domain A)
\definecolor{domainB}{RGB}{254, 247, 224} % 浅橙色 (Target Domain B)
\definecolor{frozen}{RGB}{240, 240, 240}  % 浅灰色 (Frozen Networks)
\definecolor{lossred}{RGB}{220, 50, 47}   % 损失函数红

\begin{tikzpicture}[
    font=\Large\sffamily, % 全局字体放大
    >=Stealth,
    % -------------------------
    % 定义向量块样式 (3个小方格垂直堆叠)
    % -------------------------
    vec/.style={
        rectangle, draw, thick, rounded corners=1.5pt,
        minimum width=1.4cm, minimum height=2.1cm,
        align=center, fill=#1,
        drop shadow={opacity=0.1, shadow xshift=2pt, shadow yshift=-2pt},
        path picture={
            \draw[thick] ($(path picture bounding box.north west)!0.333!(path picture bounding box.south west)$) -- 
                         ($(path picture bounding box.north east)!0.333!(path picture bounding box.south east)$);
            \draw[thick] ($(path picture bounding box.north west)!0.667!(path picture bounding box.south west)$) -- 
                         ($(path picture bounding box.north east)!0.667!(path picture bounding box.south east)$);
        }
    },
    % -------------------------
    % 定义网络节点样式
    % -------------------------
    netF/.style={draw, thick, rounded corners=6pt, fill=frozen, minimum width=2.8cm, minimum height=1.4cm, align=center},
    netT/.style={draw, thick, dashed, rounded corners=6pt, fill=domainB!60, minimum width=2.8cm, minimum height=1.4cm, align=center},
    % -------------------------
    % 判别器样式
    % -------------------------
    disc/.style={draw=lossred, thick, diamond, aspect=2.5, fill=lossred!10, minimum width=3.8cm, align=center},
    % -------------------------
    % 连线样式
    % -------------------------
    mainline/.style={->, thick, line width=1.5pt},
    lossline/.style={->, dashed, thick, color=lossred, line width=1.2pt}
]

    % ==========================================
    % 第一阶段 (Top Row) [y = 0]
    % ==========================================
    
    % 左侧 (Domain B)
    \node[vec=domainB] (sB) at (0, 0) {$s_B$};
    \node[netT] (FB) at (3.0, 0) {Encoder\\$F_B$};
    \node[vec=domainB] (yB) at (6.0, 0) {$y_B$};
    
    % 右侧 (Domain A)
    \node[vec=domainA] (yA) at (12.0, 0) {$y_A$};
    \node[netF] (FA) at (15.0, 0) {Encoder\\$F_A$\\ \scriptsize(Frozen)};
    \node[vec=domainA] (sA) at (18.0, 0) {$s_A$};
    
    % 中心 (Discriminator) [y = -3.5]
    \node[disc] (D) at (9.0, -3.5) {Discriminator $D$};
    
    % --- 第一阶段连线 ---
    \draw[mainline] (sB) -- (FB);
    \draw[mainline] (FB) -- (yB);
    \draw[mainline] (sA) -- (FA);
    \draw[mainline] (FA) -- (yA);
    
    % 指向 D 的线条 (完美避开下方干涉)
    \draw[mainline] (yB.south) |- (D.west) node[pos=0.75, above] {Fake};
    \draw[mainline] (yA.west) |- (D.east) node[pos=0.75, above] {Real};
    
    % 对抗损失 L_adv 走线：从 D 底部出发，在 D 和 Projection 层之间穿过
    \draw[lossline] (D.south) -- ++(0, -1.5) -| (FB.south) 
        node[pos=0.25, below, text=lossred] {$L_{adv}$};

    % ==========================================
    % 第二阶段 (Middle Row): 跨域投影 [y = -7.5]
    % ==========================================
    
    % 严格在右侧下方布置，与顶层变量精确对齐
    \node[vec=domainB] (yBp) at (12.0, -7.5) {$y'_B$};
    \node[netT] (FBinv) at (6.0, -7.5) {Decoder\\$F_B^{-1}$};
    \node[vec=domainB] (sBp) at (3.0, -7.5) {$s'_B$};
    
    % --- 第二阶段连线 ---
    \draw[mainline] (yA.south) -- (yBp.north) node[midway, right] {Projection};
    \draw[mainline] (yBp) -- (FBinv);
    \draw[mainline] (FBinv) -- (sBp);

    % ==========================================
    % 第三阶段 (Bottom Row): 循环闭环 [y = -12.0]
    % ==========================================
    
    \node[netT] (FB2) at (6.0, -12.0) {Encoder\\$F_B$};
    \node[vec=domainA] (yAp) at (10.0, -12.0) {$y'_A$};
    \node[netF] (FAinv) at (14.0, -12.0) {Decoder\\$F_A^{-1}$\\ \scriptsize(Frozen)};
    \node[vec=domainA] (sAp) at (18.0, -12.0) {$s'_A$};
    
    % --- 第三阶段连线 ---
    % 从 s'_B 下降并向右进入 Encoder (拐弯线)
    \draw[mainline] (sBp.south) |- (FB2.west);
    \draw[mainline] (FB2) -- (yAp);
    \draw[mainline] (yAp) -- (FAinv);
    \draw[mainline] (FAinv) -- (sAp);

    % ==========================================
    % 循环损失 (Cycle Loss): 右侧大包抄 
    % ==========================================
    \draw[lossline] (sA.east) -- ++(1.5, 0) |- (sAp.east) 
        node[pos=0.25, right, font=\huge\bfseries, text=lossred] {$L_{cyc}$};

\end{tikzpicture}

\end{document}











# from graphviz import Digraph
# import os

# dot = Digraph('GAN_Cycle_Alignment_V2')

# # 全局紧凑设置
# dot.attr(rankdir='LR', splines='spline', nodesep='0.6', ranksep='0.5', dpi='300')

# # =======================
# # 辅助函数: 画向量矩阵块
# # =======================
# def vector(label, color):
#     return f'''<
#     <TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0">
#         <TR><TD BGCOLOR="{color}" WIDTH="15" HEIGHT="8"></TD></TR>
#         <TR><TD BGCOLOR="{color}" WIDTH="15" HEIGHT="8"></TD></TR>
#         <TR><TD>{label}</TD></TR>
#     </TABLE>
#     >'''

# # =======================
# # Stage 1: Cycle Consistency Path (A -> yA -> sB' -> yA' -> sA')
# # =======================
# with dot.subgraph(name='cluster_top') as T:
#     T.attr(label='<<B>Cycle Consistency Flow (Source Domain A)</B>>', 
#            fontsize='14', style='dashed', color='#756bb1')
    
#     # 1. 初始状态 sA
#     T.node('sA', vector('<I>s<SUB>A</SUB></I>', '#c6dbef'), shape='none')
    
#     # 2. 冻结的源域编码器 FA
#     T.node('FA', '<Encoder<BR/><I>F<SUB>A</SUB></I><BR/><FONT POINT-SIZE="10">(Frozen)</FONT>>', 
#            shape='box', style='filled,rounded', fillcolor='#bdbdbd')
    
#     # 3. 源域潜在表示 yA
#     T.node('yA', vector('<I>y<SUB>A</SUB></I>', '#e0e0e0'), shape='none')
    
#     # 4. 可训练的目标域逆映射 FB^-1 (Decoder B)
#     T.node('FB_inv', '<Decoder<BR/><I>F<SUB>B</SUB><SUP>-1</SUP></I><BR/><FONT POINT-SIZE="10">(Trainable)</FONT>>', 
#            shape='box', style='filled,dashed,rounded', fillcolor='#fdd0a2')
    
#     # 5. 生成的目标域状态 sB'
#     T.node('sB_p', vector('<I>s\'<SUB>B</SUB></I>', '#fee6ce'), shape='none')
    
#     # 6. 可训练的目标域编码器 FB
#     T.node('FB_cycle', '<Encoder<BR/><I>F<SUB>B</SUB></I>>', 
#            shape='box', style='filled,dashed,rounded', fillcolor='#fdae6b')

#     # 7. 新增：循环后的潜在表示 yA'
#     T.node('yA_p', vector('<I>y\'<SUB>A</SUB></I>', '#e0e0e0'), shape='none')

#     # 8. 新增：冻结的源域解码器 FA^-1
#     T.node('FA_inv', '<Decoder<BR/><I>F<SUB>A</SUB><SUP>-1</SUP></I><BR/><FONT POINT-SIZE="10">(Frozen)</FONT>>', 
#            shape='box', style='filled,rounded', fillcolor='#bdbdbd')
    
#     # 9. 最终重构状态 sA'
#     T.node('sA_hat', vector('<I>s\'<SUB>A</SUB></I>', '#deebf7'), shape='none')
    
#     # 顺序连接
#     T.edge('sA', 'FA')
#     T.edge('FA', 'yA')
#     T.edge('yA', 'FB_inv')
#     T.edge('FB_inv', 'sB_p')
#     T.edge('sB_p', 'FB_cycle')
#     T.edge('FB_cycle', 'yA_p')
#     T.edge('yA_p', 'FA_inv')
#     T.edge('FA_inv', 'sA_hat')

# # =======================
# # Stage 2: Adversarial Alignment Path (Target Domain B)
# # =======================
# with dot.subgraph(name='cluster_bottom') as B:
#     B.attr(label='<<B>Target Projection (Domain B)</B>>', 
#            fontsize='14', style='dashed', color='#e6550d')
    
#     B.node('sB', vector('<I>s<SUB>B</SUB></I>', '#fdd0a2'), shape='none')
#     B.node('FB_main', '<Encoder<BR/><I>F<SUB>B</SUB></I>>', 
#            shape='box', style='filled,dashed,rounded', fillcolor='#fdae6b')
#     B.node('yB', vector('<I>y<SUB>B</SUB></I>', '#eeeeee'), shape='none')
    
#     B.edge('sB', 'FB_main')
#     B.edge('FB_main', 'yB')

# # =======================
# # Stage 3: Discriminator & Loss
# # =======================
# dot.node('Disc', '<<I>Discriminator D</I>>', shape='diamond', style='filled', fillcolor='#fb9a99')

# # 判别器输入：真实 yA vs 伪造 yB
# dot.edge('yA', 'Disc', label='Real')
# dot.edge('yB', 'Disc', label='Fake')

# # 循环一致性损失 (连接初始 sA 和最终重构 sA')
# dot.edge('sA', 'sA_hat', dir='both', style='dashed', color='red', constraint='false',
#          label='<<FONT COLOR="red"><I>L<SUB>cyc</SUB></I></FONT>>')

# # 对抗损失反馈
# dot.edge('Disc', 'FB_main', style='dashed', color='red', constraint='false',
#          label='<<FONT COLOR="red"><I>L<SUB>adv</SUB></I></FONT>>')

# # 布局对齐
# dot.edge('sA', 'sB', style='invis')

# # 保存
# output_dir = '/home/nng/koopman_project/lunwen'
# if not os.path.exists(output_dir): os.makedirs(output_dir)
# save_file = os.path.join(output_dir, 'GAN_Cycle_Alignment_V2')

# dot.render(save_file, format='png', cleanup=True)
# print(f"✅ 更新完成！文件已保存至: {save_file}.png")