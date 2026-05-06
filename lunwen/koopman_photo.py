# #修改成了latex版本， https://www.overleaf.com/上直接画


\documentclass[tikz,border=15pt]{standalone}

% 引入必要的宏包
\usetikzlibrary{shapes.geometric, arrows.meta, positioning, calc, shadows, backgrounds, fit}

\begin{document}

% =========================
% 定义全局颜色方案
% =========================
\definecolor{sblue}{RGB}{210, 235, 255}   % 浅蓝色 (s_t, Z_t+1, Z_t+K, s_t+1)
\definecolor{agreen}{RGB}{210, 245, 230}  % 浅青绿 (a_t)
\definecolor{ygray}{RGB}{230, 230, 230}   % 浅灰色 (Y_t)
\definecolor{gpurple}{RGB}{230, 210, 245} % 紫色 (g_theta)
\definecolor{upink}{RGB}{255, 215, 220}   % 浅粉色 (U_t)
\definecolor{netgray}{RGB}{250, 250, 250} % 网络的浅灰底色

\begin{tikzpicture}[
    >=Stealth,
    font=\Large, % 全局字体适度放大
    % -------------------------
    % 定义向量块样式 (3个小方格垂直堆叠)
    % -------------------------
    vec/.style={
        rectangle, draw, thick, rounded corners=1.5pt,
        minimum width=1.4cm, minimum height=2.1cm,
        align=center, fill=#1,
        drop shadow={opacity=0.15, shadow xshift=2pt, shadow yshift=-2pt},
        path picture={
            % 利用 path picture 自动绘制内部的两条横线将块三等分
            \draw[thick] ($(path picture bounding box.north west)!0.333!(path picture bounding box.south west)$) -- 
                         ($(path picture bounding box.north east)!0.333!(path picture bounding box.south east)$);
            \draw[thick] ($(path picture bounding box.north west)!0.667!(path picture bounding box.south west)$) -- 
                         ($(path picture bounding box.north east)!0.667!(path picture bounding box.south east)$);
        }
    },
    % 无阴影版本的向量块（专用于放在虚线框内部防止阴影重叠）
    vec_noshadow/.style={
        rectangle, draw, thick, rounded corners=1.5pt,
        minimum width=1.4cm, minimum height=2.1cm,
        align=center, fill=#1,
        path picture={
            \draw[thick] ($(path picture bounding box.north west)!0.333!(path picture bounding box.south west)$) -- 
                         ($(path picture bounding box.north east)!0.333!(path picture bounding box.south east)$);
            \draw[thick] ($(path picture bounding box.north west)!0.667!(path picture bounding box.south west)$) -- 
                         ($(path picture bounding box.north east)!0.667!(path picture bounding box.south east)$);
        }
    },
    % -------------------------
    % 定义网络与矩阵块样式
    % -------------------------
    net/.style={
        rectangle, draw, thick, rounded corners=4pt,
        minimum width=2.4cm, minimum height=1.3cm,
        align=center, fill=netgray,
        drop shadow={opacity=0.15, shadow xshift=2pt, shadow yshift=-2pt}
    },
    % -------------------------
    % 定义相加节点样式
    % -------------------------
    plus/.style={
        circle, draw, thick, fill=white,
        minimum size=0.7cm, inner sep=0pt, font=\Large\bfseries,
        drop shadow={opacity=0.15, shadow xshift=1.5pt, shadow yshift=-1.5pt}
    },
    % -------------------------
    % 定义连线样式
    % -------------------------
    myarrow/.style={->, thick, line width=1.2pt},
    mydash/.style={->, dashed, thick, line width=1.2pt}
]

    % ==========================================
    % 第一层 (Top Row): 编码与提升 (自左向右)
    % ==========================================
    
    % --- 输入与网络节点定位 ---
    \node[vec=sblue]  (st)    at (0, 0)     {$s_t$};
    \node[vec=agreen] (at)    at (0, -3.5)  {$a_t$};
    \node[plus]       (plus1) at (2, -3.5)  {$\mathbf{+}$};
    
    \node[net]        (encF)  at (3.8, 0)   {State Encoder\\$F$};
    \node[net]        (encG)  at (5.5, -3.5){Action Encoder\\$G$};
    
    \node[vec=ygray]  (Yt)    at (6.8, 0)   {$Y_t$};
    \node[vec=upink]  (ut)    at (9.5, -3.5){$U_t$};
    
    \node[net]        (lift)  at (10.5, 0)  {Lifting Net\\$g_\theta$};
    \node[vec=gpurple](gy)    at (13.0, 0)  {$g_\theta(Y_t)$};
    
    % --- Koopman 状态 (大虚线框) ---
    % 上下紧密贴合拼接 Y_t 和 g_theta(Y_t)
    \node[vec_noshadow=ygray]   (Z_yt) at (16.5,  1.05) {$Y_t$};
    \node[vec_noshadow=gpurple] (Z_gy) at (16.5, -1.05) {$g_\theta(Y_t)$};
    
    \begin{scope}[on background layer]
        \node[draw, thick, dashed, rounded corners=6pt, fill=gray!5, inner sep=18pt,
              drop shadow={opacity=0.15, shadow xshift=3pt, shadow yshift=-3pt},
              fit=(Z_yt) (Z_gy),
              label={[align=center, font=\Large\bfseries, text=black]above:Koopman State\\$Z_t = [Y_t; g_\theta(Y_t)]$}] (Zbox) {};
    \end{scope}

    % --- 第一层连线 ---
    % 状态路径
    \draw[myarrow] (st.east)   -- (encF.west);
    \draw[myarrow] (encF.east) -- (Yt.west);
    \draw[myarrow] (Yt.east)   -- (lift.west);
    \draw[myarrow] (lift.east) -- (gy.west);
    % Y_t 的跨越线，连至右侧的 Koopman State 虚线框内
    \draw[myarrow] (Yt.north)  -- ++(0, 2.5) -| (Z_yt.north);
    % g_\theta(Y_t) 进入拼合
    \draw[myarrow] (gy.east)   -- ++(0.6, 0) |- (Z_gy.west);
    
    % 动作路径
    % 【修改】缩短 s_t 向下延伸的距离到 -0.8，避免与 a_t 框体相交
    \draw[myarrow] (st.south)  -- ++(0, -0.8) -| (plus1.north); 
    \draw[myarrow] (at.east)   -- (plus1.west);
    \draw[myarrow] (plus1.east)-- (encG.west);
    \draw[myarrow] (encG.east) -- (ut.west);

    % ==========================================
    % 第二层 (Middle Row): 线性演化 (承上启下)
    % ==========================================
    
    % 精准位于第一层 Zbox 和 Ut 的正下方
    \node[net]  (matA)  at (16.5, -7.5) {Koopman\\Matrix $A$};
    \node[net]  (matB)  at (9.5,  -7.5) {Koopman\\Matrix $B$};
    \node[plus] (plus2) at (13.0, -7.5) {$\mathbf{+}$}; % 汇合点居中对齐
    
    % --- 第二层连线 ---
    \draw[myarrow] (Zbox.south) -- (matA.north);
    \draw[myarrow] (ut.south)   -- (matB.north);
    \draw[myarrow] (matA.west)  -- (plus2.east);
    \draw[myarrow] (matB.east)  -- (plus2.west);

    % ==========================================
    % 第三层 (Bottom Row): 预测与解码 (自右向左折叠)
    % ==========================================
    
    % Z_t+1 位于第二层相加节点的正下方
    \node[vec=sblue] (Zt1) at (13.0, -11.5) {$Z_{t+1}$};
    
    % --- 向右分支：K步预测 ---
    \node[vec=sblue] (ZtK) at (17.5, -11.5) {$Z_{t+K}$};
    
    % --- 向左折叠分支：解码至物理状态 ---
    \node[net]       (Y)   at (9.5, -11.5)  {$Y = CZ$};
    \node[net]       (dec) at (5.5, -11.5)  {Decoder\\$F^{-1}$};
    \node[vec=sblue] (st1) at (0,   -11.5)  {$s_{t+1}$}; % 与顶层 s_t 完美左侧垂直对齐
    
    % --- 第三层连线 ---
    % 演化输出 (标注公式)
    \draw[myarrow] (plus2.south) -- node[right=0.2cm, font=\Large\bfseries] {$Z_{t+1} = A Z_t + B U_t$} (Zt1.north);
    
    % 【修改】K步预测虚线，分离 K-step 和 evolution 至线上与线下
    \draw[mydash]  (Zt1.east) -- 
        node[above, font=\Large\itshape] {$K$-step} 
        node[below, font=\Large\itshape] {evolution} (ZtK.west);
    
    % 核心向左倒车路径 (Right-to-Left)
    \draw[myarrow] (Zt1.west) -- (Y.east);
    \draw[myarrow] (Y.west)   -- (dec.east);
    \draw[myarrow] (dec.west) -- (st1.east);

\end{tikzpicture}
\end{document}






















# from graphviz import Digraph

# dot = Digraph('Koopman_Architecture')

# # 极致紧凑的全局设置
# # dpi='300' 保证矢量转图时的超高清晰度
# dot.attr(rankdir='LR', splines='spline', nodesep='0.35', ranksep='0.4', dpi='300')

# # =======================
# # 辅助函数: 画向量矩阵块
# # =======================
# def vector(label, color):
#     return f'''<
#     <TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0">
#         <TR><TD BGCOLOR="{color}" WIDTH="20" HEIGHT="10"></TD></TR>
#         <TR><TD BGCOLOR="{color}" WIDTH="20" HEIGHT="10"></TD></TR>
#         <TR><TD BGCOLOR="{color}" WIDTH="20" HEIGHT="10"></TD></TR>
#         <TR><TD>{label}</TD></TR>
#     </TABLE>
#     >'''

# # =======================
# # Stage 1: 输入层
# # =======================
# dot.node('a_t', vector('<I>a<SUB>t</SUB></I>', 'paleturquoise'), shape='none')
# dot.node('s_t', vector('<I>s<SUB>t</SUB></I>', 'lightblue'), shape='none')

# # 强制将 a_t 和 s_t 上下对齐
# with dot.subgraph() as s:
#     s.attr(rank='same')
#     s.node('a_t')
#     s.node('s_t')

# # =======================
# # Stage 2: 编码器层
# # =======================
# dot.node('sum1', '<<I>⊕</I>>', shape='circle', width='0.3', fixedsize='true')
# dot.node('G', 'Action Encoder\nG', shape='box', style='rounded')
# dot.node('F', 'State Encoder\nF', shape='box', style='rounded')

# dot.node('U_t', vector('<I>U<SUB>t</SUB></I>', 'pink'), shape='none')
# dot.node('Y_t', vector('<I>Y<SUB>t</SUB></I>', 'lightgray'), shape='none')

# # 连接
# dot.edge('a_t', 'sum1')
# dot.edge('s_t', 'sum1')
# dot.edge('s_t', 'F')

# dot.edge('sum1', 'G')
# dot.edge('F', 'Y_t')
# dot.edge('G', 'U_t')

# # =======================
# # Stage 3: Lifting (提升网络)
# # =======================
# dot.node('lift', '<Lifting Net<BR/><I>g_θ</I>>', shape='box', style='rounded')
# dot.node('gY', vector('<I>g_θ(Y<SUB>t</SUB>)</I>', 'mediumpurple'), shape='none')

# # Koopman 状态簇 (虚线框)
# with dot.subgraph(name='cluster_Z') as c:
#     c.attr(label='<<I>Koopman State Z<SUB>t</SUB> =<BR/>[Y<SUB>t</SUB>; g_θ(Y<SUB>t</SUB>)]</I>>', style='dashed', margin='10')
#     c.node('Z_top', vector('<I>Y<SUB>t</SUB></I>', 'lightgray'), shape='none')
#     c.node('Z_bottom', vector('<I>g_θ(Y<SUB>t</SUB>)</I>', 'mediumpurple'), shape='none')

# dot.edge('Y_t', 'Z_top')
# dot.edge('Y_t', 'lift')
# dot.edge('lift', 'gY')
# dot.edge('gY', 'Z_bottom')

# # =======================
# # Stage 4: Koopman 动力学
# # =======================
# dot.node('B', 'Koopman\nMatrix B', shape='box')
# dot.node('A', 'Koopman\nMatrix A', shape='box')
# dot.node('sum2', '<<I>⊕</I>>', shape='circle', width='0.3', fixedsize='true')

# dot.node('Z_t1', vector('<I>Z<SUB>t+1</SUB></I>', 'lightblue'), shape='none')

# dot.edge('U_t', 'B')
# dot.edge('Z_top', 'A')
# dot.edge('Z_bottom', 'A')

# dot.edge('B', 'sum2')
# dot.edge('A', 'sum2')
# dot.edge('sum2', 'Z_t1', label='<<I>Z<SUB>t+1</SUB> = A Z<SUB>t</SUB><BR/>+ B U<SUB>t</SUB></I>>')

# # 强制矩阵 A 和 B 上下对齐
# with dot.subgraph() as s:
#     s.attr(rank='same')
#     s.node('B')
#     s.node('A')

# # =======================
# # Stage 5: U型折叠与解码器 (核心优化区)
# # =======================
# dot.node('Z_tk', vector('<I>Z<SUB>t+K</SUB></I>', 'lightblue'), shape='none')
# dot.node('extract', '<<I>Y = CZ</I>>', shape='box')
# dot.node('Finv', '<Decoder<BR/><I>F<SUP>-1</SUP></I>>', shape='box', style='rounded')
# dot.node('s_t1', vector('<I>s<SUB>t+1</SUB></I>', 'lightblue'), shape='none')

# # 【分支1】继续向右演化
# # 这里顺便修复了之前 K-step 换行失效的 Bug
# dot.edge('Z_t1', 'Z_tk', style='dashed', label='<<I>K-step<BR/>evolution</I>>')

# # 【分支2】垂直向下折叠提取 Y
# with dot.subgraph() as s:
#     s.attr(rank='same')
#     s.node('Z_t1')
#     s.node('extract')
# dot.edge('Z_t1', 'extract')

# # 【分支3】向左“倒车”完成解码 (利用 dir='back' 反转箭头视觉方向)
# dot.edge('Finv', 'extract', dir='back')
# dot.edge('s_t1', 'Finv', dir='back')

# # =======================
# # 渲染与输出
# # =======================
# output_path = '/home/nng/koopman_project/lunwen/koopman_architecture'

# # 输出矢量图
# dot.render(output_path, format='pdf', view=False)
# dot.render(output_path, format='svg', view=False)

# print(f"✅ 搞定！【U型折叠版】4:3架构图已生成！")