#修改成了latex版本， https://www.overleaf.com/上直接画




\documentclass[tikz,border=20pt]{standalone}

\usepackage{amsmath} % 确保公式环境稳定加载

% 引入排版专家常用的 TikZ 库
\usetikzlibrary{shapes.geometric, arrows.meta, positioning, calc, shadows, backgrounds, fit}

% =========================
% 定义学术论文标准颜色
% =========================
\definecolor{xgre}{RGB}{220, 245, 220}  % 浅绿色 (x 系列)
\definecolor{yblu}{RGB}{210, 235, 255}  % 浅蓝色 (y 系列)
\definecolor{upink}{RGB}{255, 215, 220} % 浅粉色 (u 系列)
\definecolor{koob}{RGB}{0, 51, 102}     % 深蓝色 (Koopman 算子)
\definecolor{lossred}{RGB}{220, 50, 47} % 损失函数红
\definecolor{lossblu}{RGB}{38, 139, 210}% 线性损失蓝

\begin{document}

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
        drop shadow={opacity=0.15, shadow xshift=2pt, shadow yshift=-2pt},
        path picture={
            \draw[thick] ($(path picture bounding box.north west)!0.333!(path picture bounding box.south west)$) -- 
                         ($(path picture bounding box.north east)!0.333!(path picture bounding box.south east)$);
            \draw[thick] ($(path picture bounding box.north west)!0.667!(path picture bounding box.south west)$) -- 
                         ($(path picture bounding box.north east)!0.667!(path picture bounding box.south east)$);
        }
    },
    % -------------------------
    % 定义网络节点 (圆角矩形)
    % -------------------------
    enc/.style={draw, thick, rounded corners=6pt, fill=yblu!30, minimum width=2.4cm, minimum height=1.2cm},
    dec/.style={draw, thick, rounded corners=6pt, fill=xgre!30, minimum width=2.4cm, minimum height=1.2cm},
    % -------------------------
    % 定义算子节点 (圆形)
    % -------------------------
    op/.style={circle, draw, thick, fill=koob, text=white, minimum size=1.4cm, font=\large\bfseries},
    % -------------------------
    % 连线样式
    % -------------------------
    mainline/.style={->, thick, line width=1.5pt},
    lossline_red/.style={->, dashed, thick, color=lossred, line width=1.2pt},
    lossline_blu/.style={<->, dashed, thick, color=lossblu, line width=1.2pt}
]

    % ==========================================
    % 第一阶段 (Top Segment: t=0)
    % ==========================================
    
    % 中央主轴: Encoder & y_t
    \node[enc] (enc0) at (0, 0) {Encoder $F$};
    \node[vec=yblu, below=1.2cm of enc0] (yt) {$y_t$};
    
    % 左侧: x_t
    \node[vec=xgre, left=3cm of enc0] (xt) {$x_t$};
    
    % 右侧: Decoder & x_hat_t
    \node[dec, right=1.5cm of yt] (dec0) {Decoder $F^{-1}$};
    \node[vec=xgre, right=1.5cm of dec0] (xhat0) {$\hat{x}_t$};
    
    % 推进: 第一个 Koopman 算子
    \node[op, below=2cm of yt] (op1) {A, B};
    \node[vec=upink, right=1.2cm of op1] (ut) {$u_t$};

    % --- 连线 ---
    \draw[mainline] (xt) -- (enc0);
    \draw[mainline] (enc0) -- (yt);
    \draw[mainline] (yt) -- (dec0);
    \draw[mainline] (dec0) -- (xhat0);
    \draw[mainline] (yt) -- (op1);
    \draw[mainline] (ut) -- (op1);
    
    % L_rec 大包抄 (顶部)
    \draw[lossline_red] (xhat0.north) -- ++(0, 3cm) -| (xt.north) 
        node[pos=0.25, above, font=\Large\bfseries] {$L_{rec}$};

    % ==========================================
    % 第二阶段 (Middle Segment: t=1)
    % ==========================================
    
    % 中央主轴: y_hat_t+1
    \node[vec=yblu, below=2cm of op1] (yhat1) {$\hat{y}_{t+1}$};
    
    % 左侧对齐: x_t+1 & F(x_t+1)
    \node[enc, left=2.5cm of yhat1] (enc1) {Encoder $F$};
    \node[vec=xgre, left=1.2cm of enc1] (xt1) {$x_{t+1}$};
    \node[vec=yblu, below=0.8cm of enc1] (fxt1) {$F(x_{t+1})$};
    
    % 右侧: Decoder & x_hat_t+1
    \node[dec, right=1.5cm of yhat1] (dec1) {Decoder $F^{-1}$};
    \node[vec=xgre, right=1.5cm of dec1] (xhat1) {$\hat{x}_{t+1}$};
    
    % 继续推进: 第二个 Koopman 算子
    \node[op, below=2.5cm of yhat1] (op2) {A, B};
    \node[vec=upink, right=1.2cm of op2] (ut1) {$u_{t+1}$};

    % --- 连线 ---
    \draw[mainline] (op1) -- (yhat1);
    \draw[mainline] (xt1) -- (enc1);
    \draw[mainline] (enc1) -- (fxt1);
    \draw[lossline_blu] (yhat1) -- (fxt1) node[midway, above] {$L_{lin}$};
    
    \draw[mainline] (yhat1) -- (dec1);
    \draw[mainline] (dec1) -- (xhat1);
    
    % 【重点修复】中间层 L_pred 向上方绕行 (避免与主干横线穿模)
    \draw[lossline_red] (xhat1.north) -- ++(0, 1.2cm) -- ($(xt1.north) + (0, 1.2cm)$) 
        node[midway, above, font=\Large\bfseries] {$L_{pred}$} -- (xt1.north);
    
    \draw[mainline] (yhat1) -- (op2);
    \draw[mainline] (ut1) -- (op2);

    % ==========================================
    % 第三阶段 (Bottom Segment: t=2)
    % ==========================================
    
    % 中央主轴: y_hat_t+2
    \node[vec=yblu, below=2cm of op2] (yhat2) {$\hat{y}_{t+2}$};
    
    % 右侧: Decoder & x_hat_t+2
    \node[dec, right=1.5cm of yhat2] (dec2) {Decoder $F^{-1}$};
    \node[vec=xgre, right=1.5cm of dec2] (xhat2) {$\hat{x}_{t+2}$};
    
    % 左侧: x_t+2
    \node[vec=xgre] (xt2) at (xt1 |- xhat2) {$x_{t+2}$};

    % --- 连线 ---
    \draw[mainline] (op2) -- (yhat2);
    \draw[mainline] (yhat2) -- (dec2);
    \draw[mainline] (dec2) -- (xhat2);
    
    % L_pred 大包抄 (底部)
    \draw[lossline_red] (xhat2.south) -- ++(0, -1.5cm) -| (xt2.south)
        node[pos=0.25, below, font=\Large\bfseries] {$L_{pred}$};

\end{tikzpicture}

\end{document}













# from graphviz import Digraph
# import os

# dot = Digraph('Koopman_Unrolled_Loss')

# # 设置论文级 DPI 和布局属性
# dot.attr(rankdir='LR', splines='spline', nodesep='0.6', ranksep='0.8', dpi='300')

# # =======================
# # 辅助函数: 绘制专业矩阵/向量块
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
# # Stage 1: Ground Truth (观测/真实值层)
# # =======================
# # 使用绿色系表示真实观测状态 x
# dot.node('x_t', vector('<I>x<SUB>t</SUB></I>', '#e5f5e0'), shape='none')
# dot.node('x_t1', vector('<I>x<SUB>t+1</SUB></I>', '#e5f5e0'), shape='none')
# dot.node('x_t2', vector('<I>x<SUB>t+2</SUB></I>', '#e5f5e0'), shape='none')

# # 使用橙色系表示外部控制输入 u
# dot.node('u_t', vector('<I>u<SUB>t</SUB></I>', '#fee0d2'), shape='none')
# dot.node('u_t1', vector('<I>u<SUB>t+1</SUB></I>', '#fee0d2'), shape='none')

# # 确保观测状态水平对齐
# with dot.subgraph() as s:
#     s.attr(rank='same')
#     s.node('x_t'); s.node('x_t1'); s.node('x_t2')

# # =======================
# # Stage 2: Koopman Latent Space (潜在演化层)
# # =======================
# # 编码器 F (蓝色系)
# dot.node('F', '<Encoder<BR/><I>F</I>>', shape='box', style='filled,rounded', fillcolor='#bdd7e7')

# # 潜在状态 y (浅蓝色)
# dot.node('y_t', vector('<I>y<SUB>t</SUB></I>', '#deebf7'), shape='none')
# dot.node('y_t1_hat', vector('<I>y&#770;<SUB>t+1</SUB></I>', '#deebf7'), shape='none')
# dot.node('y_t2_hat', vector('<I>y&#770;<SUB>t+2</SUB></I>', '#deebf7'), shape='none')

# # 线性转移算子 (A, B)
# dot.node('K1', '<<I>A, B</I>>', shape='circle', width='0.5', style='filled', fillcolor='#9ecae1')
# dot.node('K2', '<<I>A, B</I>>', shape='circle', width='0.5', style='filled', fillcolor='#9ecae1')

# # 时序演化连接
# dot.edge('x_t', 'F')
# dot.edge('F', 'y_t')
# dot.edge('y_t', 'K1')
# dot.edge('u_t', 'K1')
# dot.edge('K1', 'y_t1_hat')
# dot.edge('y_t1_hat', 'K2')
# dot.edge('u_t1', 'K2')
# dot.edge('K2', 'y_t2_hat')

# # =======================
# # Stage 3: Decoding & Reconstruction (解码与重构层)
# # =======================
# # 解码器 F^-1
# dot.node('Finv', '<Decoder<BR/><I>F<SUP>-1</SUP></I>>', shape='box', style='filled,rounded', fillcolor='#c7e9c0')

# # 预测出的观测状态 x_hat
# dot.node('x_t_hat', vector('<I>x&#770;<SUB>t</SUB></I>', '#edf8e9'), shape='none')
# dot.node('x_t1_hat', vector('<I>x&#770;<SUB>t+1</SUB></I>', '#edf8e9'), shape='none')
# dot.node('x_t2_hat', vector('<I>x&#770;<SUB>t+2</SUB></I>', '#edf8e9'), shape='none')

# # 解码路径
# dot.edge('y_t', 'Finv')
# dot.edge('y_t1_hat', 'Finv')
# dot.edge('y_t2_hat', 'Finv')
# dot.edge('Finv', 'x_t_hat')
# dot.edge('Finv', 'x_t1_hat')
# dot.edge('Finv', 'x_t2_hat')

# # =======================
# # Stage 4: Loss Functions (三类关键损失标注)
# # =======================
# # 1. 重构损失 L_rec (红色虚线)
# dot.edge('x_t_hat', 'x_t', dir='both', style='dashed', color='red', 
#          label='<<FONT COLOR=\"red\"><I>L<SUB>rec</SUB></I></FONT>>')

# # 2. 预测损失 L_pred (红色虚线)
# dot.edge('x_t1_hat', 'x_t1', dir='both', style='dashed', color='red', 
#          label='<<FONT COLOR=\"red\"><I>L<SUB>pred</SUB></I></FONT>>')
# dot.edge('x_t2_hat', 'x_t2', dir='both', style='dashed', color='red')

# # 3. 线性演化损失 L_lin (潜在空间蓝色虚线)
# dot.node('y_t1_true', vector('<I>F(x<SUB>t+1</SUB>)</I>', '#f0f0f0'), shape='none')
# dot.edge('x_t1', 'y_t1_true', style='dotted')
# dot.edge('y_t1_hat', 'y_t1_true', dir='both', style='dashed', color='blue', 
#          label='<<FONT COLOR=\"blue\"><I>L<SUB>lin</SUB></I></FONT>>')

# # =======================
# # 渲染与输出设置
# # =======================
# output_dir = '/home/nng/koopman_project/lunwen'
# if not os.path.exists(output_dir):
#     os.makedirs(output_dir)
# output_path = os.path.join(output_dir, 'koopman_time_unrolled_professional')

# dot.render(output_path, format='png', cleanup=True)
# dot.render(output_path, format='pdf', cleanup=True)

# print(f"✅ 专业版【时序展开损失图】已生成至: {output_path}.png")