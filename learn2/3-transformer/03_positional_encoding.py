#coding=utf-8
"""
=============================================================================
03. 位置编码 (Positional Encoding)
=============================================================================

【为什么需要位置编码？】
Transformer 的自注意力机制是"位置无关"的：
- 它只关注内容之间的关系
- 不知道词在句子中的顺序

但语言中位置很重要：
- "猫吃鱼" vs "鱼吃猫" - 完全不同的意思！

位置编码的作用：给模型提供位置信息

【正弦/余弦位置编码】
原始论文 "Attention is All You Need" 使用的方法：

PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

其中:
- pos: 位置索引 (0, 1, 2, ...)
- i: 维度索引
- d_model: 嵌入维度

【为什么用正弦/余弦？】
1. 有界：值在 [-1, 1] 之间，不会因位置增大而爆炸
2. 周期性：不同频率的正弦波可以表示不同尺度的位置关系
3. 可学习相对位置：PE(pos+k) 可以表示为 PE(pos) 的线性函数

=============================================================================
"""

import tensorflow as tf
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


# ============================================================================
# 第一部分：正弦/余弦位置编码
# ============================================================================

def get_positional_encoding(max_seq_len, d_model):
    """
    生成正弦/余弦位置编码

    参数:
        max_seq_len: 最大序列长度
        d_model: 模型维度（嵌入维度）

    返回:
        位置编码矩阵, shape = (1, max_seq_len, d_model)
    """
    # 创建位置索引 [0, 1, 2, ..., max_seq_len-1]
    positions = np.arange(max_seq_len)[:, np.newaxis]  # (max_seq_len, 1)

    # 创建维度索引
    dims = np.arange(d_model)[np.newaxis, :]  # (1, d_model)

    # 计算角度
    # 对于偶数维度 2i，使用 sin；对于奇数维度 2i+1，使用 cos
    # 分母: 10000^(2i/d_model)
    angles = positions / np.power(10000, (2 * (dims // 2)) / d_model)

    # 偶数索引用 sin
    angles[:, 0::2] = np.sin(angles[:, 0::2])

    # 奇数索引用 cos
    angles[:, 1::2] = np.cos(angles[:, 1::2])

    # 添加 batch 维度
    pos_encoding = angles[np.newaxis, ...]  # (1, max_seq_len, d_model)

    return tf.cast(pos_encoding, dtype=tf.float32)


class PositionalEncoding(tf.keras.layers.Layer):
    """
    位置编码层

    将位置编码加到输入嵌入上：
    output = embedding + positional_encoding
    """

    def __init__(self, max_seq_len, d_model):
        super().__init__()
        self.d_model = d_model
        # 预计算位置编码（不需要训练）
        self.pos_encoding = get_positional_encoding(max_seq_len, d_model)

    def call(self, x):
        """
        参数:
            x: 输入嵌入, shape = (batch_size, seq_len, d_model)

        返回:
            加上位置编码后的嵌入
        """
        seq_len = tf.shape(x)[1]

        # 缩放嵌入（原论文做法）
        x = x * tf.math.sqrt(tf.cast(self.d_model, tf.float32))

        # 加上位置编码（只取需要的长度）
        x = x + self.pos_encoding[:, :seq_len, :]

        return x


# ============================================================================
# 第二部分：可视化位置编码
# ============================================================================

def visualize_positional_encoding():
    """可视化位置编码的模式"""
    print("=" * 60)
    print("可视化位置编码")
    print("=" * 60)

    max_seq_len = 100
    d_model = 128

    # 获取位置编码
    pos_encoding = get_positional_encoding(max_seq_len, d_model)
    pos_encoding = pos_encoding[0].numpy()  # (max_seq_len, d_model)

    # 绘制热图
    plt.figure(figsize=(15, 8))

    plt.subplot(2, 2, 1)
    plt.imshow(pos_encoding, cmap='RdBu', aspect='auto')
    plt.colorbar()
    plt.xlabel('Embedding Dimension')
    plt.ylabel('Position')
    plt.title('Position Encoding Heatmap')

    # 绘制不同维度的编码曲线
    plt.subplot(2, 2, 2)
    for dim in [0, 1, 10, 11, 50, 51]:
        label = f'dim {dim} ({"sin" if dim % 2 == 0 else "cos"})'
        plt.plot(pos_encoding[:, dim], label=label)
    plt.xlabel('Position')
    plt.ylabel('Value')
    plt.title('Position Encoding for Different Dimensions')
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)

    # 绘制不同位置的编码向量
    plt.subplot(2, 2, 3)
    for pos in [0, 10, 20, 50]:
        plt.plot(pos_encoding[pos, :50], label=f'pos {pos}', alpha=0.7)
    plt.xlabel('Dimension')
    plt.ylabel('Value')
    plt.title('Position Encoding Vectors (first 50 dims)')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 绘制位置之间的相似度
    plt.subplot(2, 2, 4)
    # 计算位置之间的余弦相似度
    similarity = np.zeros((20, 20))
    for i in range(20):
        for j in range(20):
            dot = np.dot(pos_encoding[i], pos_encoding[j])
            norm_i = np.linalg.norm(pos_encoding[i])
            norm_j = np.linalg.norm(pos_encoding[j])
            similarity[i, j] = dot / (norm_i * norm_j)

    plt.imshow(similarity, cmap='RdBu', vmin=-1, vmax=1)
    plt.colorbar(label='Cosine Similarity')
    plt.xlabel('Position')
    plt.ylabel('Position')
    plt.title('Position Encoding Similarity (first 20 positions)')

    plt.tight_layout()
    plt.savefig('positional_encoding_visualization.png', dpi=150)
    plt.show()
    print("\n位置编码可视化已保存为 positional_encoding_visualization.png")


# ============================================================================
# 第三部分：理解位置编码的性质
# ============================================================================

def explain_positional_encoding():
    """解释位置编码的关键性质"""
    print("\n" + "=" * 60)
    print("位置编码的关键性质")
    print("=" * 60)

    d_model = 64
    max_seq_len = 100

    pos_encoding = get_positional_encoding(max_seq_len, d_model)[0].numpy()

    # 性质1：有界性
    print("\n【性质1: 有界性】")
    print(f"位置编码最大值: {pos_encoding.max():.4f}")
    print(f"位置编码最小值: {pos_encoding.min():.4f}")
    print("值都在 [-1, 1] 之间，不会随位置增大而爆炸")

    # 性质2：相对位置
    print("\n【性质2: 相对位置可学习】")
    print("相邻位置的编码差异应该相似：")
    for pos in [0, 10, 50]:
        diff = np.linalg.norm(pos_encoding[pos + 1] - pos_encoding[pos])
        print(f"  ||PE({pos+1}) - PE({pos})|| = {diff:.4f}")

    # 性质3：不同频率
    print("\n【性质3: 多频率特征】")
    print("低维度: 高频变化 (捕捉局部位置)")
    print("高维度: 低频变化 (捕捉全局位置)")

    # 展示不同频率
    print("\n频率分析 (观察值变化的周期):")
    for dim in [0, 20, 40, 60]:
        # 找到从正到负的零交叉点来估计周期
        signal = pos_encoding[:, dim]
        zero_crossings = np.where(np.diff(np.sign(signal)))[0]
        if len(zero_crossings) >= 2:
            period = np.mean(np.diff(zero_crossings)) * 2
            print(f"  维度 {dim}: 周期 ≈ {period:.1f} 位置")
        else:
            print(f"  维度 {dim}: 周期 > {max_seq_len} 位置")


# ============================================================================
# 第四部分：可学习的位置编码
# ============================================================================

class LearnablePositionalEncoding(tf.keras.layers.Layer):
    """
    可学习的位置编码

    与正弦/余弦不同，这种方法让位置编码成为可训练的参数。
    GPT、BERT 等模型使用这种方式。

    优点：可以学习到任务特定的位置模式
    缺点：无法推广到训练时未见过的长度
    """

    def __init__(self, max_seq_len, d_model):
        super().__init__()
        self.d_model = d_model
        # 创建可训练的位置嵌入
        self.pos_embedding = tf.keras.layers.Embedding(
            input_dim=max_seq_len,
            output_dim=d_model,
            name='position_embedding'
        )

    def call(self, x):
        """
        参数:
            x: 输入嵌入, shape = (batch_size, seq_len, d_model)

        返回:
            加上位置编码后的嵌入
        """
        seq_len = tf.shape(x)[1]

        # 创建位置索引 [0, 1, 2, ..., seq_len-1]
        positions = tf.range(seq_len)

        # 获取位置嵌入
        pos_emb = self.pos_embedding(positions)  # (seq_len, d_model)

        # 缩放并加上位置编码
        x = x * tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x = x + pos_emb

        return x


def compare_positional_encodings():
    """比较两种位置编码方法"""
    print("\n" + "=" * 60)
    print("正弦/余弦 vs 可学习位置编码")
    print("=" * 60)

    comparison = """
┌────────────────────┬─────────────────────────┬─────────────────────────┐
│       特性         │    正弦/余弦编码        │    可学习编码           │
├────────────────────┼─────────────────────────┼─────────────────────────┤
│ 参数量             │ 0 (不需要训练)          │ max_len × d_model      │
├────────────────────┼─────────────────────────┼─────────────────────────┤
│ 泛化到更长序列     │ 可以                    │ 不可以                  │
├────────────────────┼─────────────────────────┼─────────────────────────┤
│ 相对位置           │ 天然支持                │ 需要学习                │
├────────────────────┼─────────────────────────┼─────────────────────────┤
│ 任务适应性         │ 固定                    │ 可学习适应任务          │
├────────────────────┼─────────────────────────┼─────────────────────────┤
│ 使用者             │ 原始 Transformer        │ GPT, BERT              │
└────────────────────┴─────────────────────────┴─────────────────────────┘
"""
    print(comparison)


# ============================================================================
# 第五部分：测试位置编码
# ============================================================================

def test_positional_encoding():
    """测试位置编码层"""
    print("\n" + "=" * 60)
    print("测试位置编码层")
    print("=" * 60)

    d_model = 64
    max_seq_len = 100
    batch_size = 2
    seq_len = 20

    # 创建位置编码层
    pos_encoding_layer = PositionalEncoding(max_seq_len, d_model)

    # 模拟词嵌入输入
    embeddings = tf.random.normal((batch_size, seq_len, d_model))

    # 加上位置编码
    output = pos_encoding_layer(embeddings)

    print(f"\n输入嵌入 shape: {embeddings.shape}")
    print(f"输出 shape: {output.shape}")

    # 验证输出和输入的差异就是位置编码
    diff = output[0] - embeddings[0] * tf.math.sqrt(tf.cast(d_model, tf.float32))
    expected_pe = get_positional_encoding(max_seq_len, d_model)[0, :seq_len, :]

    print(f"\n验证: output - scaled_input 是否等于位置编码?")
    print(f"差异的最大绝对值: {tf.reduce_max(tf.abs(diff - expected_pe)).numpy():.2e}")
    print("(接近0说明正确)")


# ============================================================================
# 主函数
# ============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Transformer 学习 - 03. 位置编码")
    print("=" * 60)

    # 1. 可视化位置编码
    visualize_positional_encoding()

    # 2. 解释位置编码性质
    explain_positional_encoding()

    # 3. 比较两种方法
    compare_positional_encodings()

    # 4. 测试位置编码层
    test_positional_encoding()

    print("\n" + "=" * 60)
    print("下一步: 04_transformer_block.py - Transformer 块")
    print("=" * 60)
