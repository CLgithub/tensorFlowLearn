#coding=utf-8
"""
=============================================================================
01. 自注意力机制 (Self-Attention)
=============================================================================

这是 Transformer 最核心的组件。

【核心思想】
对于序列中的每个位置，计算它与其他所有位置的相关性（注意力权重），
然后用这些权重对所有位置的信息进行加权求和。

【Q、K、V 的直觉理解】
- Query (Q): "我在找什么？" - 当前位置发出的查询
- Key (K): "我有什么？" - 每个位置的标识/索引
- Value (V): "我的内容是什么？" - 每个位置实际的信息

举例：图书馆找书
- Q: 你想找的书的关键词（如"机器学习"）
- K: 每本书的标签/索引
- V: 每本书的实际内容
- 注意力权重: Q 和 K 的匹配程度决定你会参考哪些书
- 输出: 根据匹配程度加权后的书籍内容组合

【公式】
Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) * V

其中 sqrt(d_k) 是缩放因子，防止点积值过大导致 softmax 饱和

=============================================================================
"""

import tensorflow as tf
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# ============================================================================
# 第一部分：从零实现 Scaled Dot-Product Attention
# ============================================================================

def scaled_dot_product_attention(query, key, value, mask=None):
    """
    计算缩放点积注意力

    参数:
        query: 查询张量, shape = (..., seq_len_q, d_k)
        key: 键张量, shape = (..., seq_len_k, d_k)
        value: 值张量, shape = (..., seq_len_v, d_v)，通常 seq_len_k == seq_len_v
        mask: 可选的掩码张量

    返回:
        output: 注意力输出
        attention_weights: 注意力权重（用于可视化）
    """

    # 步骤1: 计算 Q 和 K 的点积
    # matmul 会对最后两个维度做矩阵乘法
    # query: (..., seq_len_q, d_k)
    # key^T: (..., d_k, seq_len_k)
    # 结果: (..., seq_len_q, seq_len_k)
    matmul_qk = tf.matmul(query, key, transpose_b=True)

    # 步骤2: 缩放
    # 为什么要缩放？当 d_k 很大时，点积的值会很大，
    # 导致 softmax 后的梯度非常小（进入饱和区）
    d_k = tf.cast(tf.shape(key)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(d_k)

    # 步骤3: 应用掩码（可选）
    # 掩码用于：1) padding掩码 2) 因果掩码（生成任务中防止看到未来）
    if mask is not None:
        # 将 mask 为 1 的位置设为极小值，softmax 后接近 0
        scaled_attention_logits += (mask * -1e9)

    # 步骤4: Softmax 归一化，得到注意力权重
    # 每一行的权重和为 1
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)

    # 步骤5: 用注意力权重对 V 加权求和
    output = tf.matmul(attention_weights, value)

    return output, attention_weights


# ============================================================================
# 第二部分：简单示例 - 理解注意力机制
# ============================================================================

def simple_attention_example():
    """
    一个简单的例子来理解自注意力

    假设我们有一个句子："我 爱 中国"
    每个词用一个 4 维向量表示
    """
    print("=" * 60)
    print("简单示例：理解自注意力机制")
    print("=" * 60)

    # 模拟 3 个词的嵌入向量，每个词 4 维
    # 实际中这些向量是学习得到的，这里我们手动设置便于理解
    embeddings = tf.constant([
        [1.0, 0.0, 1.0, 0.0],   # "我"
        [0.0, 1.0, 1.0, 0.0],   # "爱"
        [1.0, 1.0, 0.0, 1.0],   # "中国"
    ], dtype=tf.float32)

    # 在自注意力中，Q、K、V 来自同一个输入
    # 这里简化：直接用 embeddings 作为 Q、K、V
    # (实际中会有可学习的投影矩阵 W_Q, W_K, W_V)

    # 添加 batch 维度: (1, 3, 4)
    query = tf.expand_dims(embeddings, 0)
    key = tf.expand_dims(embeddings, 0)
    value = tf.expand_dims(embeddings, 0)

    # 计算注意力
    output, attention_weights = scaled_dot_product_attention(query, key, value)

    print("\n输入词向量 (3个词，每个4维):")
    print(embeddings.numpy())

    print("\n注意力权重矩阵 (3x3):")
    print("每一行表示一个词对所有词的注意力分布")
    weights = attention_weights[0].numpy()
    print(weights)

    print("\n解读注意力权重:")
    words = ["我", "爱", "中国"]
    for i, word in enumerate(words):
        print(f"  '{word}' 的注意力分布: ", end="")
        for j, target in enumerate(words):
            print(f"'{target}'={weights[i,j]:.3f}  ", end="")
        print()

    print("\n输出 (加权求和后的新表示):")
    print(output[0].numpy())

    return attention_weights


# ============================================================================
# 第三部分：可视化注意力权重
# ============================================================================

def visualize_attention(attention_weights, tokens):
    """
    将注意力权重可视化为热图

    参数:
        attention_weights: 注意力权重矩阵
        tokens: token 列表（用于标注轴）
    """
    weights = attention_weights[0].numpy()

    plt.figure(figsize=(8, 6))
    plt.imshow(weights, cmap='Blues', aspect='auto')
    plt.colorbar(label='Attention Weight')

    # 设置刻度标签
    plt.xticks(range(len(tokens)), tokens, fontsize=12)
    plt.yticks(range(len(tokens)), tokens, fontsize=12)

    # 在每个格子中显示数值
    for i in range(len(tokens)):
        for j in range(len(tokens)):
            plt.text(j, i, f'{weights[i,j]:.2f}',
                    ha='center', va='center', fontsize=10,
                    color='white' if weights[i,j] > 0.5 else 'black')

    plt.xlabel('Key (被关注的词)', fontsize=12)
    plt.ylabel('Query (发起查询的词)', fontsize=12)
    plt.title('自注意力权重热图\n每行表示一个词对其他词的注意力分布', fontsize=14)
    plt.tight_layout()
    plt.savefig('attention_visualization.png', dpi=150)
    plt.show()
    print("\n注意力热图已保存为 attention_visualization.png")


# ============================================================================
# 第四部分：带有可学习参数的自注意力层
# ============================================================================

class SelfAttentionLayer(tf.keras.layers.Layer):
    """
    完整的自注意力层，包含可学习的投影矩阵

    Q = X @ W_Q
    K = X @ W_K
    V = X @ W_V

    其中 W_Q, W_K, W_V 是可学习的权重矩阵
    """

    def __init__(self, d_model):
        """
        参数:
            d_model: 模型维度（输入和输出的维度）
        """
        super().__init__()
        self.d_model = d_model

        # 创建三个投影矩阵
        self.W_Q = tf.keras.layers.Dense(d_model, use_bias=False, name='query_proj')
        self.W_K = tf.keras.layers.Dense(d_model, use_bias=False, name='key_proj')
        self.W_V = tf.keras.layers.Dense(d_model, use_bias=False, name='value_proj')

    def call(self, x, mask=None):
        """
        前向传播

        参数:
            x: 输入张量, shape = (batch_size, seq_len, d_model)
            mask: 可选的掩码

        返回:
            output: 注意力输出
            attention_weights: 注意力权重
        """
        # 线性投影
        query = self.W_Q(x)  # (batch_size, seq_len, d_model)
        key = self.W_K(x)    # (batch_size, seq_len, d_model)
        value = self.W_V(x)  # (batch_size, seq_len, d_model)

        # 计算注意力
        output, attention_weights = scaled_dot_product_attention(
            query, key, value, mask
        )

        return output, attention_weights


def test_self_attention_layer():
    """测试自注意力层"""
    print("\n" + "=" * 60)
    print("测试可学习的自注意力层")
    print("=" * 60)

    # 创建自注意力层
    d_model = 64
    attention_layer = SelfAttentionLayer(d_model)

    # 模拟输入: batch_size=2, seq_len=5, d_model=64
    batch_size = 2
    seq_len = 5
    x = tf.random.normal((batch_size, seq_len, d_model))

    # 前向传播
    output, weights = attention_layer(x)

    print(f"\n输入 shape: {x.shape}")
    print(f"输出 shape: {output.shape}")
    print(f"注意力权重 shape: {weights.shape}")

    # 验证注意力权重每行和为 1
    row_sums = tf.reduce_sum(weights[0], axis=-1)
    print(f"\n注意力权重每行之和 (应该都是1): {row_sums.numpy()}")

    # 打印可学习参数数量
    total_params = sum([tf.reduce_prod(var.shape).numpy()
                       for var in attention_layer.trainable_variables])
    print(f"\n可学习参数数量: {total_params}")
    print(f"  - W_Q: {d_model} x {d_model} = {d_model**2}")
    print(f"  - W_K: {d_model} x {d_model} = {d_model**2}")
    print(f"  - W_V: {d_model} x {d_model} = {d_model**2}")


# ============================================================================
# 第五部分：因果掩码（用于文本生成）
# ============================================================================

def create_causal_mask(seq_len):
    """
    创建因果掩码（也叫 look-ahead mask）

    在生成任务中，每个位置只能看到自己和之前的位置，不能看到未来

    例如 seq_len=4 的掩码:
    [[0, 1, 1, 1],   # 位置0 只能看位置0
     [0, 0, 1, 1],   # 位置1 能看位置0,1
     [0, 0, 0, 1],   # 位置2 能看位置0,1,2
     [0, 0, 0, 0]]   # 位置3 能看所有位置

    0 表示可以看，1 表示要遮挡
    """
    # 创建上三角矩阵（对角线以上为1）
    mask = 1 - tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0)
    return mask


def test_causal_mask():
    """测试因果掩码"""
    print("\n" + "=" * 60)
    print("因果掩码示例 (用于文本生成)")
    print("=" * 60)

    seq_len = 5
    mask = create_causal_mask(seq_len)

    print(f"\n序列长度: {seq_len}")
    print("\n因果掩码矩阵 (0=可见, 1=遮挡):")
    print(mask.numpy())

    # 用因果掩码计算注意力
    print("\n使用因果掩码计算注意力:")

    # 模拟输入
    x = tf.random.normal((1, seq_len, 16))

    # 计算注意力
    output, weights = scaled_dot_product_attention(x, x, x, mask)

    print("\n注意力权重矩阵:")
    print("(注意：每行只有对角线及左边有值，右边都是0)")
    print(np.round(weights[0].numpy(), 3))


# ============================================================================
# 主函数
# ============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Transformer 学习 - 01. 自注意力机制")
    print("=" * 60)

    # 1. 简单示例
    attention_weights = simple_attention_example()

    # 2. 可视化
    tokens = ["我", "爱", "中国"]
    visualize_attention(attention_weights, tokens)

    # 3. 测试可学习的注意力层
    test_self_attention_layer()

    # 4. 测试因果掩码
    test_causal_mask()

    print("\n" + "=" * 60)
    print("下一步: 02_multi_head_attention.py - 多头注意力")
    print("=" * 60)
