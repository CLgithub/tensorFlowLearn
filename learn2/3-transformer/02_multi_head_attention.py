#coding=utf-8
"""
=============================================================================
02. 多头注意力 (Multi-Head Attention)
=============================================================================

【为什么需要多头？】
单头注意力只能学习一种"关注模式"。
多头注意力允许模型同时关注不同类型的信息：
- 某些头可能关注语法关系
- 某些头可能关注语义相似性
- 某些头可能关注位置相近的词

【核心思想】
1. 将 Q、K、V 分成多个头（head）
2. 每个头独立计算注意力
3. 将所有头的结果拼接起来
4. 通过线性变换得到最终输出

【公式】
MultiHead(Q, K, V) = Concat(head_1, ..., head_h) @ W_O

其中:
head_i = Attention(Q @ W_Q^i, K @ W_K^i, V @ W_V^i)

=============================================================================
"""

import tensorflow as tf
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# 导入上一节的注意力函数
# 注意：文件名以数字开头，需要特殊导入方式
import importlib.util
import sys
import os

# 获取当前文件所在目录
current_dir = os.path.dirname(os.path.abspath(__file__))

# 加载 01_self_attention 模块
spec = importlib.util.spec_from_file_location(
    "self_attention",
    os.path.join(current_dir, "01_self_attention.py")
)
self_attention_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(self_attention_module)

# 导入需要的函数
scaled_dot_product_attention = self_attention_module.scaled_dot_product_attention
create_causal_mask = self_attention_module.create_causal_mask


# ============================================================================
# 第一部分：从零实现多头注意力
# ============================================================================

class MultiHeadAttention(tf.keras.layers.Layer):
    """
    多头注意力层

    假设 d_model=512, num_heads=8:
    - 每个头的维度 d_k = d_v = 512/8 = 64
    - 8 个头并行计算，然后拼接
    """

    def __init__(self, d_model, num_heads):
        """
        参数:
            d_model: 模型维度（必须能被 num_heads 整除）
            num_heads: 注意力头的数量
        """
        super().__init__()
        self.num_heads = num_heads
        self.d_model = d_model

        # 确保 d_model 能被 num_heads 整除
        assert d_model % num_heads == 0, \
            f"d_model ({d_model}) 必须能被 num_heads ({num_heads}) 整除"

        # 每个头的维度
        self.depth = d_model // num_heads

        # 投影矩阵
        self.W_Q = tf.keras.layers.Dense(d_model, name='query_proj')
        self.W_K = tf.keras.layers.Dense(d_model, name='key_proj')
        self.W_V = tf.keras.layers.Dense(d_model, name='value_proj')

        # 输出投影矩阵
        self.W_O = tf.keras.layers.Dense(d_model, name='output_proj')

    def split_heads(self, x, batch_size):
        """
        将最后一个维度分成 (num_heads, depth)
        然后转置，使 head 维度在 seq_len 之前

        输入: (batch_size, seq_len, d_model)
        输出: (batch_size, num_heads, seq_len, depth)
        """
        # (batch_size, seq_len, d_model) -> (batch_size, seq_len, num_heads, depth)
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))

        # (batch_size, seq_len, num_heads, depth) -> (batch_size, num_heads, seq_len, depth)
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, query, key, value, mask=None):
        """
        前向传播

        参数:
            query: (batch_size, seq_len_q, d_model)
            key: (batch_size, seq_len_k, d_model)
            value: (batch_size, seq_len_v, d_model)
            mask: 可选掩码

        返回:
            output: (batch_size, seq_len_q, d_model)
            attention_weights: (batch_size, num_heads, seq_len_q, seq_len_k)
        """
        batch_size = tf.shape(query)[0]

        # 步骤1: 线性投影
        query = self.W_Q(query)  # (batch_size, seq_len_q, d_model)
        key = self.W_K(key)      # (batch_size, seq_len_k, d_model)
        value = self.W_V(value)  # (batch_size, seq_len_v, d_model)

        # 步骤2: 分成多个头
        query = self.split_heads(query, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
        key = self.split_heads(key, batch_size)      # (batch_size, num_heads, seq_len_k, depth)
        value = self.split_heads(value, batch_size)  # (batch_size, num_heads, seq_len_v, depth)

        # 步骤3: 计算缩放点积注意力（每个头独立计算）
        # scaled_dot_product_attention 会自动处理多头情况
        # 因为它在最后两个维度上操作
        scaled_attention, attention_weights = scaled_dot_product_attention(
            query, key, value, mask
        )
        # scaled_attention: (batch_size, num_heads, seq_len_q, depth)
        # attention_weights: (batch_size, num_heads, seq_len_q, seq_len_k)

        # 步骤4: 转置并合并多个头
        # (batch_size, num_heads, seq_len_q, depth) -> (batch_size, seq_len_q, num_heads, depth)
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])

        # (batch_size, seq_len_q, num_heads, depth) -> (batch_size, seq_len_q, d_model)
        concat_attention = tf.reshape(scaled_attention,
                                       (batch_size, -1, self.d_model))

        # 步骤5: 最终线性投影
        output = self.W_O(concat_attention)  # (batch_size, seq_len_q, d_model)

        return output, attention_weights


# ============================================================================
# 第二部分：测试多头注意力
# ============================================================================

def test_multi_head_attention():
    """测试多头注意力层"""
    print("=" * 60)
    print("测试多头注意力层")
    print("=" * 60)

    # 超参数
    d_model = 512
    num_heads = 8
    batch_size = 2
    seq_len = 10

    # 创建多头注意力层
    mha = MultiHeadAttention(d_model, num_heads)

    # 模拟输入（自注意力：Q=K=V=x）
    x = tf.random.normal((batch_size, seq_len, d_model))

    # 前向传播
    output, attention_weights = mha(x, x, x)

    print(f"\n配置:")
    print(f"  d_model = {d_model}")
    print(f"  num_heads = {num_heads}")
    print(f"  每个头的维度 depth = {d_model // num_heads}")

    print(f"\n输入 shape: {x.shape}")
    print(f"输出 shape: {output.shape}")
    print(f"注意力权重 shape: {attention_weights.shape}")

    # 打印参数量
    total_params = sum([tf.reduce_prod(var.shape).numpy()
                       for var in mha.trainable_variables])
    print(f"\n可学习参数数量: {total_params:,}")
    print(f"  - W_Q: {d_model} x {d_model} = {d_model**2:,}")
    print(f"  - W_K: {d_model} x {d_model} = {d_model**2:,}")
    print(f"  - W_V: {d_model} x {d_model} = {d_model**2:,}")
    print(f"  - W_O: {d_model} x {d_model} = {d_model**2:,}")
    print(f"  - 总计: 4 x {d_model}^2 = {4 * d_model**2:,}")


# ============================================================================
# 第三部分：可视化多头注意力
# ============================================================================

def visualize_multi_head_attention():
    """可视化不同头的注意力模式"""
    print("\n" + "=" * 60)
    print("可视化多头注意力")
    print("=" * 60)

    # 简单示例
    d_model = 64
    num_heads = 4
    seq_len = 8

    # 创建并训练一下（随机初始化的权重）
    mha = MultiHeadAttention(d_model, num_heads)

    # 模拟一个句子的嵌入
    x = tf.random.normal((1, seq_len, d_model))

    # 计算注意力
    _, attention_weights = mha(x, x, x)

    # attention_weights: (1, num_heads, seq_len, seq_len)
    weights = attention_weights[0].numpy()  # (num_heads, seq_len, seq_len)

    # 绘制每个头的注意力热图
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    tokens = [f'pos{i}' for i in range(seq_len)]

    for head_idx in range(num_heads):
        ax = axes[head_idx // 2, head_idx % 2]
        im = ax.imshow(weights[head_idx], cmap='Blues', aspect='auto')

        ax.set_xticks(range(seq_len))
        ax.set_yticks(range(seq_len))
        ax.set_xticklabels(tokens, fontsize=8)
        ax.set_yticklabels(tokens, fontsize=8)

        ax.set_xlabel('Key', fontsize=10)
        ax.set_ylabel('Query', fontsize=10)
        ax.set_title(f'Head {head_idx + 1}', fontsize=12)

        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.suptitle('不同注意力头的注意力模式\n(随机初始化的权重，训练后会学到有意义的模式)', fontsize=14)
    plt.tight_layout()
    plt.savefig('multi_head_attention_visualization.png', dpi=150)
    plt.show()
    print("\n多头注意力热图已保存为 multi_head_attention_visualization.png")


# ============================================================================
# 第四部分：使用 Keras 内置的多头注意力
# ============================================================================

def compare_with_keras_mha():
    """与 Keras 内置的 MultiHeadAttention 对比"""
    print("\n" + "=" * 60)
    print("对比 Keras 内置的 MultiHeadAttention")
    print("=" * 60)

    d_model = 64
    num_heads = 4
    seq_len = 10

    # 我们的实现
    our_mha = MultiHeadAttention(d_model, num_heads)

    # Keras 内置实现
    keras_mha = tf.keras.layers.MultiHeadAttention(
        num_heads=num_heads,
        key_dim=d_model // num_heads,  # 每个头的维度
        value_dim=d_model // num_heads
    )

    # 测试输入
    x = tf.random.normal((2, seq_len, d_model))

    # 我们的输出
    our_output, our_weights = our_mha(x, x, x)

    # Keras 的输出
    keras_output = keras_mha(x, x, return_attention_scores=False)

    print(f"\n我们的实现:")
    print(f"  输出 shape: {our_output.shape}")
    print(f"  注意力权重 shape: {our_weights.shape}")

    print(f"\nKeras 内置实现:")
    print(f"  输出 shape: {keras_output.shape}")

    print("\n结论: 两种实现的输出形状完全一致！")
    print("我们的实现额外返回了注意力权重，便于可视化和分析。")


# ============================================================================
# 第五部分：多头注意力的优势解释
# ============================================================================

def explain_multi_head_benefit():
    """解释多头注意力的优势"""
    print("\n" + "=" * 60)
    print("多头注意力的优势")
    print("=" * 60)

    explanation = """
【计算效率】
假设 d_model = 512, num_heads = 8:

单头注意力:
  - Q, K, V 都是 512 维
  - 注意力计算: O(seq_len^2 * 512)

多头注意力:
  - 每个头: Q, K, V 都是 64 维
  - 8 个头并行计算: 8 * O(seq_len^2 * 64) = O(seq_len^2 * 512)
  - 计算量相同，但可以并行！

【表达能力】
多头注意力可以学习多种不同的注意力模式：

Head 1: 可能学习关注"主语-谓语"关系
  "猫 吃 鱼" -> "猫"强烈关注"吃"

Head 2: 可能学习关注"谓语-宾语"关系
  "猫 吃 鱼" -> "吃"强烈关注"鱼"

Head 3: 可能学习关注位置相近的词
  每个词主要关注邻近的词

Head 4: 可能学习关注语义相似的词
  同义词、近义词之间有高注意力

【类比】
单头注意力: 一个人用一种视角看问题
多头注意力: 一个团队从多个角度分析问题，然后综合意见
"""
    print(explanation)


# ============================================================================
# 主函数
# ============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Transformer 学习 - 02. 多头注意力")
    print("=" * 60)

    # 1. 测试多头注意力
    test_multi_head_attention()

    # 2. 可视化
    visualize_multi_head_attention()

    # 3. 与 Keras 对比
    compare_with_keras_mha()

    # 4. 解释优势
    explain_multi_head_benefit()

    print("\n" + "=" * 60)
    print("下一步: 03_positional_encoding.py - 位置编码")
    print("=" * 60)
