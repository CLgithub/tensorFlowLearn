#coding=utf-8
"""
=============================================================================
04. Transformer 块 (Transformer Block)
=============================================================================

【Transformer Block 结构】

一个完整的 Transformer Block 包含:

1. Multi-Head Attention (多头注意力)
2. Add & Norm (残差连接 + 层归一化)
3. Feed Forward Network (前馈神经网络)
4. Add & Norm (残差连接 + 层归一化)

图示:
    ┌─────────────────┐
    │      输入 x     │
    └────────┬────────┘
             │
    ┌────────▼────────┐
    │ Multi-Head Attn │
    └────────┬────────┘
             │
    ┌────────▼────────┐
    │   Add & Norm    │←── 残差连接: x + Attention(x)
    └────────┬────────┘
             │
    ┌────────▼────────┐
    │  Feed Forward   │
    └────────┬────────┘
             │
    ┌────────▼────────┐
    │   Add & Norm    │←── 残差连接
    └────────┬────────┘
             │
    ┌────────▼────────┐
    │      输出       │
    └─────────────────┘

【关键组件解释】

1. 残差连接 (Residual Connection)
   - output = x + F(x)
   - 让梯度更容易回传，解决深层网络训练困难的问题

2. 层归一化 (Layer Normalization)
   - 对每个样本的特征维度进行归一化
   - 稳定训练，加速收敛

3. 前馈网络 (Feed Forward Network)
   - 两层全连接: Linear -> ReLU -> Linear
   - 中间维度通常是 4 * d_model

=============================================================================
"""

import tensorflow as tf
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# 导入之前实现的组件
import importlib.util
import os

current_dir = os.path.dirname(os.path.abspath(__file__))

# 导入多头注意力
spec = importlib.util.spec_from_file_location(
    "multi_head_attention",
    os.path.join(current_dir, "02_multi_head_attention.py")
)
mha_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mha_module)
MultiHeadAttention = mha_module.MultiHeadAttention

# 导入位置编码
spec = importlib.util.spec_from_file_location(
    "positional_encoding",
    os.path.join(current_dir, "03_positional_encoding.py")
)
pe_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pe_module)
PositionalEncoding = pe_module.PositionalEncoding


# ============================================================================
# 第一部分：前馈神经网络
# ============================================================================

class FeedForward(tf.keras.layers.Layer):
    """
    前馈神经网络

    结构: Linear(d_model -> d_ff) -> ReLU -> Dropout -> Linear(d_ff -> d_model)

    这个网络对每个位置独立应用，不涉及位置之间的交互。
    位置之间的交互由注意力机制完成。
    """

    def __init__(self, d_model, d_ff, dropout_rate=0.1):
        """
        参数:
            d_model: 模型维度（输入和输出维度）
            d_ff: 前馈网络中间层维度，通常是 4 * d_model
            dropout_rate: Dropout 比率
        """
        super().__init__()

        self.dense1 = tf.keras.layers.Dense(d_ff, activation='relu', name='ff_dense1')
        self.dense2 = tf.keras.layers.Dense(d_model, name='ff_dense2')
        self.dropout = tf.keras.layers.Dropout(dropout_rate)

    def call(self, x, training=False):
        """
        参数:
            x: 输入, shape = (batch_size, seq_len, d_model)
            training: 是否训练模式（影响 dropout）

        返回:
            输出, shape = (batch_size, seq_len, d_model)
        """
        x = self.dense1(x)       # (batch_size, seq_len, d_ff)
        x = self.dropout(x, training=training)
        x = self.dense2(x)       # (batch_size, seq_len, d_model)
        return x


# ============================================================================
# 第二部分：Transformer Encoder Block
# ============================================================================

class TransformerEncoderBlock(tf.keras.layers.Layer):
    """
    Transformer Encoder 块

    用于编码器，处理完整序列，所有位置可以相互看到。
    适用于：BERT、文本分类、情感分析等。
    """

    def __init__(self, d_model, num_heads, d_ff, dropout_rate=0.1):
        """
        参数:
            d_model: 模型维度
            num_heads: 注意力头数
            d_ff: 前馈网络中间层维度
            dropout_rate: Dropout 比率
        """
        super().__init__()

        # 多头注意力
        self.mha = MultiHeadAttention(d_model, num_heads)
        # 前馈网络
        self.ffn = FeedForward(d_model, d_ff, dropout_rate)

        # 层归一化
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        # Dropout
        self.dropout1 = tf.keras.layers.Dropout(dropout_rate)
        self.dropout2 = tf.keras.layers.Dropout(dropout_rate)

    def call(self, x, training=False, mask=None):
        """
        参数:
            x: 输入, shape = (batch_size, seq_len, d_model)
            training: 是否训练模式
            mask: 注意力掩码

        返回:
            输出, shape = (batch_size, seq_len, d_model)
        """
        # 多头自注意力
        attn_output, attention_weights = self.mha(x, x, x, mask)
        attn_output = self.dropout1(attn_output, training=training)

        # 残差连接 + 层归一化
        out1 = self.layernorm1(x + attn_output)

        # 前馈网络
        ffn_output = self.ffn(out1, training=training)
        ffn_output = self.dropout2(ffn_output, training=training)

        # 残差连接 + 层归一化
        out2 = self.layernorm2(out1 + ffn_output)

        return out2, attention_weights


# ============================================================================
# 第三部分：Transformer Decoder Block
# ============================================================================

class TransformerDecoderBlock(tf.keras.layers.Layer):
    """
    Transformer Decoder 块

    用于解码器，有两个注意力层：
    1. Masked Self-Attention: 只能看到之前的位置
    2. Cross-Attention: 关注编码器的输出

    适用于：机器翻译、序列到序列任务
    """

    def __init__(self, d_model, num_heads, d_ff, dropout_rate=0.1):
        super().__init__()

        # 掩码自注意力（只看之前的位置）
        self.masked_mha = MultiHeadAttention(d_model, num_heads)
        # 交叉注意力（关注编码器输出）
        self.cross_mha = MultiHeadAttention(d_model, num_heads)
        # 前馈网络
        self.ffn = FeedForward(d_model, d_ff, dropout_rate)

        # 层归一化
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        # Dropout
        self.dropout1 = tf.keras.layers.Dropout(dropout_rate)
        self.dropout2 = tf.keras.layers.Dropout(dropout_rate)
        self.dropout3 = tf.keras.layers.Dropout(dropout_rate)

    def call(self, x, encoder_output, training=False,
             look_ahead_mask=None, padding_mask=None):
        """
        参数:
            x: 解码器输入, shape = (batch_size, target_seq_len, d_model)
            encoder_output: 编码器输出, shape = (batch_size, input_seq_len, d_model)
            look_ahead_mask: 因果掩码，防止看到未来
            padding_mask: padding 掩码

        返回:
            输出和注意力权重
        """
        # 掩码自注意力
        attn1, attn_weights_1 = self.masked_mha(x, x, x, look_ahead_mask)
        attn1 = self.dropout1(attn1, training=training)
        out1 = self.layernorm1(x + attn1)

        # 交叉注意力（Q 来自解码器，K、V 来自编码器）
        attn2, attn_weights_2 = self.cross_mha(out1, encoder_output, encoder_output, padding_mask)
        attn2 = self.dropout2(attn2, training=training)
        out2 = self.layernorm2(out1 + attn2)

        # 前馈网络
        ffn_output = self.ffn(out2, training=training)
        ffn_output = self.dropout3(ffn_output, training=training)
        out3 = self.layernorm3(out2 + ffn_output)

        return out3, attn_weights_1, attn_weights_2


# ============================================================================
# 第四部分：测试 Transformer Block
# ============================================================================

def test_encoder_block():
    """测试 Encoder Block"""
    print("=" * 60)
    print("测试 Transformer Encoder Block")
    print("=" * 60)

    # 超参数
    d_model = 512
    num_heads = 8
    d_ff = 2048
    dropout_rate = 0.1

    batch_size = 2
    seq_len = 10

    # 创建 Encoder Block
    encoder_block = TransformerEncoderBlock(d_model, num_heads, d_ff, dropout_rate)

    # 模拟输入
    x = tf.random.normal((batch_size, seq_len, d_model))

    # 前向传播
    output, attention_weights = encoder_block(x, training=False)

    print(f"\n配置:")
    print(f"  d_model = {d_model}")
    print(f"  num_heads = {num_heads}")
    print(f"  d_ff = {d_ff}")

    print(f"\n输入 shape: {x.shape}")
    print(f"输出 shape: {output.shape}")
    print(f"注意力权重 shape: {attention_weights.shape}")

    # 统计参数
    total_params = sum([tf.reduce_prod(var.shape).numpy()
                       for var in encoder_block.trainable_variables])
    print(f"\n总参数数量: {total_params:,}")


def test_decoder_block():
    """测试 Decoder Block"""
    print("\n" + "=" * 60)
    print("测试 Transformer Decoder Block")
    print("=" * 60)

    # 超参数
    d_model = 512
    num_heads = 8
    d_ff = 2048
    dropout_rate = 0.1

    batch_size = 2
    input_seq_len = 10   # 编码器输入长度
    target_seq_len = 15  # 解码器输入长度

    # 创建 Decoder Block
    decoder_block = TransformerDecoderBlock(d_model, num_heads, d_ff, dropout_rate)

    # 模拟输入
    x = tf.random.normal((batch_size, target_seq_len, d_model))
    encoder_output = tf.random.normal((batch_size, input_seq_len, d_model))

    # 前向传播
    output, attn1, attn2 = decoder_block(x, encoder_output, training=False)

    print(f"\n配置:")
    print(f"  d_model = {d_model}")
    print(f"  num_heads = {num_heads}")
    print(f"  d_ff = {d_ff}")

    print(f"\n解码器输入 shape: {x.shape}")
    print(f"编码器输出 shape: {encoder_output.shape}")
    print(f"解码器输出 shape: {output.shape}")
    print(f"自注意力权重 shape: {attn1.shape}")
    print(f"交叉注意力权重 shape: {attn2.shape}")

    # 统计参数
    total_params = sum([tf.reduce_prod(var.shape).numpy()
                       for var in decoder_block.trainable_variables])
    print(f"\n总参数数量: {total_params:,}")


# ============================================================================
# 第五部分：残差连接和层归一化的重要性
# ============================================================================

def explain_residual_and_layernorm():
    """解释残差连接和层归一化"""
    print("\n" + "=" * 60)
    print("残差连接和层归一化的重要性")
    print("=" * 60)

    explanation = """
【残差连接 (Residual Connection)】

公式: output = x + F(x)

为什么重要？
1. 梯度直通：梯度可以直接通过恒等映射回传
   ∂output/∂x = 1 + ∂F(x)/∂x
   即使 ∂F(x)/∂x 很小，梯度也不会消失

2. 特征复用：底层特征可以直接传到高层

3. 训练稳定：网络可以学习"做什么改变"而不是"输出什么"

类比：
不用残差：告诉画家"请画一幅完整的画"
用残差：告诉画家"请在这幅画上修改一些地方"


【层归一化 (Layer Normalization)】

公式: LayerNorm(x) = γ * (x - μ) / σ + β

其中 μ, σ 是在特征维度上计算的均值和标准差
γ, β 是可学习的缩放和偏移参数

为什么用层归一化而不是批归一化？
1. 批归一化需要足够大的 batch size
2. 层归一化对每个样本独立处理，适合变长序列
3. 推理时不需要依赖训练时的统计量

作用：
1. 稳定激活值的分布
2. 减少内部协变量偏移
3. 允许更大的学习率，加速训练
"""
    print(explanation)


# ============================================================================
# 第六部分：Pre-Norm vs Post-Norm
# ============================================================================

class PreNormTransformerBlock(tf.keras.layers.Layer):
    """
    Pre-Norm Transformer Block

    结构变化：先归一化，再做变换
    x -> LayerNorm -> Attention -> + -> LayerNorm -> FFN -> +

    优点：训练更稳定，尤其是深层网络
    GPT-2, GPT-3 等使用 Pre-Norm
    """

    def __init__(self, d_model, num_heads, d_ff, dropout_rate=0.1):
        super().__init__()

        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = FeedForward(d_model, d_ff, dropout_rate)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(dropout_rate)
        self.dropout2 = tf.keras.layers.Dropout(dropout_rate)

    def call(self, x, training=False, mask=None):
        # Pre-Norm: 先归一化
        norm_x = self.layernorm1(x)
        attn_output, attention_weights = self.mha(norm_x, norm_x, norm_x, mask)
        attn_output = self.dropout1(attn_output, training=training)
        x = x + attn_output  # 残差连接

        # Pre-Norm FFN
        norm_x = self.layernorm2(x)
        ffn_output = self.ffn(norm_x, training=training)
        ffn_output = self.dropout2(ffn_output, training=training)
        x = x + ffn_output  # 残差连接

        return x, attention_weights


def compare_prenorm_postnorm():
    """比较 Pre-Norm 和 Post-Norm"""
    print("\n" + "=" * 60)
    print("Pre-Norm vs Post-Norm")
    print("=" * 60)

    comparison = """
┌─────────────────────┬─────────────────────────┬─────────────────────────┐
│       特性          │      Post-Norm          │      Pre-Norm           │
├─────────────────────┼─────────────────────────┼─────────────────────────┤
│ 结构                │ Attn -> Add -> Norm     │ Norm -> Attn -> Add     │
├─────────────────────┼─────────────────────────┼─────────────────────────┤
│ 训练稳定性          │ 较差，需要 warmup       │ 较好，可省略 warmup     │
├─────────────────────┼─────────────────────────┼─────────────────────────┤
│ 最终性能            │ 略好（充分训练后）      │ 略差                    │
├─────────────────────┼─────────────────────────┼─────────────────────────┤
│ 适合场景            │ 中等深度网络            │ 非常深的网络            │
├─────────────────────┼─────────────────────────┼─────────────────────────┤
│ 使用者              │ 原始 Transformer, BERT  │ GPT-2, GPT-3            │
└─────────────────────┴─────────────────────────┴─────────────────────────┘
"""
    print(comparison)


# ============================================================================
# 主函数
# ============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Transformer 学习 - 04. Transformer Block")
    print("=" * 60)

    # 1. 测试 Encoder Block
    test_encoder_block()

    # 2. 测试 Decoder Block
    test_decoder_block()

    # 3. 解释残差和层归一化
    explain_residual_and_layernorm()

    # 4. 比较 Pre-Norm 和 Post-Norm
    compare_prenorm_postnorm()

    print("\n" + "=" * 60)
    print("下一步: 05_decoder_only.py - Decoder-Only 架构 (GPT 风格)")
    print("=" * 60)
