#coding=utf-8
"""
=============================================================================
05. Decoder-Only 架构 (GPT 风格文本生成)
=============================================================================

【Decoder-Only 架构】

这是 GPT 系列（GPT-1, GPT-2, GPT-3, ChatGPT）使用的架构。
与完整的 Encoder-Decoder 不同，它只有解码器部分。

特点：
1. 使用因果掩码 (Causal Mask)，每个位置只能看到之前的位置
2. 自回归生成：逐个 token 生成，每次把之前的输出作为输入

生成过程示例：
  输入: "今天天气"
  步骤1: 模型输入 "今天天气" -> 预测下一个字 "很"
  步骤2: 模型输入 "今天天气很" -> 预测下一个字 "好"
  步骤3: 模型输入 "今天天气很好" -> 预测下一个字 "。"
  ...

=============================================================================
"""

import tensorflow as tf
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# 导入之前的组件
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

# 导入自注意力函数
spec = importlib.util.spec_from_file_location(
    "self_attention",
    os.path.join(current_dir, "01_self_attention.py")
)
sa_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sa_module)
create_causal_mask = sa_module.create_causal_mask


# ============================================================================
# 第一部分：Decoder Block (带因果掩码)
# ============================================================================

class DecoderBlock(tf.keras.layers.Layer):
    """
    GPT 风格的 Decoder Block

    使用 Pre-Norm 结构（更稳定）：
    x -> LayerNorm -> Masked Attention -> + -> LayerNorm -> FFN -> +
    """

    def __init__(self, d_model, num_heads, d_ff, dropout_rate=0.1):
        super().__init__()

        self.d_model = d_model

        # 带掩码的多头自注意力
        self.mha = MultiHeadAttention(d_model, num_heads)

        # 前馈网络
        self.ffn_dense1 = tf.keras.layers.Dense(d_ff, activation='gelu', name='ffn1')
        self.ffn_dense2 = tf.keras.layers.Dense(d_model, name='ffn2')

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
            mask: 因果掩码

        返回:
            输出, shape = (batch_size, seq_len, d_model)
        """
        # Pre-Norm 自注意力
        norm_x = self.layernorm1(x)
        attn_output, attention_weights = self.mha(norm_x, norm_x, norm_x, mask)
        attn_output = self.dropout1(attn_output, training=training)
        x = x + attn_output

        # Pre-Norm FFN
        norm_x = self.layernorm2(x)
        ffn_output = self.ffn_dense1(norm_x)
        ffn_output = self.ffn_dense2(ffn_output)
        ffn_output = self.dropout2(ffn_output, training=training)
        x = x + ffn_output

        return x, attention_weights


# ============================================================================
# 第二部分：完整的 GPT 模型
# ============================================================================

class GPTModel(tf.keras.Model):
    """
    简化版 GPT 模型

    结构:
    1. Token Embedding
    2. Position Embedding
    3. N 层 Decoder Block
    4. 最终的 LayerNorm
    5. 输出层（预测下一个 token 的概率分布）
    """

    def __init__(self, vocab_size, max_seq_len, d_model, num_heads, d_ff,
                 num_layers, dropout_rate=0.1):
        """
        参数:
            vocab_size: 词表大小
            max_seq_len: 最大序列长度
            d_model: 模型维度
            num_heads: 注意力头数
            d_ff: 前馈网络中间层维度
            num_layers: Decoder Block 层数
            dropout_rate: Dropout 比率
        """
        super().__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        # Token 嵌入
        self.token_embedding = tf.keras.layers.Embedding(
            vocab_size, d_model, name='token_embedding'
        )

        # 位置嵌入（可学习）
        self.position_embedding = tf.keras.layers.Embedding(
            max_seq_len, d_model, name='position_embedding'
        )

        # N 层 Decoder Block
        self.decoder_blocks = [
            DecoderBlock(d_model, num_heads, d_ff, dropout_rate)
            for _ in range(num_layers)
        ]

        # 最终层归一化
        self.final_layernorm = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        # 输出层（与 token embedding 共享权重，节省参数）
        # 这里简化处理，使用独立的 Dense 层
        self.output_layer = tf.keras.layers.Dense(vocab_size, name='output')

        # Dropout
        self.dropout = tf.keras.layers.Dropout(dropout_rate)

    def call(self, x, training=False):
        """
        参数:
            x: 输入 token IDs, shape = (batch_size, seq_len)
            training: 是否训练模式

        返回:
            logits: 每个位置的词表概率分布, shape = (batch_size, seq_len, vocab_size)
        """
        seq_len = tf.shape(x)[1]

        # 创建因果掩码
        causal_mask = create_causal_mask(seq_len)

        # Token 嵌入
        x = self.token_embedding(x)  # (batch_size, seq_len, d_model)

        # 缩放嵌入
        x = x * tf.math.sqrt(tf.cast(self.d_model, tf.float32))

        # 位置嵌入
        positions = tf.range(seq_len)
        pos_emb = self.position_embedding(positions)  # (seq_len, d_model)
        x = x + pos_emb

        # Dropout
        x = self.dropout(x, training=training)

        # 通过 N 层 Decoder Block
        attention_weights_list = []
        for decoder_block in self.decoder_blocks:
            x, attention_weights = decoder_block(x, training=training, mask=causal_mask)
            attention_weights_list.append(attention_weights)

        # 最终层归一化
        x = self.final_layernorm(x)

        # 输出层
        logits = self.output_layer(x)  # (batch_size, seq_len, vocab_size)

        return logits

    def generate(self, input_ids, max_new_tokens, temperature=1.0, top_k=None):
        """
        自回归生成文本

        参数:
            input_ids: 初始 token IDs, shape = (1, initial_len)
            max_new_tokens: 要生成的新 token 数量
            temperature: 温度参数，控制随机性
                - temperature < 1: 更确定性（选择概率高的）
                - temperature > 1: 更随机（更有创意）
            top_k: 只从概率最高的 k 个 token 中采样

        返回:
            生成的 token IDs
        """
        for _ in range(max_new_tokens):
            # 获取当前序列的 logits
            logits = self(input_ids, training=False)

            # 只取最后一个位置的 logits
            next_logits = logits[:, -1, :]  # (batch_size, vocab_size)

            # 应用温度
            next_logits = next_logits / temperature

            # Top-K 采样
            if top_k is not None:
                # 找到 top-k 的阈值
                top_k_values, _ = tf.math.top_k(next_logits, k=top_k)
                threshold = top_k_values[:, -1:]  # 第 k 大的值
                # 将小于阈值的设为极小值
                next_logits = tf.where(
                    next_logits < threshold,
                    tf.ones_like(next_logits) * -1e9,
                    next_logits
                )

            # Softmax 得到概率
            probs = tf.nn.softmax(next_logits, axis=-1)

            # 采样下一个 token
            next_token = tf.random.categorical(tf.math.log(probs), num_samples=1)
            next_token = tf.cast(next_token, tf.int32)

            # 拼接到序列后面
            input_ids = tf.concat([input_ids, next_token], axis=1)

        return input_ids


# ============================================================================
# 第三部分：测试 GPT 模型
# ============================================================================

def test_gpt_model():
    """测试 GPT 模型"""
    print("=" * 60)
    print("测试 GPT 模型")
    print("=" * 60)

    # 超参数（小模型用于测试）
    vocab_size = 1000
    max_seq_len = 128
    d_model = 256
    num_heads = 4
    d_ff = 1024
    num_layers = 4
    dropout_rate = 0.1

    # 创建模型
    model = GPTModel(
        vocab_size=vocab_size,
        max_seq_len=max_seq_len,
        d_model=d_model,
        num_heads=num_heads,
        d_ff=d_ff,
        num_layers=num_layers,
        dropout_rate=dropout_rate
    )

    # 测试前向传播
    batch_size = 2
    seq_len = 20
    input_ids = tf.random.uniform((batch_size, seq_len), maxval=vocab_size, dtype=tf.int32)

    logits = model(input_ids, training=False)

    print(f"\n模型配置:")
    print(f"  vocab_size = {vocab_size}")
    print(f"  max_seq_len = {max_seq_len}")
    print(f"  d_model = {d_model}")
    print(f"  num_heads = {num_heads}")
    print(f"  d_ff = {d_ff}")
    print(f"  num_layers = {num_layers}")

    print(f"\n输入 shape: {input_ids.shape}")
    print(f"输出 logits shape: {logits.shape}")

    # 统计参数
    total_params = sum([tf.reduce_prod(var.shape).numpy()
                       for var in model.trainable_variables])
    print(f"\n总参数数量: {total_params:,}")

    # 测试生成
    print("\n测试文本生成:")
    initial_ids = tf.constant([[1, 2, 3]], dtype=tf.int32)  # 模拟初始输入
    generated = model.generate(initial_ids, max_new_tokens=10, temperature=0.8, top_k=50)
    print(f"  初始序列: {initial_ids[0].numpy()}")
    print(f"  生成序列: {generated[0].numpy()}")


# ============================================================================
# 第四部分：训练示例
# ============================================================================

def create_training_example():
    """创建一个简单的训练示例"""
    print("\n" + "=" * 60)
    print("训练示例：下一个 token 预测")
    print("=" * 60)

    # 模拟一个简单的训练过程
    vocab_size = 100
    max_seq_len = 32
    d_model = 64
    num_heads = 2
    d_ff = 256
    num_layers = 2

    # 创建小模型
    model = GPTModel(vocab_size, max_seq_len, d_model, num_heads, d_ff, num_layers)

    # 损失函数：交叉熵
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    # 优化器
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

    # 模拟训练数据
    # 输入: [1, 2, 3, 4, 5]
    # 目标: [2, 3, 4, 5, 6]  (下一个 token)
    input_ids = tf.constant([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]], dtype=tf.int32)
    # 目标是输入右移一位
    target_ids = tf.constant([[2, 3, 4, 5, 6, 7, 8, 9, 10, 11]], dtype=tf.int32)

    print("\n训练数据:")
    print(f"  输入: {input_ids[0].numpy()}")
    print(f"  目标: {target_ids[0].numpy()}")

    # 训练几步
    print("\n开始训练:")
    for step in range(100):
        with tf.GradientTape() as tape:
            logits = model(input_ids, training=True)
            loss = loss_fn(target_ids, logits)

        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        if step % 20 == 0:
            # 预测
            predictions = tf.argmax(logits, axis=-1)
            accuracy = tf.reduce_mean(
                tf.cast(predictions == target_ids, tf.float32)
            )
            print(f"  Step {step}: loss = {loss.numpy():.4f}, accuracy = {accuracy.numpy():.4f}")

    # 最终预测
    final_logits = model(input_ids, training=False)
    final_predictions = tf.argmax(final_logits, axis=-1)
    print(f"\n最终预测: {final_predictions[0].numpy()}")
    print(f"目标:      {target_ids[0].numpy()}")


# ============================================================================
# 第五部分：采样策略解释
# ============================================================================

def explain_sampling_strategies():
    """解释不同的采样策略"""
    print("\n" + "=" * 60)
    print("文本生成的采样策略")
    print("=" * 60)

    explanation = """
【温度 (Temperature)】

logits / temperature -> softmax -> 采样

temperature = 0.1 (低温):
  概率分布更"尖锐"，高概率的 token 更容易被选中
  生成更确定、更"安全"的文本
  例如: [0.01, 0.98, 0.01] -> 几乎总是选择第二个

temperature = 1.0 (标准):
  正常的概率分布

temperature = 2.0 (高温):
  概率分布更"平坦"，各 token 被选中的概率更接近
  生成更随机、更"有创意"的文本
  例如: [0.25, 0.50, 0.25] -> 有更多可能性


【Top-K 采样】

只从概率最高的 K 个 token 中采样

top_k = 1:
  贪婪解码，总是选概率最高的
  确定性最强，但可能单调无聊

top_k = 10:
  从最可能的 10 个 token 中随机选择
  平衡确定性和多样性

top_k = 50:
  更多多样性


【Top-P 采样 (Nucleus Sampling)】

选择累积概率达到 P 的最小 token 集合，然后从中采样

top_p = 0.9:
  选择概率最高的若干 token，使它们的概率和 >= 0.9
  动态调整候选数量

优点: 比 top_k 更灵活
  - 如果模型很确定，候选少
  - 如果模型不确定，候选多


【组合使用】

实践中通常组合使用:
  temperature=0.7 + top_p=0.9
  或
  temperature=0.8 + top_k=50

这样既保证质量，又有一定多样性。
"""
    print(explanation)


# ============================================================================
# 第六部分：与 GPT-2 架构对比
# ============================================================================

def compare_with_gpt2():
    """与 GPT-2 架构对比"""
    print("\n" + "=" * 60)
    print("我们的实现 vs GPT-2 官方架构")
    print("=" * 60)

    comparison = """
┌─────────────────────┬──────────────────┬──────────────────┐
│       特性          │    我们的实现    │     GPT-2        │
├─────────────────────┼──────────────────┼──────────────────┤
│ 架构类型            │ Decoder-Only     │ Decoder-Only     │
├─────────────────────┼──────────────────┼──────────────────┤
│ Norm 位置           │ Pre-Norm         │ Pre-Norm         │
├─────────────────────┼──────────────────┼──────────────────┤
│ 激活函数            │ GELU             │ GELU             │
├─────────────────────┼──────────────────┼──────────────────┤
│ 位置编码            │ 可学习           │ 可学习           │
├─────────────────────┼──────────────────┼──────────────────┤
│ 权重共享            │ 无               │ 输出层共享嵌入   │
├─────────────────────┼──────────────────┼──────────────────┤
│ 初始化              │ 默认             │ 特殊初始化       │
└─────────────────────┴──────────────────┴──────────────────┘

GPT-2 模型规模:

┌─────────────┬─────────┬──────────┬─────────┬──────────┬─────────────┐
│   模型      │ d_model │ num_heads│  d_ff   │ num_layers│   参数量    │
├─────────────┼─────────┼──────────┼─────────┼──────────┼─────────────┤
│ GPT-2 Small │   768   │    12    │  3072   │    12    │   117M      │
│ GPT-2 Medium│  1024   │    16    │  4096   │    24    │   345M      │
│ GPT-2 Large │  1280   │    20    │  5120   │    36    │   774M      │
│ GPT-2 XL    │  1600   │    25    │  6400   │    48    │   1.5B      │
└─────────────┴─────────┴──────────┴─────────┴──────────┴─────────────┘
"""
    print(comparison)


# ============================================================================
# 主函数
# ============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Transformer 学习 - 05. Decoder-Only 架构 (GPT)")
    print("=" * 60)

    # 1. 测试 GPT 模型
    test_gpt_model()

    # 2. 训练示例
    create_training_example()

    # 3. 采样策略解释
    explain_sampling_strategies()

    # 4. 与 GPT-2 对比
    compare_with_gpt2()

    print("\n" + "=" * 60)
    print("阶段一完成！")
    print("你已经从零实现了 Transformer 的核心组件：")
    print("  1. 自注意力机制")
    print("  2. 多头注意力")
    print("  3. 位置编码")
    print("  4. Transformer Block")
    print("  5. Decoder-Only 架构 (GPT)")
    print("")
    print("下一步: 06_text_generator.py - 完整的文本生成模型")
    print("=" * 60)
