#coding=utf-8
"""
=============================================================================
08. 对话模型 (Chat Model)
=============================================================================

基于 Transformer Decoder 的简单对话模型
- 可以用自定义对话数据训练
- 支持交互式对话测试

架构: GPT 风格的 Decoder-Only Transformer
"""

import tensorflow as tf
import numpy as np
import json
import os
import re

# ============================================================================
# 第一部分：简单的分词器
# ============================================================================

class SimpleTokenizer:
    """简单的字符级分词器"""

    def __init__(self):
        self.char_to_id = {}
        self.id_to_char = {}
        self.vocab_size = 0

        # 特殊 token
        self.pad_token = '<PAD>'
        self.sos_token = '<SOS>'  # Start of sentence
        self.eos_token = '<EOS>'  # End of sentence
        self.unk_token = '<UNK>'  # Unknown

    def build_vocab(self, texts):
        """从文本构建词表"""
        # 添加特殊 token
        special_tokens = [self.pad_token, self.sos_token, self.eos_token, self.unk_token]
        for i, token in enumerate(special_tokens):
            self.char_to_id[token] = i
            self.id_to_char[i] = token

        # 收集所有字符
        all_chars = set()
        for text in texts:
            all_chars.update(text)

        # 添加到词表
        for char in sorted(all_chars):
            if char not in self.char_to_id:
                idx = len(self.char_to_id)
                self.char_to_id[char] = idx
                self.id_to_char[idx] = char

        self.vocab_size = len(self.char_to_id)
        print(f"词表大小: {self.vocab_size}")

    def encode(self, text, add_special_tokens=True):
        """文本转 ID"""
        ids = []
        if add_special_tokens:
            ids.append(self.char_to_id[self.sos_token])

        for char in text:
            if char in self.char_to_id:
                ids.append(self.char_to_id[char])
            else:
                ids.append(self.char_to_id[self.unk_token])

        if add_special_tokens:
            ids.append(self.char_to_id[self.eos_token])

        return ids

    def decode(self, ids, skip_special_tokens=True):
        """ID 转文本"""
        chars = []
        special_ids = {
            self.char_to_id[self.pad_token],
            self.char_to_id[self.sos_token],
            self.char_to_id[self.eos_token],
        }

        for id in ids:
            if skip_special_tokens and id in special_ids:
                continue
            if id in self.id_to_char:
                chars.append(self.id_to_char[id])

        return ''.join(chars)

    def save(self, path):
        """保存词表"""
        with open(path, 'w', encoding='utf-8') as f:
            json.dump({
                'char_to_id': self.char_to_id,
                'id_to_char': {int(k): v for k, v in self.id_to_char.items()}
            }, f, ensure_ascii=False, indent=2)

    def load(self, path):
        """加载词表"""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            self.char_to_id = data['char_to_id']
            self.id_to_char = {int(k): v for k, v in data['id_to_char'].items()}
            self.vocab_size = len(self.char_to_id)


# ============================================================================
# 第二部分：Transformer 对话模型
# ============================================================================

class TransformerBlock(tf.keras.layers.Layer):
    """Transformer Decoder Block"""

    def __init__(self, d_model, num_heads, d_ff, dropout_rate=0.1):
        super().__init__()

        self.mha = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=d_model // num_heads,
            dropout=dropout_rate
        )

        self.ffn = tf.keras.Sequential([
            tf.keras.layers.Dense(d_ff, activation='gelu'),
            tf.keras.layers.Dropout(dropout_rate),
            tf.keras.layers.Dense(d_model),
        ])

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(dropout_rate)
        self.dropout2 = tf.keras.layers.Dropout(dropout_rate)

    def call(self, x, training=False, mask=None):
        # Self-Attention with causal mask
        attn_output = self.mha(x, x, attention_mask=mask, training=training)
        attn_output = self.dropout1(attn_output, training=training)
        x = self.layernorm1(x + attn_output)

        # Feed Forward
        ffn_output = self.ffn(x)
        ffn_output = self.dropout2(ffn_output, training=training)
        x = self.layernorm2(x + ffn_output)

        return x


class ChatModel(tf.keras.Model):
    """对话模型"""

    def __init__(self, vocab_size, max_seq_len=128, d_model=256,
                 num_heads=4, d_ff=512, num_layers=4, dropout_rate=0.1):
        super().__init__()

        self.d_model = d_model
        self.max_seq_len = max_seq_len

        # Token Embedding
        self.token_embedding = tf.keras.layers.Embedding(vocab_size, d_model)

        # Position Embedding
        self.position_embedding = tf.keras.layers.Embedding(max_seq_len, d_model)

        # Transformer Blocks
        self.transformer_blocks = [
            TransformerBlock(d_model, num_heads, d_ff, dropout_rate)
            for _ in range(num_layers)
        ]

        # Final LayerNorm
        self.final_norm = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        # Output projection
        self.output_layer = tf.keras.layers.Dense(vocab_size)

        self.dropout = tf.keras.layers.Dropout(dropout_rate)

    def create_causal_mask(self, seq_len):
        """创建因果掩码"""
        # 上三角矩阵，对角线以上为 True（需要被 mask）
        mask = tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0)
        return mask  # (seq_len, seq_len)

    def call(self, x, training=False):
        seq_len = tf.shape(x)[1]

        # Embeddings
        token_emb = self.token_embedding(x)
        token_emb = token_emb * tf.math.sqrt(tf.cast(self.d_model, tf.float32))

        positions = tf.range(seq_len)
        pos_emb = self.position_embedding(positions)

        x = token_emb + pos_emb
        x = self.dropout(x, training=training)

        # Causal mask
        causal_mask = self.create_causal_mask(seq_len)

        # Transformer blocks
        for block in self.transformer_blocks:
            x = block(x, training=training, mask=causal_mask)

        x = self.final_norm(x)
        logits = self.output_layer(x)

        return logits

    def generate(self, input_ids, max_new_tokens=50, temperature=0.8, top_k=40, eos_id=2):
        """生成文本"""
        for _ in range(max_new_tokens):
            # 截断到最大长度
            if input_ids.shape[1] >= self.max_seq_len:
                input_ids = input_ids[:, -self.max_seq_len:]

            # 前向传播
            logits = self(input_ids, training=False)
            next_logits = logits[:, -1, :] / temperature

            # Top-K 采样
            if top_k is not None:
                top_k_values, _ = tf.math.top_k(next_logits, k=top_k)
                threshold = top_k_values[:, -1:]
                next_logits = tf.where(
                    next_logits < threshold,
                    tf.ones_like(next_logits) * -1e9,
                    next_logits
                )

            # 采样
            probs = tf.nn.softmax(next_logits, axis=-1)
            next_token = tf.random.categorical(tf.math.log(probs + 1e-10), num_samples=1)
            next_token = tf.cast(next_token, tf.int32)

            # 检查是否结束
            if next_token[0, 0].numpy() == eos_id:
                break

            input_ids = tf.concat([input_ids, next_token], axis=1)

        return input_ids


# ============================================================================
# 第三部分：对话数据准备
# ============================================================================

def create_sample_dialogues():
    """创建示例对话数据"""
    dialogues = [
        # 问候
        ("你好", "你好！很高兴见到你！"),
        ("你好啊", "你好！有什么可以帮助你的吗？"),
        ("嗨", "嗨！今天怎么样？"),
        ("早上好", "早上好！祝你有美好的一天！"),
        ("晚上好", "晚上好！今天过得怎么样？"),
        ("下午好", "下午好！需要帮忙吗？"),

        # 自我介绍
        ("你是谁", "我是一个对话机器人，可以和你聊天。"),
        ("你叫什么名字", "我是小助手，很高兴认识你！"),
        ("你是什么", "我是基于Transformer的对话模型。"),
        ("介绍一下你自己", "我是一个AI助手，可以回答问题和闲聊。"),

        # 能力
        ("你能做什么", "我可以和你聊天，回答简单的问题。"),
        ("你会什么", "我会聊天、回答问题，还在学习中。"),

        # 日常对话
        ("今天天气怎么样", "我不太清楚天气，但希望是个好天气！"),
        ("吃饭了吗", "我不需要吃饭，但你要按时吃饭哦！"),
        ("在干什么", "我在等你和我聊天呢！"),
        ("无聊", "那我们聊聊天吧，有什么想说的？"),

        # 情感
        ("我很开心", "太好了！开心的心情最重要！"),
        ("我不开心", "别难过，有什么可以帮助你的吗？"),
        ("谢谢", "不客气，很高兴能帮到你！"),
        ("谢谢你", "不用谢，这是我应该做的！"),

        # 告别
        ("再见", "再见！期待下次见面！"),
        ("拜拜", "拜拜！保重！"),
        ("我要走了", "好的，再见！随时欢迎回来！"),
        ("晚安", "晚安！做个好梦！"),

        # 更多对话
        ("你喜欢什么", "我喜欢和人类交流，学习新知识。"),
        ("你多大了", "我刚刚被创建，还是个新生的AI。"),
        ("你在哪里", "我在服务器上运行，随时为你服务。"),
        ("讲个笑话", "为什么程序员总是分不清万圣节和圣诞节？因为Oct31等于Dec25！"),
        ("好笑", "哈哈，很高兴你喜欢！"),
    ]
    return dialogues


def prepare_training_data(dialogues, tokenizer, max_len=64):
    """准备训练数据"""
    inputs = []
    targets = []

    for question, answer in dialogues:
        # 格式: <SOS>问题<EOS><SOS>回答<EOS>
        # 输入: <SOS>问题<EOS><SOS>回答
        # 目标: 问题<EOS><SOS>回答<EOS>

        q_ids = tokenizer.encode(question)
        a_ids = tokenizer.encode(answer)

        # 拼接
        full_seq = q_ids + a_ids[1:]  # 去掉回答的 <SOS>，因为问题的 <EOS> 可以当作分隔

        if len(full_seq) > max_len:
            full_seq = full_seq[:max_len]

        # 输入是去掉最后一个 token
        input_seq = full_seq[:-1]
        # 目标是去掉第一个 token
        target_seq = full_seq[1:]

        # Padding
        pad_len = max_len - 1 - len(input_seq)
        if pad_len > 0:
            input_seq = input_seq + [0] * pad_len
            target_seq = target_seq + [0] * pad_len

        inputs.append(input_seq)
        targets.append(target_seq)

    return np.array(inputs), np.array(targets)


# ============================================================================
# 第四部分：训练和对话
# ============================================================================

class ChatBot:
    """聊天机器人"""

    def __init__(self, model_dir='chat_model'):
        self.model_dir = model_dir
        self.model = None
        self.tokenizer = SimpleTokenizer()
        self.max_len = 64

    def train(self, dialogues=None, epochs=100, batch_size=8):
        """训练模型"""
        if dialogues is None:
            dialogues = create_sample_dialogues()

        print("=" * 60)
        print("训练对话模型")
        print("=" * 60)

        # 构建词表
        all_texts = [q + a for q, a in dialogues]
        self.tokenizer.build_vocab(all_texts)

        # 准备数据
        X, y = prepare_training_data(dialogues, self.tokenizer, self.max_len)
        print(f"训练样本数: {len(X)}")
        print(f"序列长度: {X.shape[1]}")

        # 创建模型
        self.model = ChatModel(
            vocab_size=self.tokenizer.vocab_size,
            max_seq_len=self.max_len,
            d_model=128,
            num_heads=4,
            d_ff=256,
            num_layers=3,
            dropout_rate=0.1
        )

        # 编译
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=['accuracy']
        )

        # 训练
        print(f"\n开始训练 {epochs} 轮...")
        history = self.model.fit(
            X, y,
            batch_size=batch_size,
            epochs=epochs,
            verbose=1
        )

        # 保存
        self.save()

        print(f"\n训练完成！")
        print(f"最终 loss: {history.history['loss'][-1]:.4f}")
        print(f"最终 accuracy: {history.history['accuracy'][-1]:.4f}")

        return history

    def chat(self, user_input, temperature=0.8, top_k=40):
        """对话"""
        if self.model is None:
            print("模型未加载，请先训练或加载模型")
            return None

        # 编码用户输入
        input_ids = self.tokenizer.encode(user_input)

        # 转换为 tensor
        input_tensor = tf.constant([input_ids], dtype=tf.int32)

        # 生成回复
        eos_id = self.tokenizer.char_to_id[self.tokenizer.eos_token]
        output_ids = self.model.generate(
            input_tensor,
            max_new_tokens=50,
            temperature=temperature,
            top_k=top_k,
            eos_id=eos_id
        )

        # 解码
        output_ids = output_ids[0].numpy().tolist()

        # 找到回复部分（输入之后的内容）
        response_ids = output_ids[len(input_ids):]
        response = self.tokenizer.decode(response_ids, skip_special_tokens=True)

        return response

    def save(self):
        """保存模型和词表"""
        os.makedirs(self.model_dir, exist_ok=True)

        # 保存模型权重
        self.model.save_weights(os.path.join(self.model_dir, 'model_weights.weights.h5'))

        # 保存词表
        self.tokenizer.save(os.path.join(self.model_dir, 'tokenizer.json'))

        # 保存配置
        config = {
            'vocab_size': self.tokenizer.vocab_size,
            'max_len': self.max_len,
        }
        with open(os.path.join(self.model_dir, 'config.json'), 'w') as f:
            json.dump(config, f)

        print(f"模型已保存到: {self.model_dir}")

    def load(self):
        """加载模型"""
        # 加载词表
        self.tokenizer.load(os.path.join(self.model_dir, 'tokenizer.json'))

        # 加载配置
        with open(os.path.join(self.model_dir, 'config.json'), 'r') as f:
            config = json.load(f)

        self.max_len = config['max_len']

        # 创建模型
        self.model = ChatModel(
            vocab_size=self.tokenizer.vocab_size,
            max_seq_len=self.max_len,
            d_model=128,
            num_heads=4,
            d_ff=256,
            num_layers=3,
            dropout_rate=0.1
        )

        # 构建模型
        dummy_input = tf.zeros((1, 10), dtype=tf.int32)
        _ = self.model(dummy_input)

        # 加载权重
        self.model.load_weights(os.path.join(self.model_dir, 'model_weights.weights.h5'))

        print(f"模型已从 {self.model_dir} 加载")

    def interactive_chat(self):
        """交互式对话"""
        print("\n" + "=" * 60)
        print("交互式对话 (输入 'quit' 退出)")
        print("=" * 60)

        while True:
            user_input = input("\n你: ").strip()

            if user_input.lower() in ['quit', 'exit', 'q', '退出']:
                print("再见！")
                break

            if not user_input:
                continue

            response = self.chat(user_input)
            print(f"机器人: {response}")


# ============================================================================
# 主程序
# ============================================================================

if __name__ == "__main__":
    import sys

    # 模型保存目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(current_dir, 'chat_model')

    # 创建聊天机器人
    bot = ChatBot(model_dir=model_dir)

    # 检查是否有已训练的模型
    if os.path.exists(os.path.join(model_dir, 'model_weights.weights.h5')):
        print("发现已训练的模型，是否加载？")
        choice = input("输入 'y' 加载，其他键重新训练: ").strip().lower()

        if choice == 'y':
            bot.load()
        else:
            bot.train(epochs=200)
    else:
        # 训练新模型
        bot.train(epochs=200)

    # 测试对话
    print("\n" + "=" * 60)
    print("测试对话")
    print("=" * 60)

    test_inputs = ["你好", "你是谁", "你能做什么", "讲个笑话", "再见"]
    for inp in test_inputs:
        response = bot.chat(inp)
        print(f"你: {inp}")
        print(f"机器人: {response}")
        print()

    # 进入交互模式
    bot.interactive_chat()
