#coding=utf-8
"""
=============================================================================
06. CNN vs Transformer 架构对比 + Vision Transformer 猫狗分类
=============================================================================

本文件包含：
1. CNN 和 Transformer 架构的详细对比
2. 用 Vision Transformer (ViT) 实现猫狗分类
3. 与之前 CNN 实现的对比

=============================================================================
                        架构对比图解
=============================================================================

【CNN 典型架构】(你的 book5.2.5.py)

    输入图像 (150, 150, 3)
            ↓
    ┌─────────────────────────┐
    │  Conv2D(32, 3×3) + ReLU │  ← 卷积层：提取局部特征
    └───────────┬─────────────┘
                ↓
    ┌─────────────────────────┐
    │    MaxPooling(2×2)      │  ← 池化层：降维，扩大感受野
    └───────────┬─────────────┘
                ↓
    ┌─────────────────────────┐
    │  Conv2D(64, 3×3) + ReLU │
    └───────────┬─────────────┘
                ↓
    ┌─────────────────────────┐
    │    MaxPooling(2×2)      │
    └───────────┬─────────────┘
                ↓
          ... 重复 ...
                ↓
    ┌─────────────────────────┐
    │       Flatten           │  ← 展平成一维
    └───────────┬─────────────┘
                ↓
    ┌─────────────────────────┐
    │       Dropout           │  ← 防止过拟合
    └───────────┬─────────────┘
                ↓
    ┌─────────────────────────┐
    │    Dense + Softmax      │  ← 分类输出
    └─────────────────────────┘


【Transformer 典型架构】(GPT 风格)

    输入序列 "我 爱 中 国"
            ↓
    ┌─────────────────────────┐
    │   Token Embedding       │  ← 词嵌入：词 → 向量
    │         +               │
    │   Position Embedding    │  ← 位置编码：注入顺序信息
    └───────────┬─────────────┘
                ↓
    ┌─────────────────────────┐
    │      Dropout            │
    └───────────┬─────────────┘
                ↓
    ╔═════════════════════════╗
    ║   Transformer Block ×N  ║  ← 重复 N 次
    ║  ┌───────────────────┐  ║
    ║  │    LayerNorm      │  ║
    ║  └─────────┬─────────┘  ║
    ║            ↓            ║
    ║  ┌───────────────────┐  ║
    ║  │ Multi-Head Attn   │  ║  ← 注意力：全局信息交互
    ║  └─────────┬─────────┘  ║
    ║            ↓            ║
    ║       + 残差连接        ║  ← x + Attention(x)
    ║            ↓            ║
    ║  ┌───────────────────┐  ║
    ║  │    LayerNorm      │  ║
    ║  └─────────┬─────────┘  ║
    ║            ↓            ║
    ║  ┌───────────────────┐  ║
    ║  │   Feed Forward    │  ║  ← 前馈网络
    ║  └─────────┬─────────┘  ║
    ║            ↓            ║
    ║       + 残差连接        ║
    ╚═══════════╤═════════════╝
                ↓
          ... 重复 N 次 ...
                ↓
    ┌─────────────────────────┐
    │      LayerNorm          │
    └───────────┬─────────────┘
                ↓
    ┌─────────────────────────┐
    │   Dense (输出层)         │
    └─────────────────────────┘


【核心组件对比表】

    ┌──────────────┬─────────────────────┬─────────────────────────┐
    │     CNN      │     Transformer     │         作用            │
    ├──────────────┼─────────────────────┼─────────────────────────┤
    │   Conv2D     │ Multi-Head Attention│  特征提取/信息交互      │
    ├──────────────┼─────────────────────┼─────────────────────────┤
    │  MaxPooling  │      （无）         │  CNN用于降维            │
    ├──────────────┼─────────────────────┼─────────────────────────┤
    │    （无）    │ Position Embedding  │  Transformer需要位置信息│
    ├──────────────┼─────────────────────┼─────────────────────────┤
    │  BatchNorm   │     LayerNorm       │  归一化稳定训练         │
    ├──────────────┼─────────────────────┼─────────────────────────┤
    │    （无）    │     残差连接        │  梯度直通，深层训练     │
    ├──────────────┼─────────────────────┼─────────────────────────┤
    │    Dense     │   Feed Forward      │  非线性变换             │
    ├──────────────┼─────────────────────┼─────────────────────────┤
    │   Dropout    │     Dropout         │  防止过拟合             │
    ├──────────────┼─────────────────────┼─────────────────────────┤
    │   Flatten    │      （无需）       │  CNN需要展平            │
    └──────────────┴─────────────────────┴─────────────────────────┘


【维度变化对比】

    CNN:
      (150,150,3) → (74,74,32) → (37,37,64) → ... → Flatten → (1,)
      空间尺寸逐渐缩小，通道数增加，最后压成一个数

    Transformer:
      (seq_len, d_model) → (seq_len, d_model) → ... → (seq_len, d_model)
      维度始终保持不变！


【公式记忆】

    CNN:        (Conv → ReLU → Pool) × N → Flatten → Dense → Output

    Transformer: Embed + PosEmbed → (Norm → Attn → +残差 → Norm → FFN → +残差) × N → Output

=============================================================================
"""

import tensorflow as tf
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import os

# ============================================================================
# 第一部分：Vision Transformer (ViT) 核心组件
# ============================================================================

@tf.keras.utils.register_keras_serializable()
class PatchEmbedding(tf.keras.layers.Layer):
    """
    图像分块嵌入

    将图像分成固定大小的 patches，然后线性投影到嵌入维度。

    例如：150x150 图像，patch_size=15
    - 分成 (150/15) × (150/15) = 10 × 10 = 100 个 patches
    - 每个 patch 是 15×15×3 = 675 维
    - 投影到 d_model 维（如 256 维）

    这样图像就变成了一个"序列"，可以用 Transformer 处理！
    """

    def __init__(self, image_size, patch_size, d_model):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.d_model = d_model

        # 计算 patch 数量
        self.num_patches = (image_size // patch_size) ** 2

        # 使用卷积来实现分块 + 线性投影（更高效）
        # kernel_size = patch_size, stride = patch_size 实现不重叠分块
        self.projection = tf.keras.layers.Conv2D(
            filters=d_model,
            kernel_size=patch_size,
            strides=patch_size,
            padding='valid',
            name='patch_projection'
        )

    def call(self, images):
        """
        参数:
            images: (batch_size, image_size, image_size, channels)

        返回:
            patches: (batch_size, num_patches, d_model)
        """
        # 卷积分块: (batch, H/P, W/P, d_model)
        x = self.projection(images)

        # 展平空间维度: (batch, num_patches, d_model)
        batch_size = tf.shape(x)[0]
        x = tf.reshape(x, [batch_size, -1, self.d_model])

        return x

    def get_config(self):
        config = super().get_config()
        config.update({
            'image_size': self.image_size,
            'patch_size': self.patch_size,
            'd_model': self.d_model,
        })
        return config


@tf.keras.utils.register_keras_serializable()
class ViTEncoder(tf.keras.layers.Layer):
    """
    Vision Transformer 编码器块

    结构与标准 Transformer Encoder 相同：
    LayerNorm → Multi-Head Attention → 残差 → LayerNorm → FFN → 残差
    """

    def __init__(self, d_model, num_heads, d_ff, dropout_rate=0.1):
        super().__init__()

        # 保存配置
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.dropout_rate = dropout_rate

        self.mha = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=d_model // num_heads,
            dropout=dropout_rate
        )

        self.ffn = tf.keras.Sequential([
            tf.keras.layers.Dense(d_ff, activation='gelu'),
            tf.keras.layers.Dropout(dropout_rate),
            tf.keras.layers.Dense(d_model),
            tf.keras.layers.Dropout(dropout_rate)
        ])

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout = tf.keras.layers.Dropout(dropout_rate)

    def call(self, x, training=False):
        # Pre-Norm + Attention
        norm_x = self.layernorm1(x)
        attn_output = self.mha(norm_x, norm_x, training=training)
        x = x + self.dropout(attn_output, training=training)

        # Pre-Norm + FFN
        norm_x = self.layernorm2(x)
        ffn_output = self.ffn(norm_x, training=training)
        x = x + ffn_output

        return x

    def get_config(self):
        config = super().get_config()
        config.update({
            'd_model': self.d_model,
            'num_heads': self.num_heads,
            'd_ff': self.d_ff,
            'dropout_rate': self.dropout_rate,
        })
        return config


# ============================================================================
# 第二部分：完整的 Vision Transformer 模型
# ============================================================================

@tf.keras.utils.register_keras_serializable()
class VisionTransformer(tf.keras.Model):
    """
    Vision Transformer (ViT) 用于图像分类

    架构:
    1. 图像分块 + 线性嵌入
    2. 添加 [CLS] token（用于分类）
    3. 添加位置嵌入
    4. N 层 Transformer Encoder
    5. 取 [CLS] token 的输出做分类

    图解:
    ┌─────────────────────────────────────────────────────┐
    │  输入图像 (150, 150, 3)                              │
    └──────────────────────┬──────────────────────────────┘
                           ↓
    ┌─────────────────────────────────────────────────────┐
    │  分成 10×10 = 100 个 patches                         │
    │  每个 patch: (15, 15, 3) → 线性投影 → (d_model,)     │
    └──────────────────────┬──────────────────────────────┘
                           ↓
    ┌─────────────────────────────────────────────────────┐
    │  [CLS] + [Patch1] + [Patch2] + ... + [Patch100]     │
    │  序列长度: 1 + 100 = 101                             │
    └──────────────────────┬──────────────────────────────┘
                           ↓
    ┌─────────────────────────────────────────────────────┐
    │  + Position Embedding (可学习)                       │
    └──────────────────────┬──────────────────────────────┘
                           ↓
    ┌─────────────────────────────────────────────────────┐
    │  Transformer Encoder × N                             │
    └──────────────────────┬──────────────────────────────┘
                           ↓
    ┌─────────────────────────────────────────────────────┐
    │  取 [CLS] token 的输出                               │
    └──────────────────────┬──────────────────────────────┘
                           ↓
    ┌─────────────────────────────────────────────────────┐
    │  Dense → 分类结果 (猫/狗)                            │
    └─────────────────────────────────────────────────────┘
    """

    def __init__(self, image_size=150, patch_size=15, num_classes=1,
                 d_model=256, num_heads=4, d_ff=512, num_layers=4,
                 dropout_rate=0.1):
        super().__init__()

        # 保存配置用于序列化
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_classes = num_classes
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate

        self.num_patches = (image_size // patch_size) ** 2

        # 1. Patch Embedding
        self.patch_embedding = PatchEmbedding(image_size, patch_size, d_model)

        # 2. [CLS] token (可学习的分类 token)
        self.cls_token = self.add_weight(
            name='cls_token',
            shape=(1, 1, d_model),
            initializer='zeros',
            trainable=True
        )

        # 3. Position Embedding (可学习)
        # +1 是因为加了 [CLS] token
        self.position_embedding = self.add_weight(
            name='position_embedding',
            shape=(1, self.num_patches + 1, d_model),
            initializer='zeros',
            trainable=True
        )

        # 4. Transformer Encoder 层
        self.encoder_layers = [
            ViTEncoder(d_model, num_heads, d_ff, dropout_rate)
            for _ in range(num_layers)
        ]

        # 5. 最终层归一化
        self.final_norm = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        # 6. 分类头
        self.classifier = tf.keras.Sequential([
            tf.keras.layers.Dense(d_model, activation='gelu'),
            tf.keras.layers.Dropout(dropout_rate),
            tf.keras.layers.Dense(num_classes, activation='sigmoid')  # 二分类
        ])

        self.dropout = tf.keras.layers.Dropout(dropout_rate)

    def call(self, images, training=False):
        """
        参数:
            images: (batch_size, image_size, image_size, 3)

        返回:
            logits: (batch_size, num_classes)
        """
        batch_size = tf.shape(images)[0]

        # 1. Patch Embedding
        x = self.patch_embedding(images)  # (batch, num_patches, d_model)

        # 2. 添加 [CLS] token
        cls_tokens = tf.broadcast_to(self.cls_token, [batch_size, 1, self.d_model])
        x = tf.concat([cls_tokens, x], axis=1)  # (batch, num_patches+1, d_model)

        # 3. 添加位置嵌入
        x = x + self.position_embedding
        x = self.dropout(x, training=training)

        # 4. Transformer Encoder
        for encoder_layer in self.encoder_layers:
            x = encoder_layer(x, training=training)

        # 5. 最终归一化
        x = self.final_norm(x)

        # 6. 取 [CLS] token 的输出用于分类
        cls_output = x[:, 0, :]  # (batch, d_model)

        # 7. 分类
        output = self.classifier(cls_output)  # (batch, num_classes)

        return output

    def get_config(self):
        """返回模型配置，用于序列化"""
        config = super().get_config()
        config.update({
            'image_size': self.image_size,
            'patch_size': self.patch_size,
            'num_classes': self.num_classes,
            'd_model': self.d_model,
            'num_heads': self.num_heads,
            'd_ff': self.d_ff,
            'num_layers': self.num_layers,
            'dropout_rate': self.dropout_rate,
        })
        return config

    @classmethod
    def from_config(cls, config):
        """从配置创建模型"""
        # 移除 Keras 自动添加的额外参数
        config = config.copy()
        config.pop('name', None)
        config.pop('trainable', None)
        config.pop('dtype', None)
        return cls(**config)


# ============================================================================
# 第三部分：对比 CNN 模型（你之前的实现）
# ============================================================================

def create_cnn_model():
    """
    创建 CNN 模型（与 book5.2.5.py 相同）
    """
    model = tf.keras.Sequential([
        # 卷积块 1
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
        tf.keras.layers.MaxPooling2D(2, 2),

        # 卷积块 2
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),

        # 卷积块 3
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),

        # 卷积块 4
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),

        # 分类头
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    return model


def create_vit_model():
    """
    创建 Vision Transformer 模型
    """
    model = VisionTransformer(
        image_size=150,
        patch_size=15,      # 150/15 = 10x10 = 100 patches
        num_classes=1,      # 二分类
        d_model=256,
        num_heads=4,
        d_ff=512,
        num_layers=4,
        dropout_rate=0.1
    )

    return model


# ============================================================================
# 第四部分：模型对比分析
# ============================================================================

def compare_models():
    """对比 CNN 和 ViT 模型"""
    print("=" * 70)
    print("CNN vs Vision Transformer 模型对比")
    print("=" * 70)

    # 创建模型
    cnn = create_cnn_model()
    vit = create_vit_model()

    # 构建模型（需要先调用一次）
    dummy_input = tf.random.normal((1, 150, 150, 3))
    _ = cnn(dummy_input)
    _ = vit(dummy_input)

    # 统计参数
    cnn_params = sum([tf.reduce_prod(var.shape).numpy() for var in cnn.trainable_variables])
    vit_params = sum([tf.reduce_prod(var.shape).numpy() for var in vit.trainable_variables])

    print(f"\n【参数量对比】")
    print(f"  CNN:  {cnn_params:>10,} 参数")
    print(f"  ViT:  {vit_params:>10,} 参数")

    print(f"\n【结构对比】")
    print("\nCNN 结构:")
    cnn.summary()

    print("\n" + "=" * 70)
    print("\nViT 结构:")
    vit.summary()

    print(f"\n【计算流程对比】")
    print("""
    CNN:
    ┌────────────────────────────────────────────────────────────────┐
    │ (150,150,3) → Conv → (74,74,32) → Pool → (37,37,32)           │
    │            → Conv → (35,35,64) → Pool → (17,17,64)            │
    │            → Conv → (15,15,128) → Pool → (7,7,128)            │
    │            → Conv → (5,5,128) → Pool → (2,2,128)              │
    │            → Flatten → (512,) → Dense → (1,)                  │
    └────────────────────────────────────────────────────────────────┘

    ViT:
    ┌────────────────────────────────────────────────────────────────┐
    │ (150,150,3) → 分成 100 个 patches                              │
    │            → 每个 patch (15,15,3)=675 维 → 投影到 256 维        │
    │            → 加入 [CLS] token → 101 个 tokens                  │
    │            → + Position Embedding                              │
    │            → Transformer × 4 层                                │
    │            → 取 [CLS] 输出 → Dense → (1,)                      │
    └────────────────────────────────────────────────────────────────┘
    """)


# ============================================================================
# 第五部分：训练猫狗分类器
# ============================================================================

def train_cats_dogs_classifier(model_type='vit', epochs=10):
    """
    训练猫狗分类器

    参数:
        model_type: 'cnn' 或 'vit'
        epochs: 训练轮数
    """
    print("=" * 70)
    print(f"使用 {model_type.upper()} 训练猫狗分类器")
    print("=" * 70)

    # 数据目录（与你的 book5.2.5.py 相同）
    base_dir = '/Volumes/Lssd/develop/clProject/1-MachineLearn/tensorFlowLearn/data/cats_and_dogs_small'
    train_dir = os.path.join(base_dir, 'train')
    validation_dir = os.path.join(base_dir, 'validation')

    # 检查数据是否存在
    if not os.path.exists(train_dir):
        print(f"\n错误: 找不到数据目录 {train_dir}")
        print("请先准备好猫狗数据集，参考 book5.2.5.py 中的 copyData() 函数")
        return None

    # 数据增强 (TF 2.16+ 使用 keras.src.legacy)
    try:
        from keras.src.legacy.preprocessing.image import ImageDataGenerator
    except ImportError:
        from tensorflow.keras.preprocessing.image import ImageDataGenerator

    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
    )

    test_datagen = ImageDataGenerator(rescale=1./255)

    # 创建数据生成器
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(150, 150),
        batch_size=20,
        class_mode='binary'
    )

    validation_generator = test_datagen.flow_from_directory(
        validation_dir,
        target_size=(150, 150),
        batch_size=20,
        class_mode='binary'
    )

    # 创建模型
    if model_type == 'cnn':
        model = create_cnn_model()
    else:
        model = create_vit_model()

    # 编译模型
    model.compile(
        loss='binary_crossentropy',
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        metrics=['accuracy']
    )

    # 训练
    history = model.fit(
        train_generator,
        steps_per_epoch=100,
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=50
    )

    # 保存模型
    model_name = f'cats_and_dogs_{model_type}.h5'
    model.save(model_name)
    print(f"\n模型已保存为 {model_name}")

    return history


def plot_training_history(history, title='Training History'):
    """绘制训练历史"""
    if history is None:
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # 损失曲线
    axes[0].plot(history.history['loss'], 'b-', label='Train Loss')
    axes[0].plot(history.history['val_loss'], 'r-', label='Val Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # 准确率曲线
    axes[1].plot(history.history['accuracy'], 'b-', label='Train Acc')
    axes[1].plot(history.history['val_accuracy'], 'r-', label='Val Acc')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig('training_history.png', dpi=150)
    plt.show()


# ============================================================================
# 第六部分：可视化 ViT 的注意力
# ============================================================================

def visualize_vit_attention():
    """
    可视化 Vision Transformer 的注意力

    这可以帮助理解模型在"看"图像的哪些部分
    """
    print("\n" + "=" * 70)
    print("Vision Transformer 注意力可视化")
    print("=" * 70)

    print("""
    ViT 的注意力可以告诉我们：
    - 模型在分类时关注图像的哪些区域
    - [CLS] token 与哪些 patches 有强关联

    例如，对于一张猫的图片：
    ┌─────────────────────────────────┐
    │  [低] [低] [低] [低] [低]        │
    │  [低] [高] [高] [高] [低]        │  ← 模型关注猫脸区域
    │  [低] [高] [猫脸] [高] [低]      │
    │  [低] [高] [高] [高] [低]        │
    │  [低] [低] [低] [低] [低]        │
    └─────────────────────────────────┘

    注意力权重高的区域 = 模型认为重要的区域
    """)


# ============================================================================
# 主函数
# ============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("CNN vs Transformer 架构对比 + Vision Transformer 猫狗分类")
    print("=" * 70)

    # 1. 对比模型结构和参数
    compare_models()

    # 2. 询问是否训练
    print("\n" + "=" * 70)
    print("训练选项")
    print("=" * 70)
    print("""
    你可以选择运行以下训练：

    1. 使用 CNN 训练（快，效果好）:
       history = train_cats_dogs_classifier('cnn', epochs=30)

    2. 使用 ViT 训练（慢，需要更多数据/epochs）:
       history = train_cats_dogs_classifier('vit', epochs=50)

    注意：ViT 通常需要更多数据才能达到 CNN 的效果，
    因为它没有 CNN 的"归纳偏置"（局部性、平移不变性）。
    对于小数据集，CNN 通常更好。
    """)

    # 可视化注意力
    visualize_vit_attention()

    print("\n" + "=" * 70)
    print("总结")
    print("=" * 70)
    print("""
    【什么时候用 CNN？】
    - 图像数据较少（< 10000 张）
    - 需要快速训练和推理
    - 局部特征很重要（纹理、边缘）

    【什么时候用 Transformer？】
    - 大量数据（> 100000 张）
    - 需要理解全局关系
    - 可以使用预训练模型（如 ViT-Base）

    【实际建议】
    对于猫狗分类这样的任务，CNN 就足够好了。
    Transformer 在大规模预训练后效果更好（如 ImageNet-21K 预训练的 ViT）。
    """)
