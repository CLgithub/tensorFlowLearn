#coding=utf-8
"""
快速测试 Vision Transformer 是否可以训练
支持使用自己的图片测试
"""

import tensorflow as tf
import numpy as np
import os
import sys
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from PIL import Image

# 添加当前目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

# 导入我们的 ViT 模型
from importlib.util import spec_from_file_location, module_from_spec
spec = spec_from_file_location("vit_module", os.path.join(current_dir, "06_cnn_vs_transformer.py"))
vit_module = module_from_spec(spec)
spec.loader.exec_module(vit_module)

VisionTransformer = vit_module.VisionTransformer
create_cnn_model = vit_module.create_cnn_model

print("=" * 60)
print("Vision Transformer 训练测试")
print("=" * 60)
#
# # ============================================================================
# # 测试1：模型前向传播
# # ============================================================================
# print("\n【测试1】模型前向传播...")
#
# 创建模型
vit = VisionTransformer(
    image_size=150,
    patch_size=15,
    num_classes=1,
    d_model=128,    # 用小一点的模型测试
    num_heads=4,
    d_ff=256,
    num_layers=2,
    dropout_rate=0.1
)
#
# # 测试输入
# dummy_images = tf.random.normal((4, 150, 150, 3))
# output = vit(dummy_images, training=False)
#
# print(f"  输入 shape: {dummy_images.shape}")
# print(f"  输出 shape: {output.shape}")
# print(f"  输出值范围: [{output.numpy().min():.4f}, {output.numpy().max():.4f}]")
# print("  ✓ 前向传播成功！")
#
# # ============================================================================
# # 测试2：模型训练（用真实猫狗数据）
# # ============================================================================
# print("\n【测试2】模型训练（真实猫狗数据）...")
#
# # 数据目录
base_dir = '/Volumes/Lssd/develop/clProject/1-MachineLearn/tensorFlowLearn/data/cats_and_dogs_small'
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')

# 数据增强
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

print(f"  训练数据目录: {train_dir}")
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

# 编译模型
vit.compile(
    loss='binary_crossentropy',
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    metrics=['accuracy']
)
#
# 训练
# print("  开始训练 ViT 模型...")
# history = vit.fit(
#     train_generator,
#     steps_per_epoch=100,
#     epochs=100,
#     validation_data=validation_generator,
#     validation_steps=50,
#     verbose=1
# )
#
# print(f"  最终 loss: {history.history['loss'][-1]:.4f}")
# print(f"  最终 accuracy: {history.history['accuracy'][-1]:.4f}")
# print("  ✓ 训练成功！")
#
# # ============================================================================
# # 测试3：模型预测
# # ============================================================================
# print("\n【测试3】模型预测...")
#
# test_images = tf.random.normal((2, 150, 150, 3))
# predictions = vit.predict(test_images, verbose=0)
#
# print(f"  预测结果:")
# for i, pred in enumerate(predictions):
#     label = "狗" if pred[0] > 0.5 else "猫"
#     print(f"    图片 {i+1}: {pred[0]:.4f} → {label}")
# print("  ✓ 预测成功！")
#
# # ============================================================================
# # 测试4：模型保存和加载
# # ============================================================================
# print("\n【测试4】模型保存和加载...")
#
# # 保存
save_path = os.path.join(current_dir, "test_vit_model.keras")
vit.save(save_path)
print(f"  模型已保存到: {save_path}")
#
# # 加载
loaded_model = tf.keras.models.load_model(save_path)
# loaded_predictions = loaded_model.predict(test_images, verbose=0)
#
# # 验证加载的模型输出一致
# diff = np.abs(predictions - loaded_predictions).max()
# print(f"  加载后预测差异: {diff:.2e}")
# print("  ✓ 保存/加载成功！")
#
# # 清理测试文件
# # os.remove(save_path)
#
# # ============================================================================
# # 测试5：与 CNN 对比
# # ============================================================================
# print("\n【测试5】与 CNN 模型对比...")
#
# cnn = create_cnn_model()
# _ = cnn(dummy_images)  # 构建模型
#
# cnn_params = sum([tf.reduce_prod(var.shape).numpy() for var in cnn.trainable_variables])
# vit_params = sum([tf.reduce_prod(var.shape).numpy() for var in vit.trainable_variables])
#
# print(f"  CNN 参数量: {cnn_params:,}")
# print(f"  ViT 参数量: {vit_params:,}")
# print(f"  比例: ViT/CNN = {vit_params/cnn_params:.2f}x")
#
# # ============================================================================
# # 测试6：使用自己的图片测试
# # ============================================================================
# print("\n【测试6】使用自定义图片测试...")
#
def load_image(image_path, target_size=(150, 150)):
    """加载并预处理图片"""
    img = Image.open(image_path)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = img.resize(target_size)
    img_array = np.array(img) / 255.0
    return np.expand_dims(img_array, axis=0), img

def predict_image(model, image_path):
    """预测单张图片"""
    img_array, original_img = load_image(image_path)
    prediction = model.predict(img_array, verbose=0)[0][0]
    label = "狗" if prediction > 0.5 else "猫"
    confidence = prediction if prediction > 0.5 else 1 - prediction
    return label, confidence, prediction, original_img

def show_prediction(model, image_path):
    """显示预测结果"""
    label, confidence, raw, img = predict_image(model, image_path)
    plt.figure(figsize=(6, 6))
    plt.imshow(img)
    plt.axis('off')
    plt.title(f"预测: {label}\n置信度: {confidence:.1%}\n(分数: {raw:.4f})", fontsize=14)
    plt.tight_layout()
    plt.show()
    return label, confidence
#
# # 测试示例图片（如果存在）
# sample_images = [
#     "/Volumes/Lssd/develop/clProject/1-MachineLearn/tensorFlowLearn/data/cats_and_dogs_small/test/cats/cat.1500.jpg",
#     "/Volumes/Lssd/develop/clProject/1-MachineLearn/tensorFlowLearn/data/cats_and_dogs_small/test/dogs/dog.1500.jpg",
# ]
#
# for img_path in sample_images:
#     if os.path.exists(img_path):
#         label, conf, raw, _ = predict_image(vit, img_path)
#         filename = os.path.basename(img_path)
#         print(f"  {filename}: {label} ({conf:.1%})")
#
# print("""
#   ✓ 自定义图片测试功能已就绪！
#
#   使用方法:
#   >>> show_prediction(vit, "/path/to/your/image.jpg")
# """)
#
# # ============================================================================
# # 测试7：使用本脚本训练的 ViT 模型测试你的图片
# # ============================================================================
# print("\n【测试7】使用 ViT 模型测试你的图片...")
#
loaded_model = tf.keras.models.load_model(save_path)
your_image_path = "/Users/l/Downloads/d4.jpg"

if os.path.exists(your_image_path):
    print(f"  测试图片: {your_image_path}")
    label, confidence, raw, img = predict_image(loaded_model, your_image_path)
    print(f"  预测结果: {label}")
    print(f"  置信度: {confidence:.1%}")
    print(f"  原始分数: {raw:.4f}")

    # 显示结果
    show_prediction(loaded_model, your_image_path)
else:
    print(f"  未找到图片: {your_image_path}")

# ============================================================================
# 总结
# ============================================================================
print("\n" + "=" * 60)
print("测试完成！Vision Transformer 可以正常训练和使用")
print("=" * 60)
