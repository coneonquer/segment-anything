import os
import sys
import time
import argparse
import time

import cv2
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from PIL import Image

# 这里是导入segment_anything包
from segment_anything import SamPredictor, sam_model_registry
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
from segment_anything import sam_model_registry

# 输入必要的参数
model_path = r'..\checkpoint\sam_vit_b_01ec64.pth'
image_path = r'..\notebooks\images\2007_000027.jpg'
output_folder = r"..\output\demo1"
# output_path = r'C:\AI\segment-anything-main\demo_mask\demo.png'

# 确保输出文件夹存在
os.makedirs(output_folder, exist_ok=True)
# 这里的model_path是模型的路径，image_path是输入的图片路径，output_path是输出的图片路径

# 这里是加载模型


# 官方demo加载模型的方式
sam = sam_model_registry["vit_b"](checkpoint=model_path)
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#sam = sam.to(device)

# 输出模型加载完成的current时间

current_time1 = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
print("Model loaded done", current_time1)
print("1111111111111111111111111111111111111111111111111111")

# 这里是加载模型，这里的model_path是模型的路径，sam_model_registry是模型的名称

# 这里是加载图片
image = cv2.imread(image_path)
# 输出图片加载完成的current时间
current_time2 = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
print("Image loaded done", current_time2)
print("2222222222222222222222222222222222222222222222222222")

# 这里是加载图片，这里的image_path是图片的路径

# 这里是预测,不用提示词,进行全图分割
mask_generator = SamAutomaticMaskGenerator(sam)
masks = mask_generator.generate(image)

# 使用提示词,进行局部分割
# predictor = SamPredictor(sam)
# predictor.set_image(image)
# masks, _, _ = predictor.predict(point_coords=None, point_labels=None, box=None, mask_input=None, multimask_output=True, return_logits=False)

# 输出预测完成的current时间
current_time3 = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
print("Predict done", current_time3)
print("3333333333333333333333333333333333333333333333333333")

# #展示预测结果img和mask
# plt.figure(figsize=(10, 10))
# plt.subplot(1, 2, 1)
# plt.imshow(image)
# plt.subplot(1, 2, 2)
# plt.imshow(masks[0])
# plt.show()


# 保存mask
print(masks[0])

# # mask_array = masks[0]['segmentation']


# # mask_uint8 = (mask_array * 255).astype(np.uint8)

# cv2.imwrite(output_path, mask_uint8)

# 循环保存mask
# 遍历 masks 列表并保存每个掩码
for i, mask in enumerate(masks):
    mask_array = mask['segmentation']
    mask_uint8 = (mask_array * 255).astype(np.uint8)

    # 为每个掩码生成一个唯一的文件名
    output_file = os.path.join(output_folder, f"mask_{i + 1}.png")

    # 保存掩码
    cv2.imwrite(output_file, mask_uint8)

# 输出完整的mask
# 获取输入图像的尺寸
height, width, _ = image.shape

# 创建一个全零数组，用于合并掩码
merged_mask = np.zeros((height, width), dtype=np.uint8)

# 遍历 masks 列表并合并每个掩码
for i, mask in enumerate(masks):
    mask_array = mask['segmentation']
    mask_uint8 = (mask_array * 255).astype(np.uint8)

    # 为每个掩码生成一个唯一的文件名
    output_file = os.path.join(output_folder, f"mask_{i + 1}.png")

    # 保存掩码
    cv2.imwrite(output_file, mask_uint8)

    # 将当前掩码添加到合并掩码上
    merged_mask = np.maximum(merged_mask, mask_uint8)

# 保存合并后的掩码
merged_output_file = os.path.join(output_folder, "mask_all.png")
cv2.imwrite(merged_output_file, merged_mask)
# #释放cv2
# cv2.destroyAllWindows()