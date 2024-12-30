import os
import cv2

# 文件夹路径
folder_path = r'E:\segmentation-format-fix-main\s'  # 替换成你的文件夹路径

# 遍历文件夹中的图像文件
for filename in os.listdir(folder_path):
    if filename.endswith(('.jpg', '.jpeg', '.png')):  # 确保文件是图像文件
        # 读取灰度图像
        image_path = os.path.join(folder_path, filename)
        gray_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        # 将所有像素值为 255 的像素替换为 1

        gray_image[gray_image == 255] = 1
        # gray_image[gray_image == 2] = 75



        # 保存修改后的图像
        output_path = os.path.join(r'E:\segmentation-format-fix-main\se', filename)  # 替换成你想要保存的输出文件夹路径
        cv2.imwrite(output_path, gray_image)
