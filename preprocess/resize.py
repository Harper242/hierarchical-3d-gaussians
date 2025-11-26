import os
from PIL import Image

def resize_and_rename_images(folder_path, target_width=768, target_height=512):
    # 确保输出目录存在
    output_folder = r'../dataset/CUS3D/inputs/masks'
    os.makedirs(output_folder, exist_ok=True)

    # 获取文件夹中所有的图像文件
    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.PNG', '.JPG'))]
    image_files.sort()  # 按文件名排序

    # 遍历每一个图像文件，调整分辨率并重命名
    for idx, image_file in enumerate(image_files, start=0):
        image_path = os.path.join(folder_path, image_file)
        img = Image.open(image_path)

        # 调整分辨率
        img = img.resize((target_width, target_height), Image.NEAREST)

        # 保存图像，按顺序重命名
        new_name = f'{idx:05d}.JPG'
        # output_path = os.path.join(output_folder, new_name)
        output_path = os.path.join(output_folder, image_file)
        img.save(output_path)

        # print(f'Resized and renamed {image_file} to {new_name}')
        print(f'Resized {image_file}')
        # break

if __name__ == '__main__':
    folder_path = r'../dataset/CUS3D_Comprehensive_Urban_Scale_Semantic_Segmentation_3D_Dataset/CUS3D_Images_Data/CUS3D_Semantic_Images'  # 替换为你的文件夹路径
    resize_and_rename_images(folder_path)
