from PIL import Image
import os

def convert_images_to_rgb(folder_path):
    for class_folder in os.listdir(folder_path):
        class_path = os.path.join(folder_path, class_folder)
        if not os.path.isdir(class_path):
            continue
        for img_file in os.listdir(class_path):
            img_path = os.path.join(class_path, img_file)
            try:
                img = Image.open(img_path)
                # Convert to RGB if not already
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                    img.save(img_path)
            except Exception as e:
                print(f"Error converting {img_path}: {e}")

# Convert train and test folders
convert_images_to_rgb("Dataset/train")
convert_images_to_rgb("Dataset/test")

print("✅ Conversion to RGB completed!")
