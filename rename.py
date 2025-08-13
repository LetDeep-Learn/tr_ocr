import os

# Folder containing your images
folder = "modi_dataset/sys_images"

for filename in os.listdir(folder):
    if filename.startswith("clean_") and filename.endswith(".png"):
        new_name = filename.replace("clean_", "image_", 1)
        old_path = os.path.join(folder, filename)
        new_path = os.path.join(folder, new_name)
        os.rename(old_path, new_path)

print("Renaming complete!")
