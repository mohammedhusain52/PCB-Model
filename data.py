import os
import shutil
import random

# Define paths
source_img_dir = '/PCB_DATASET/images'
source_label_dir = '/Input_Data'
dest_img_dir = '/dataset/images'
dest_label_dir = '/dataset/labels'

# Ensure destination directories exist
for split in ['train', 'val', 'test']:
    for class_folder in os.listdir(source_img_dir):
        os.makedirs(os.path.join(dest_img_dir, split, class_folder), exist_ok=True)
        os.makedirs(os.path.join(dest_label_dir, split, class_folder), exist_ok=True)


def debug_naming_consistency(class_folder):
    img_folder = os.path.join(source_img_dir, class_folder)
    label_folder = os.path.join(source_label_dir, class_folder)

    img_files = {f for f in os.listdir(img_folder) if f.endswith('.jpg')}
    label_files = {f for f in os.listdir(label_folder) if f.endswith('.txt')}

    img_names = {f.replace('.jpg', '') for f in img_files}
    label_names = {f.replace('.txt', '') for f in label_files}

    img_only = img_names - label_names
    label_only = label_names - img_names

    if img_only or label_only:
        print(f"Class: {class_folder}")
        if img_only:
            print("Image names without matching labels:", img_only)
        if label_only:
            print("Label names without matching images:", label_only)


def create_empty_labels(img_files, label_folder):
    for img_file in img_files:
        label_file = img_file.replace('.jpg', '.txt')
        label_path = os.path.join(label_folder, label_file)
        if not os.path.exists(label_path):
            with open(label_path, 'w') as f:
                f.write('')  # Create an empty label file


def split_class_files(class_folder, split_ratio):
    img_folder = os.path.join(source_img_dir, class_folder)
    label_folder = os.path.join(source_label_dir, class_folder)

    img_files = {f for f in os.listdir(img_folder) if f.endswith('.jpg')}
    label_files = {f for f in os.listdir(label_folder) if f.endswith('.txt')}

    # Create empty labels for missing label files
    create_empty_labels(img_files, label_folder)

    # Filter out files with no corresponding pair
    valid_img_files = {f for f in img_files if f.replace('.jpg', '.txt') in label_files}

    if not valid_img_files:
        print(f"No valid image files found for class: {class_folder}")
        return [], [], []

    valid_img_files = list(valid_img_files)

    # Shuffle files and split
    random.shuffle(valid_img_files)
    split1 = int(len(valid_img_files) * split_ratio[0])
    split2 = int(len(valid_img_files) * (split_ratio[0] + split_ratio[1]))

    train_files = valid_img_files[:split1]
    val_files = valid_img_files[split1:split2]
    test_files = valid_img_files[split2:]

    return train_files, val_files, test_files


def move_files(class_folder, train_files, val_files, test_files):
    img_folder = os.path.join(source_img_dir, class_folder)
    label_folder = os.path.join(source_label_dir, class_folder)

    for file_list, split in zip([train_files, val_files, test_files], ['train', 'val', 'test']):
        img_dest_folder = os.path.join(dest_img_dir, split, class_folder)
        label_dest_folder = os.path.join(dest_label_dir, split, class_folder)

        for file in file_list:
            shutil.copy(os.path.join(img_folder, file), os.path.join(img_dest_folder, file))
            shutil.copy(os.path.join(label_folder, file.replace('.jpg', '.txt')),
                        os.path.join(label_dest_folder, file.replace('.jpg', '.txt')))


# Debug naming consistency for each class
for class_folder in os.listdir(source_img_dir):
    if os.path.isdir(os.path.join(source_img_dir, class_folder)):
        debug_naming_consistency(class_folder)

# Define split ratios
split_ratios = {'train': 0.7, 'val': 0.2, 'test': 0.1}

# Process each class folder
for class_folder in os.listdir(source_img_dir):
    if os.path.isdir(os.path.join(source_img_dir, class_folder)):
        train_files, val_files, test_files = split_class_files(class_folder, list(split_ratios.values()))
        move_files(class_folder, train_files, val_files, test_files)

print("Dataset splitting complete.")
