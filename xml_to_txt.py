import os
import xml.etree.ElementTree as ET
from PIL import Image

# Define paths
annotations_dir = '/PCB_DATASET/Annotations'  # Folder where XML files are located
images_dir = '/PCB_DATASET/images'  # Folder where images are located
output_dir = '/Input_Data'  # Folder where YOLO-format annotations will be saved
classes = ["missing_hole", "mouse_bite", "open_circuit", "short", "spur", "spurious_copper"]

# Ensure output folder exists
if not os.path.exists(output_dir):
    os.makedirs(output_dir)


# Update the path in XML files
def update_xml_paths(xml_file, img_dir):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    for path_elem in root.findall('path'):
        # Assuming the file names are consistent between XML and image directories
        img_file = path_elem.text.split('/')[-1]
        new_path = os.path.join(img_dir, img_file)
        path_elem.text = new_path
    tree.write(xml_file)


# Convert Pascal VOC to YOLO
def convert_voc_to_yolo(xml_file, img_size):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    yolo_annotations = []
    for obj in root.findall('object'):
        class_name = obj.find('name').text.lower()  # Convert class names to lowercase
        if class_name not in classes:
            print(f"Class '{class_name}' not in defined classes, skipping.")
            continue
        class_id = classes.index(class_name)

        bndbox = obj.find('bndbox')
        xmin = int(bndbox.find('xmin').text)
        ymin = int(bndbox.find('ymin').text)
        xmax = int(bndbox.find('xmax').text)
        ymax = int(bndbox.find('ymax').text)

        # Convert to YOLO format
        dw = 1.0 / img_size[0]
        dh = 1.0 / img_size[1]
        x_center = (xmin + xmax) / 2.0 * dw
        y_center = (ymin + ymax) / 2.0 * dh
        width = (xmax - xmin) * dw
        height = (ymax - ymin) * dh

        yolo_annotations.append(f"{class_id} {x_center} {y_center} {width} {height}\n")

    return yolo_annotations


# Process all XML files in subfolders
for subfolder in os.listdir(annotations_dir):
    subfolder_path = os.path.join(annotations_dir, subfolder)
    if not os.path.isdir(subfolder_path):
        continue

    img_subfolder_path = os.path.join(images_dir, subfolder)
    output_subfolder_path = os.path.join(output_dir, subfolder)
    if not os.path.exists(output_subfolder_path):
        os.makedirs(output_subfolder_path)

    for xml_file in os.listdir(subfolder_path):
        if not xml_file.endswith(".xml"):
            continue
        xml_path = os.path.join(subfolder_path, xml_file)

        img_file = xml_file.replace(".xml", ".jpg")  # Assuming your images are in .jpg format
        img_path = os.path.join(img_subfolder_path, img_file)

        if not os.path.exists(img_path):
            print(f"Image {img_file} not found in {img_subfolder_path}, skipping.")
            continue

        # Update the path in the XML file
        update_xml_paths(xml_path, img_subfolder_path)

        # Get image size (width, height)
        try:
            img = Image.open(img_path)
            img_size = img.size
        except Exception as e:
            print(f"Error opening image {img_path}: {e}")
            continue

        # Convert annotations
        yolo_annotations = convert_voc_to_yolo(xml_path, img_size)

        # Debug: Print out the first few lines of YOLO annotations
        if yolo_annotations:
            print(f"YOLO annotations for {xml_file}:")
            print(''.join(yolo_annotations[:5]))  # Print the first 5 annotations for debugging

        # Write YOLO annotations to file
        output_txt_path = os.path.join(output_subfolder_path, xml_file.replace(".xml", ".txt"))
        with open(output_txt_path, 'w') as out_file:
            out_file.writelines(yolo_annotations)

        if yolo_annotations:
            print(f"Converted {xml_file} to YOLO format in {output_subfolder_path}.")
        else:
            print(f"No annotations found in {xml_file}.")
