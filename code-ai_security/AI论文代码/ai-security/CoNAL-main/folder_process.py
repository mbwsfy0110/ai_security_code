import os
import shutil
import numpy as np
import os
import xml.etree.ElementTree as ET
import cv2


test_label = np.load("./data/LabelMe/prepared/test/labels_test.npy")

class_name = ["highway", "insidecity", "tallbuilding", "street", "forest", "coast", "mountain", "opencountry"]


def get_file_names(folder_path):
    file_names = os.listdir(folder_path)
    return file_names


# 获取test——data文件中左右文件的名字
def filename_list(folder_path):
    # 示例用法
    filenameList = []
    for i in class_name:
        filenameList.append(get_file_names(folder_path + "/%s" % i))
    return filenameList


# 加载记录并标记对应的错误标签情况-->单独保存数据集
err_record = np.load("./testResult/err_record.npy")


# 复制到新的文件夹下
def get_image_dimensions(image_path):
    import cv2
    image = cv2.imread(image_path)
    height, width, _ = image.shape
    return width, height


# 图片绝对路径，标签名，图片坐在文件夹名称
def create_xml_annotation(Absolute_path, label, father_folder, file_name):
    # 创建根节点
    annotation = ET.Element("annotation")
    # 添加子节点
    folder = ET.SubElement(annotation, "folder")
    folder.text = father_folder
    filename = ET.SubElement(annotation, "filename")
    filename.text = file_name
    path = ET.SubElement(annotation, "path")
    path.text = Absolute_path
    source = ET.SubElement(annotation, "source")
    database = ET.SubElement(source, "database")
    database.text = "Unknown"
    # 获取图片的长宽高
    image_size = os.path.getsize(Absolute_path)
    width, height = get_image_dimensions(Absolute_path)
    size = ET.SubElement(annotation, "size")
    width_elem = ET.SubElement(size, "width")
    width_elem.text = str(width)
    height_elem = ET.SubElement(size, "height")
    height_elem.text = str(height)
    depth = ET.SubElement(size, "depth")
    depth.text = "3"  # 这里假设是RGB图像

    segmented = ET.SubElement(annotation, "segmented")
    segmented.text = "0"
    # 添加每个标签的信息
    obj = ET.SubElement(annotation, "object")
    name = ET.SubElement(obj, "name")
    name.text = label
    pose = ET.SubElement(obj, "pose")
    pose.text = "Unspecified"
    truncated = ET.SubElement(obj, "truncated")
    truncated.text = "0"
    difficult = ET.SubElement(obj, "difficult")
    difficult.text = "0"
    bndbox = ET.SubElement(obj, "bndbox")
    xmin_elem = ET.SubElement(bndbox, "xmin")
    xmin_elem.text = "1"
    ymin_elem = ET.SubElement(bndbox, "ymin")
    ymin_elem.text = "1"
    xmax_elem = ET.SubElement(bndbox, "xmax")
    xmax_elem.text = "256"
    ymax_elem = ET.SubElement(bndbox, "ymax")
    ymax_elem.text = "256"

    # 创建XML树
    tree = ET.ElementTree(annotation)
    return tree

def parse_labelimg_xml(xml_file,dest_pth):
    # 解析XML文件
    tree = ET.parse(xml_file)
    root = tree.getroot()
    # 获取文件夹名
    # folder = root.find("folder").text
    # 获取所有的object标签
    # 获取图片路径
    path = root.find("path").text
    objects = root.findall("object")
    for obj in objects:
        # 获取标签名
        name = obj.find("name").text
        # 创建文件夹
        folder_path = os.path.join(os.getcwd(), dest_pth, name)
        os.makedirs(folder_path, exist_ok=True)
        # 复制图片到相应文件夹下
        image_name = os.path.basename(path)
        new_image_path = os.path.join(folder_path, image_name)
        shutil.copy(path, new_image_path)


def xml_to_data(xml_folder_path, des_path):
    os.makedirs(des_path, exist_ok=True)
    filename_list = get_file_names(xml_folder_path)
    for i in filename_list:
        xml_file = "%s/%s" % (xml_folder_path, i)
        parse_labelimg_xml(xml_file,des_path)


def delete_files_by_name(folder_path, target_name):
    for root, dirs, files in os.walk(folder_path):
        for file_name in files:
            if file_name == target_name:
                file_path = os.path.join(root, file_name)
                try:
                    os.remove(file_path)
                    print(f"Deleted: {file_path}")
                except OSError as e:
                    print(f"Error: {file_path} - {e}")


def copy_data_to_new_folder(source_folder, destination_folder, xml_pth, filename_list):
    # 创建目标文件夹
    os.makedirs(destination_folder, exist_ok=True)
    # 创建标签文件夹
    father_folder = xml_pth
    os.makedirs(father_folder, exist_ok=True)
    for i in range(len(err_record)):
        files_name = filename_list[err_record[i][0]][i % 120]
        temp_folder = source_folder + "/%s" % class_name[err_record[i][0]]
        source_file = os.path.join(temp_folder, files_name)
        if os.path.isfile(source_file):
            shutil.copy(source_file, destination_folder)
            # 创建XML注释
            xml_tree = create_xml_annotation("E:\Code\CoNAL-main\CoNAL-main\%s\%s" % (destination_folder, files_name),
                                             class_name[err_record[i][1]],
                                             father_folder, files_name)
            # 将XML注释写入文件
            xml_file_path = "%s\%s.xml" % (father_folder, files_name[:-4])
            xml_tree.write(xml_file_path)
            # 删除文件夹下原有的文件
            delete_files_by_name(temp_folder, files_name)


if __name__=="__main__":
    # 源文件夹路径
    source_folder = "./testData"
    # 目标文件夹路径
    destination_folder = "err_RecordData"
    filename_list = filename_list("./testData")  # 文件名列表
    # 复制数据
    annotator_pth = "Annotator"
    copy_data_to_new_folder(source_folder, destination_folder, annotator_pth, filename_list=filename_list)
    # 转换成对应数据集
    des_xml_pth = "sys_gene_data"
    xml_to_data(annotator_pth, des_xml_pth)
