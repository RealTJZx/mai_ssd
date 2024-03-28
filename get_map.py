import os
import xml.etree.ElementTree as ET

from PIL import Image
from tqdm import tqdm

from utils.utils import get_classes
from utils.utils_map import get_coco_map, get_map
from ssd import SSD

# if __name__ == "__main__":
# zx
def mk_map(
        classes_path, dataset_path, output_path,
        model_path, backbone,
        input_shape=None,
        anchors_size=None
):
    # zx - 加工处理后的设置参数说明
    # classes_path、dataset_path、output_path、model_path、input_shape、backbone、anchors_size
    curFile_path = os.path.dirname(os.path.abspath(__file__))

    '''
    Recall和Precision不像AP是一个面积的概念，因此在门限值（Confidence）不同时，网络的Recall和Precision值是不同的。
    默认情况下，本代码计算的Recall和Precision代表的是当门限值（Confidence）为0.5时，所对应的Recall和Precision值。

    受到mAP计算原理的限制，网络在计算mAP时需要获得近乎所有的预测框，这样才可以计算不同门限条件下的Recall和Precision值
    因此，本代码获得的map_out/detection-results/里面的txt的框的数量一般会比直接predict多一些，目的是列出所有可能的预测框，
    '''
    #------------------------------------------------------------------------------------------------------------------#
    #   map_mode用于指定该文件运行时计算的内容
    #   map_mode为0代表整个map计算流程，包括获得预测结果、获得真实框、计算VOC_map。
    #   map_mode为1代表仅仅获得预测结果。
    #   map_mode为2代表仅仅获得真实框。
    #   map_mode为3代表仅仅计算VOC_map。
    #   map_mode为4代表利用COCO工具箱计算当前数据集的0.50:0.95map。需要获得预测结果、获得真实框后并安装pycocotools才行
    #-------------------------------------------------------------------------------------------------------------------#
    map_mode        = 0
    #--------------------------------------------------------------------------------------#
    #   此处的classes_path用于指定需要测量VOC_map的类别
    #   一般情况下与训练和预测所用的classes_path一致即可
    #--------------------------------------------------------------------------------------#
    # zx
    classes_path = classes_path
    #--------------------------------------------------------------------------------------#
    #   MINOVERLAP用于指定想要获得的mAP0.x，mAP0.x的意义是什么请同学们百度一下。
    #   比如计算mAP0.75，可以设定MINOVERLAP = 0.75。
    #
    #   当某一预测框与真实框重合度大于MINOVERLAP时，该预测框被认为是正样本，否则为负样本。
    #   因此MINOVERLAP的值越大，预测框要预测的越准确才能被认为是正样本，此时算出来的mAP值越低，
    #--------------------------------------------------------------------------------------#
    MINOVERLAP      = 0.5
    #--------------------------------------------------------------------------------------#
    #   受到mAP计算原理的限制，网络在计算mAP时需要获得近乎所有的预测框，这样才可以计算mAP
    #   因此，confidence的值应当设置的尽量小进而获得全部可能的预测框。
    #   
    #   该值一般不调整。因为计算mAP需要获得近乎所有的预测框，此处的confidence不能随便更改。
    #   想要获得不同门限值下的Recall和Precision值，请修改下方的score_threhold。
    #--------------------------------------------------------------------------------------#
    # zx
    confidence      = 0.02    # default
    # confidence      = 0.1
    #--------------------------------------------------------------------------------------#
    #   预测时使用到的非极大抑制值的大小，越大表示非极大抑制越不严格。
    #   
    #   该值一般不调整。
    #--------------------------------------------------------------------------------------#
    nms_iou         = 0.5
    #---------------------------------------------------------------------------------------------------------------#
    #   Recall和Precision不像AP是一个面积的概念，因此在门限值不同时，网络的Recall和Precision值是不同的。
    #   
    #   默认情况下，本代码计算的Recall和Precision代表的是当门限值为0.5（此处定义为score_threhold）时所对应的Recall和Precision值。
    #   因为计算mAP需要获得近乎所有的预测框，上面定义的confidence不能随便更改。
    #   这里专门定义一个score_threhold用于代表门限值，进而在计算mAP时找到门限值对应的Recall和Precision值。
    #---------------------------------------------------------------------------------------------------------------#
    score_threhold  = 0.5
    #-------------------------------------------------------#
    #   map_vis用于指定是否开启VOC_map计算的可视化
    #-------------------------------------------------------#
    map_vis         = False

    #-------------------------------------------------------#
    #   指向VOC数据集所在的文件夹
    #   默认指向根目录下的VOC数据集
    #-------------------------------------------------------#
    # zx
    # VOCdevkit_path  = 'VOCdevkit'
    VOCdevkit_path = os.path.abspath(os.path.join(curFile_path, dataset_path))
    #-------------------------------------------------------#
    #   结果输出的文件夹，默认为map_out
    #-------------------------------------------------------#
    # zx
    # map_out_path    = 'map_out'
    map_out_path = output_path

    image_ids = open(os.path.join(VOCdevkit_path, "VOC2007/ImageSets/Main/test.txt")).read().strip().split()

    if not os.path.exists(map_out_path):
        os.makedirs(map_out_path)
    if not os.path.exists(os.path.join(map_out_path, 'ground-truth')):
        os.makedirs(os.path.join(map_out_path, 'ground-truth'))
    if not os.path.exists(os.path.join(map_out_path, 'detection-results')):
        os.makedirs(os.path.join(map_out_path, 'detection-results'))
    if not os.path.exists(os.path.join(map_out_path, 'images-optional')):
        os.makedirs(os.path.join(map_out_path, 'images-optional'))

    class_names, _ = get_classes(classes_path)

    # zx - 预测结果
    if map_mode == 0 or map_mode == 1:
        print("Load model.")

        # zx
        # ssd = SSD(confidence = confidence, nms_iou = nms_iou)
        # zx
        # - 自定义参数
        # 权重，类别，输入图片大小，提取网络，锚框设置
        # model_path(str)、classes_path(str)、input_shape(list)、backbone(str)、anchors_size(list)
        ssd = SSD(
            model_path=model_path,
            classes_path=classes_path,
            input_shape=input_shape if input_shape else [480, 480],
            backbone=backbone,
            anchors_size=anchors_size if anchors_size else [21, 45, 99, 153, 207, 261, 315],
            confidence=confidence,
            nms_iou=nms_iou
        )
        print("Load model done.")

        print("Get predict result.")
        for image_id in tqdm(image_ids):
            image_path  = os.path.join(VOCdevkit_path, "VOC2007/JPEGImages/"+image_id+".jpg")
            image       = Image.open(image_path)
            if map_vis:
                image.save(os.path.join(map_out_path, "images-optional/" + image_id + ".jpg"))
            ssd.get_map_txt(image_id, image, class_names, map_out_path)
        print("Get predict result done.")

    # zx - 真实框
    if map_mode == 0 or map_mode == 2:
        print("Get ground truth result.")
        for image_id in tqdm(image_ids):
            with open(os.path.join(map_out_path, "ground-truth/"+image_id+".txt"), "w") as new_f:
                root = ET.parse(os.path.join(VOCdevkit_path, "VOC2007/Annotations/"+image_id+".xml")).getroot()
                for obj in root.findall('object'):
                    difficult_flag = False
                    if obj.find('difficult')!=None:
                        difficult = obj.find('difficult').text
                        if int(difficult)==1:
                            difficult_flag = True
                    obj_name = obj.find('name').text
                    if obj_name not in class_names:
                        continue
                    bndbox  = obj.find('bndbox')
                    left    = bndbox.find('xmin').text
                    top     = bndbox.find('ymin').text
                    right   = bndbox.find('xmax').text
                    bottom  = bndbox.find('ymax').text

                    if difficult_flag:
                        new_f.write("%s %s %s %s %s difficult\n" % (obj_name, left, top, right, bottom))
                    else:
                        new_f.write("%s %s %s %s %s\n" % (obj_name, left, top, right, bottom))
        print("Get ground truth result done.")

    # zx - 得到 map
    if map_mode == 0 or map_mode == 3:
        print("Get map.")
        # zx
        map_value = get_map(MINOVERLAP, True, score_threhold = score_threhold, path = map_out_path)
        print("Get map done.")

    if map_mode == 4:
        print("Get map.")
        get_coco_map(class_names = class_names, path = map_out_path)
        print("Get map done.")

    # zx
    return round(map_value * 100, 2)



# zx
if __name__ == "__main__":
    import glob, json

    cur_dataset = "pcb"

    if cur_dataset.lower() == "neu":
        classes_path    = "./model_data/voc_neu_classes.txt"
        dataset_path    = "../../../DATASETS/detection/NEU"
        input_shape     = [480, 480]
    elif cur_dataset.lower() == "lwd":
        classes_path    = "model_data/voc_LWD_classes.txt"
        dataset_path = "../../../DATASETS/detection/LargeWoodDefect"
        input_shape     = [800, 320]
    elif cur_dataset.lower() == "pcb":
        classes_path    = "model_data/classes_deeppcb.txt"
        dataset_path = "../../../DATASETS/detection/DeepPCB"
        input_shape     = [480, 480]
    else:
        raise ValueError("没有这个数据集！")

    output_path     = "map_output"
    model_path      = None
    backbone        = "cspz_fpn_attention"

    output_path = os.path.join(output_path, f"20240318_1827_{cur_dataset.upper()}_"+backbone)

    # zx - 批量测试 map
    selected_epochs = [f"{i:03d}" for i in range(5, 250, 5)]
    selected_epochs += [f"{i:03d}" for i in range(250, 301)]
    # 加入测试的 epoch 名字再进行格式处理
    selected_epochs = ['ep' + str(i) + '-loss' for i in selected_epochs]
    selected_epochs.sort()

    # selected_epochs = ["best_epoch_weights"]

    # 权重存放地址
    dir_path = "logs/TR_20240319_01_55_SSD_PCB_cspz_fpn_attention_FEFrom_SZZero_UNE300_SZ16"
    dir_path = os.path.join(dir_path, "*.pth")
    # 获取所有 epoch 权重的路径地址
    all_pth_paths = glob.glob(dir_path)
    # 没有加入测试的权重需要删除，save_path是保存下来、加入测试的权重
    save_path = []

    # key：选择的 epoch     value：权重路径地址
    selected_pth_path = {}
    json_obj = {}

    # 从所有的权重中选出加入测试的权重
    # 加入测试权重的名字
    for sepoch in selected_epochs:
        for pth_path in all_pth_paths:
            if sepoch in pth_path:
                selected_pth_path[sepoch] = pth_path
                save_path.append(pth_path)
                all_pth_paths.remove(pth_path)

    # 对每个选出的权重形成一个对应的文件夹
    for key, value in selected_pth_path.items():
        map_value = mk_map(
            classes_path=classes_path,
            dataset_path=dataset_path,
            output_path=os.path.join(output_path, key),
            model_path=value,
            backbone=backbone,
            input_shape=input_shape
        )
        json_obj[key] = map_value
    json_obj = json.dumps(json_obj, sort_keys=True, indent=2, ensure_ascii=False)
    json_file = open(os.path.join(output_path, "map_progress.json"), 'w')
    json_file.write(json_obj)
    json_file.close()

    # 删除其余权重
    # ！！！ 直接删除不是回收站
    # for pth_path in all_pth_paths:
    #     # 防止出现 BUG
    #     if pth_path not in save_path:
    #         os.remove(pth_path)