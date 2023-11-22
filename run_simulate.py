import torch
import numpy as np
import cv2
import copy
from rknn.api import RKNN

from plate_recognition.plate_rec import init_model, cv_imread
from utils.post_process import detect_Recognition_plate, detect_Recognition_plate_rknn, draw_result, load_model


# RKNN模型参数设定
IMG_PATH = '/home/batc-jxb/Projects/Rknn/model-convert/plate-detect/figures/1.png'             # 选择测试的图片
ONNX_MODEL = './weights/plate_rec_color-new.onnx'
RKNN_MODEL = './weights/plate_rec_color-new.rknn'
QUANTIZE_ON = False
DATASET = ''

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
detect_model = load_model("./weights/plate_detect.pt", device)  #初始化检测模型
plate_rec_model = init_model(device, "./weights/plate_rec_color-new.pth", is_color = True)      #初始化识别模型


if __name__ == '__main__':

    ### Create RKNN object
    rknn = RKNN(verbose=False)

    ### pre-process config
    print('--> Config model')
    rknn.config(
        # mean_values = [[114.57069762499403, 112.06498372297212, 112.0936524574251]],
        # std_values = [[56.552368489533805, 59.74804638037939, 59.92431607351682]],
        optimization_level = 2,
        target_platform = "rk3588")
    print('done')

    ### Load ONNX model
    print('--> Loading model')
    ret = rknn.load_onnx(model=ONNX_MODEL)
    if ret != 0:
        print('Load model failed!')
        exit(ret)
    print('done')

    ### Build model
    print('--> Building model')
    ret = rknn.build(do_quantization=QUANTIZE_ON, dataset=DATASET)
    if ret != 0:
        print('Build model failed!')
        exit(ret)
    print('done')

    ### Export RKNN model
    print('--> Export rknn model')
    ret = rknn.export_rknn(RKNN_MODEL)
    if ret != 0:
        print('Export rknn model failed!')
        exit(ret)
    print('done')

    ### Init runtime environment
    print('--> Init runtime environment')
    ret = rknn.init_runtime()
    if ret != 0:
        print('Init runtime environment failed!')
        exit(ret)
    print('done')

    ### Inference
    ## (1) 读取识别车牌图片
    img = cv_imread(IMG_PATH)
    if img.shape[-1] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    ## (2) 通过pt文件检测车牌，并识别别车牌
    dict_list_torch = detect_Recognition_plate(detect_model,
                                               img,
                                               device,
                                               plate_rec_model,
                                               img_size = 640,
                                               is_color = True)
    
    dict_list_rknn = detect_Recognition_plate_rknn(detect_model,
                                                   img,
                                                   device,
                                                   rknn,
                                                   img_size = 640,
                                                   is_color = True)

    ## (4) 将结果保存为图片
    img_torch = copy.deepcopy(img)
    print("torch output:    ", end="")
    ori_img_torch = draw_result(img_torch, dict_list_torch)
    cv2.imwrite("./results/result_torch.jpg", ori_img_torch)

    img_rknn = copy.deepcopy(img)
    print("rknn output:    ", end="")
    ori_img_rknn = draw_result(img_rknn, dict_list_rknn)
    cv2.imwrite("./results/result_rknn.jpg", ori_img_rknn)

    ## (5) 释放RKNN对象
    rknn.release()