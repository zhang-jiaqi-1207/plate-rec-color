import torch
import os
from torch import nn
from plate_recognition.plateNet import myNet_ocr_color2


### File path setting 
PTH_PATH = "./weights/plate_rec_color-new.pth"          ## 车牌识别的pth模型位置
ONNX_PATH = "./weights/plate_rec_color-new.onnx"        ## 车牌识别的onnx模型保存位置


### Parameters setting where the `init_model` needs.
device = torch.device('cuda') if torch.cuda.is_available() else torch.device("cpu")
color = ['黑色','蓝色','绿色','白色','黄色']    
plateName = \
    r"#京沪津渝冀晋蒙辽吉黑苏浙皖闽赣鲁豫鄂湘粤桂琼川贵云藏陕甘青宁新学警港澳挂使领民航危0123456789ABCDEFGHJKLMNPQRSTUVWXYZ险品"
mean_value, std_value = (0.588, 0.193)


def init_model(device,model_path,is_color = False): 
    check_point = torch.load(model_path,map_location=device)
    model_state = check_point['state_dict']
    cfg = check_point['cfg']

    color_classes = 0
    if is_color:
        color_classes = 5           #颜色类别数

    model = myNet_ocr_color2(
        num_classes = len(plateName),
        export = True,
        cfg = cfg,
        color_num = color_classes)   
    model.load_state_dict(model_state, strict=False)
    model.to(device)
    model.eval()
    return model

if __name__ == "__main__":
    print("-"*15 + ">:" +"start!")

    model = init_model(device, PTH_PATH, is_color=True)
    print("-"*15 + ">:" + "have got the model! ")

    dummy_input = torch.randn(1, 3, 48, 168, device=device)

    torch.onnx.export(model, 
                      dummy_input,
                      ONNX_PATH,
                      verbose = False,
                      input_names = ['input'],
                      output_names = ['output'] 
                      )
    print("-"*15 + ">:" +"done!")
