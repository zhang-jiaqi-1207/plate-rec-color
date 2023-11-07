## 项目简介

该项目为车牌检测识别任务中的识别部分，其上游项目为[车牌检测](https://github.com/zhang-jiaqi-1207/plate-detect-pt2rk)项目。二者的模型原型均来自于[yolov5 车牌检测 车牌识别 中文车牌识别 检测 支持12种中文车牌 支持双层车牌](https://github.com/we0091234/Chinese_license_plate_detection_recognition)。

## 项目目录介绍

整体项目布局类似于[车牌检测](https://github.com/zhang-jiaqi-1207/plate-detect-pt2rk)，同样有些结果文件夹并未给出。

## 运行项目
**假设你现在已经在项目路径之下**，即`pth2onnx.py`文件所在的目录。

1. 将`.pth`文件转换为`.onnx`文件

说明：
- 由于`rknn-toolkit2`无法直接加载`.pth`文件，所以需要将其转为`.onnx`文件。
- 可以通过`PTH_PATH`以及`ONNX_PATH`修改文件的加载与保存路径。

```shell
python3 pth2onnx.py
```

2. 构建`RKNN`模型并导出为`.rknn`文件，保存在硬件磁盘上。

说明：
- 可以通过`IMG_PATH`、`ONNX_MODEL`以及`RKNN_MODEL`指定文件的加载与保存路径。
- 可以通过`QUANTIZE_ON`控制是否对模型进行量化。
- 可以通过注释(解除注释)下列代码来决定是否导出rknn模型
    ```python
    ### Export RKNN model
    print('--> Export rknn model')
    ret = rknn.export_rknn(RKNN_MODEL)
    if ret != 0:
        print('Export rknn model failed!')
        exit(ret)
    print('done')
    ```

```shell
python3 run_simulate.py
```