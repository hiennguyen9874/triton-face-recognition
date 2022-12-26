# Triton face recogition

### Prerequisites

-   Docker
-   Nvidia-driver
-   Nvidia-docker2

## Export `.engine`

### Face detection model

-   Export checkpoint to onnx model:
    -   Clone [yolov7-face-detection](https://github.com/hiennguyen9874/yolov7-face-detection/tree/using-landmark) and cd into `yolov7-face-detection` folder
    -   Download weight and save into `weights/yolov7-tiny33.pt`
    -   Export to onnx: `python3 export.py --weights ./weights/yolov7-tiny33.pt --img-size 640 --batch-size 1 --dynamic-batch --grid --end2end --max-wh 640 --topk-all 100 --iou-thres 0.5 --conf-thres 0.2 --device 1 --simplify --cleanup --trt`
-   Or download onnx file from from [github.com/hiennguyen9874/yolov7-face-detection/releases/tag/v0.1](https://github.com/hiennguyen9874/yolov7-face-detection/releases/tag/v0.1)
-   Export to TensorRT: `docker-compose run --rm triton /usr/src/tensorrt/bin/trtexec --onnx=/models/FaceDetection/1/yolov7-tiny41-nms-trt.onnx --saveEngine=/models/FaceDetection/1/model.plan --fp16 --minShapes=images:1x3x640x640 --optShapes=images:1x3x640x640 --maxShapes=images:4x3x640x640 --shapes=images:1x3x640x640 --workspace=12288`

### Face recognition model

-   Download `webface_r50.onnx` from [deepinsight/insightface](https://github.com/deepinsight/insightface/blob/master/model_zoo/README.md) and cleaning onnx file: `python3 scripts/onnx_clean.py --onnx-path samples/models/Secondary_Recognition/ms1mv3_r50_pfc.onnx --image-size 112,112 --batch-size 1 --simplify --dynamic --cleanup`
-   Or download onnx file from from: [github.com/hiennguyen9874/deepstream-face-recognition/releases/tag/v0.1](https://github.com/hiennguyen9874/deepstream-face-recognition/releases/tag/v0.1)
-   Export to TensorRT: `docker-compose run --rm triton /usr/src/tensorrt/bin/trtexec --onnx=/models/FaceRecognition/1/webface_r50_dynamic_simplify_cleanup.onnx --saveEngine=/models/FaceRecognition/1/model.plan --fp16 --minShapes=input.1:1x3x112x112 --optShapes=input.1:1x3x112x112 --maxShapes=input.1:16x3x112x112 --shapes=input.1:1x3x112x112 --workspace=12288`

## Run demo

-   [main.ipynb](main.ipynb)
