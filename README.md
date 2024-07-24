## Introduction

SCRFD is an efficient high accuracy face detection approach which initially described in [Arxiv](https://arxiv.org/abs/2105.04714), and accepted by ICLR-2022.



## Pretrained-Models

|      Name      | Easy  | Medium | Hard  | FLOPs | Params(M) | Infer(ms) | Link                                                         |
| :------------: | ----- | ------ | ----- | ----- | --------- | --------- | ------------------------------------------------------------ |
|   SCRFD_500M   | 90.57 | 88.12  | 68.51 | 500M  | 0.57      | 3.6       | [download](https://1drv.ms/u/s!AswpsDO2toNKqyYWxScdiTITY4TQ?e=DjXof9) |
|    SCRFD_1G    | 92.38 | 90.57  | 74.80 | 1G    | 0.64      | 4.1       | [download](https://1drv.ms/u/s!AswpsDO2toNKqyPVLI44ahNBsOMR?e=esPrBL) |
|   SCRFD_2.5G   | 93.78 | 92.16  | 77.87 | 2.5G  | 0.67      | 4.2       | [download](https://1drv.ms/u/s!AswpsDO2toNKqyTIXnzB1ujPq4th?e=5t1VNv) |
|   SCRFD_10G    | 95.16 | 93.87  | 83.05 | 10G   | 3.86      | 4.9       | [download](https://1drv.ms/u/s!AswpsDO2toNKqyUKwTiwXv2kaa8o?e=umfepO) |
|   SCRFD_34G    | 96.06 | 94.92  | 85.29 | 34G   | 9.80      | 11.7      | [download](https://1drv.ms/u/s!AswpsDO2toNKqyKZwFebVlmlOvzz?e=V2rqUy) |
| SCRFD_500M_KPS | 90.97 | 88.44  | 69.49 | 500M  | 0.57      | 3.6      | [download](https://1drv.ms/u/s!AswpsDO2toNKri_NDM0GIkPpkE2f?e=JkebJo) |
| SCRFD_2.5G_KPS | 93.80 | 92.02  | 77.13 | 2.5G  | 0.82      | 4.3       | [download](https://1drv.ms/u/s!AswpsDO2toNKqyGlhxnCg3smyQqX?e=A6Hufm) |
| SCRFD_10G_KPS  | 95.40 | 94.01  | 82.80 | 10G   | 4.23      | 5.0       | [download](https://1drv.ms/u/s!AswpsDO2toNKqycsF19UbaCWaLWx?e=F6i5Vm) |

mAP, FLOPs and inference latency are all evaluated on VGA resolution.
``_KPS`` means the model includes 5 keypoints prediction.

## Convert to ONNX

Please refer to `tools/scrfd2onnx.py`

Generated onnx model can accept dynamic input as default.

You can also set specific input shape by pass ``--shape 640 640``, then output onnx model can be optimized by onnx-simplifier.

## Convert to torchscript

Please refer to `tools/convert_scrfd_script.py`



## Inference

Please refer to `test_triton.py` which uses onnxruntime to do inference.

