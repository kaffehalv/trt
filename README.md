# trt

## Fixed batch model

To convert a model to fixed batch, run something similar to this

```shell
uv init
uv add onnx onnxruntime
uv run python -m onnxruntime.tools.make_dynamic_shape_fixed resnet18-v1-7.onnx resnet18-fixed.onnx --input_name data --input_shape 1,3,224,224
```

## Troubleshooting

```shell
export LD_LIBRARY_PATH=/usr/local/tensorrt-cuda12.2-10.4.0.26/lib:$LD_LIBRARY_PATH
```
