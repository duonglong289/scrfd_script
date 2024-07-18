
http_port=11000
grpc_port=12001
metrics_port=11002

docker run --gpus="device=2" -p $http_port:$http_port -p $grpc_port:$grpc_port -p $metrics_port:$metrics_port --net=host \
            -v $PWD/models:/models nvcr.io/nvidia/tritonserver:23.02-py3 tritonserver \
            --model-repository=/models --strict-model-config=false --exit-on-error=false \
            --http-port $http_port --grpc-port $grpc_port --metrics-port $metrics_port

# docker run --gpus="device=0" -p $grpc_port:$grpc_port --net=host \
#             -v $PWD/models:/models nvcr.io/nvidia/tritonserver:23.02-py3 tritonserver \
#             --model-repository=/models --strict-model-config=false --exit-on-error=false \
#             --grpc-port $grpc_port


# docker run --gpus="device=2" -p $http_port:$http_port -p $grpc_port:$grpc_port -p $metrics_port:$metrics_port --net=host -v $PWD/models:/models nvcr.io/nvidia/tritonserver:23.02-py3 tritonserver --model-repository=/models --strict-model-config=false --exit-on-error=false --http-port $http_port --grpc-port $grpc_port --metrics-port $metrics_port
