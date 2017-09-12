import paddle.v2 as paddle
import numpy

paddle.init()


with open('param_1.tar') as f:
    params = paddle.parameters.Parameters.from_tar(f)
with open('inference_topology.pkl', 'rb') as f:
    infer = paddle.inference.Inference(parameters=params, fileobj=f)
test_reader = paddle.dataset.mnist.test()

for data in test_reader():
    print numpy.argmax(infer.infer([[data[0]]]))
