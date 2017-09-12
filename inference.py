import paddle.v2 as paddle
import numpy

paddle.init()
with open('inference.protobin', 'rb') as f:
    proto = f.read()

with open('param_1.tar') as f:
    params = paddle.parameters.Parameters.from_tar(f)

infer = paddle.inference.Inference(output_layer=proto, parameters=params,
                                   data_types=[
                                       ('img', paddle.data_type.dense_vector(784))
                                   ])
test_reader = paddle.dataset.mnist.test()

for data in test_reader():
    print numpy.argmax(infer.infer([[data[0]]]))
