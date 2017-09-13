import paddle.v2 as paddle
import cPickle

train_reader = paddle.dataset.mnist.train()
test_reader = paddle.dataset.mnist.test()

def convolutional_neural_network(img):
    # first conv layer
    conv_pool_1 = paddle.networks.simple_img_conv_pool(
        input=img,
        filter_size=5,
        num_filters=20,
        num_channel=1,
        pool_size=2,
        pool_stride=2,
        act=paddle.activation.Relu())
    # second conv layer
    conv_pool_2 = paddle.networks.simple_img_conv_pool(
        input=conv_pool_1,
        filter_size=5,
        num_filters=50,
        num_channel=20,
        pool_size=2,
        pool_stride=2,
        act=paddle.activation.Relu())
    # fully-connected layer
    predict = paddle.layer.fc(
        input=conv_pool_2, size=10, act=paddle.activation.Softmax())
    return predict

paddle.init(use_gpu=False,trainer_count=1)
img = paddle.layer.data(name="img", type=paddle.data_type.dense_vector(784))
prediction = convolutional_neural_network(img)

# Save the inference topology to protobuf.
inference_topology = paddle.topology.Topology(layers=prediction)
with open("inference_topology.pkl", 'wb') as f:
    inference_topology.serialize_for_inference(f)

cost = paddle.layer.classification_cost(input=prediction,
                                        label=paddle.layer.data(name="lbl",
                                                                type=paddle.data_type.integer_value(
                                                                    10)))
params = paddle.parameters.create(cost)
trainer = paddle.trainer.SGD(cost=cost, parameters=params,
                             update_equation=paddle.optimizer.Adam(
                                 learning_rate=1e-3))


def train_event_handler(ev):
    if isinstance(ev, paddle.event.BeginPass):
        print 'Start train pass ' + str(ev.pass_id)
    elif isinstance(ev, paddle.event.EndIteration):
        if (ev.batch_id + 1) % 100 == 0:
            print 'Train Pass %d, Batch %d, Cost = %.2f %s' % (
                ev.pass_id, ev.batch_id + 1, ev.cost, ev.metrics)
    elif isinstance(ev, paddle.event.EndPass):
        result = trainer.test(reader=paddle.batch(test_reader, 100))
        print 'Saving Pass %d to param_%d.tar, test %s' % (ev.pass_id, ev.pass_id, result.metrics)
        with open('param_%d.tar' % ev.pass_id, 'w') as f:
            params.to_tar(f)

trainer.train(
    reader=paddle.batch(paddle.reader.shuffle(train_reader, 81920), 100),
    num_passes=10000,
    event_handler=train_event_handler)

print 'Training done.'
