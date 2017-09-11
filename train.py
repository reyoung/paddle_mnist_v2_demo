import paddle.v2 as paddle
import gzip

train_reader = paddle.dataset.mnist.train()
paddle.init(trainer_count=3)
img = paddle.layer.data(name="img", type=paddle.data_type.dense_vector(784))
hidden = paddle.layer.fc(input=img, size=200, act=paddle.activation.Tanh())
hidden = paddle.layer.fc(input=hidden, size=200, act=paddle.activation.Tanh())
prediction = paddle.layer.fc(input=hidden, size=10,
                             act=paddle.activation.Softmax())

# Save the inference topology to protobuf.
inference_topology = paddle.topology.Topology(layers=prediction)
with open("inference.protobin", 'wb') as f:
    proto = inference_topology.proto()
    f.write(proto.SerializeToString())

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
        print 'Start train ' + str(ev.pass_id)
    elif isinstance(ev, paddle.event.EndIteration):
        if (ev.batch_id + 1) % 100 == 0:
            print 'Train Pass %d, Batch %d, Cost = %.2f ' % (
                ev.pass_id, ev.batch_id + 1, ev.cost)
    elif isinstance(ev, paddle.event.EndPass):
        print 'Saving Pass %d to param_%d.tar.gz' % (ev.pass_id, ev.pass_id)
        with gzip.open('param_%d.tar.gz' % ev.pass_id, 'w') as f:
            params.to_tar(f)


trainer.train(
    reader=paddle.batch(paddle.reader.buffered(train_reader, 8192), 100),
    num_passes=100,
    event_handler=train_event_handler)

print 'Training done.'
