import matplotlib
matplotlib.use('Agg')

import chainer
import chainer.functions as F
import chainer.links as L
from chainer import datasets
from chainer import training
from chainer.training import extensions

import numpy as np

mushroomsfile = 'mushrooms.csv'

data_array = np.genfromtxt(
    mushroomsfile, delimiter=',', dtype=str, skip_header=1)
for col in range(data_array.shape[1]):
    data_array[:, col] = np.unique(data_array[:, col], return_inverse=True)[1]

X = data_array[:, 1:].astype(np.float32)
Y = data_array[:, 0].astype(np.int32)[:, None]
train, test = datasets.split_dataset_random(
    datasets.TupleDataset(X, Y), int(data_array.shape[0] * .7))

train_iter = chainer.iterators.SerialIterator(train, 100)
test_iter = chainer.iterators.SerialIterator(
    test, 100, repeat=False, shuffle=False)

gpu_id = 0

# Network definition
#class MLP(chainer.Chain):
#    def __init__(self, n_units, n_out):
#        super(MLP, self).__init__()
#        with self.init_scope():
#            # the input size to each layer inferred from the layer before
#            self.l1 = L.Linear(n_units)  # n_in -> n_units
#            self.l2 = L.Linear(n_units)  # n_units -> n_units
#            self.l3 = L.Linear(n_out)  # n_units -> n_out
#
#    def __call__(self, x):
#        h1 = F.relu(self.l1(x))
#        h2 = F.relu(self.l2(h1))
#        return self.l3(h2)

# I think it sadly cannot be done this way
#class MLP(chainer.Chain):
#    def __init__(self, n_units, n_out, n_layers=1):
#        super(MLP, self).__init__()
#        self.n_layers = n_layers
#        with self.init_scope():
#            # the input size to each layer inferred from the layer before
#            idx=1
#            for _ in range(n_layers+1):
#                getattr(self,'l%d' % (idx,)) = L.Linear(n_units)
#                idx += 1
#            eval('self.l%d = L.Linear(n_out)' % (idx,))
#            idx += 1
#
#    def __call__(self, x):
#        for i in range(self.n_layers+1):
#            eval('l = self.l%d' % (i+1,))
#            x = F.relu(l(x))
#        eval('x = self.l%d(x)' % (self.n_layers+2,))
#        return x

if gpu_id >= 0:
    chainer.backends.cuda.get_device_from_id(gpu_id).use()

model = L.Classifier(
    MLP(44, 1), lossfun=F.sigmoid_cross_entropy, accfun=F.binary_accuracy)

# Setup an optimizer
optimizer = chainer.optimizers.SGD()
optimizer.setup(model)

# Create the updater, using the optimizer
updater = training.StandardUpdater(train_iter, optimizer, device=gpu_id)

# Set up a trainer
trainer = training.Trainer(updater, (50, 'epoch'), out='result')

# Evaluate the model with the test dataset for each epoch
trainer.extend(extensions.Evaluator(test_iter, model, device=gpu_id))

# Dump a computational graph from 'loss' variable at the first iteration
# The "main" refers to the target link of the "main" optimizer.
trainer.extend(extensions.dump_graph('main/loss'))

trainer.extend(extensions.snapshot(), trigger=(20, 'epoch'))

# Write a log of evaluation statistics for each epoch
trainer.extend(extensions.LogReport())

# Save two plot images to the result dir
if extensions.PlotReport.available():
    trainer.extend(
        extensions.PlotReport(['main/loss', 'validation/main/loss'],
                              'epoch', file_name='loss.png'))
    trainer.extend(
        extensions.PlotReport(
            ['main/accuracy', 'validation/main/accuracy'],
            'epoch', file_name='accuracy.png'))

# Print selected entries of the log to stdout
trainer.extend(extensions.PrintReport(
    ['epoch', 'main/loss', 'validation/main/loss',
     'main/accuracy', 'validation/main/accuracy', 'elapsed_time']))

#  Run the training
trainer.run()

x, t = test[np.random.randint(len(test))]

# inference doesn't work yet ...
#predict = model.predictor(x[None]).data
#predict = predict[0][0]
#
#if predict >= 0:
#    print('Predicted Poisonous, Actual ' + ['Edible', 'Poisonous'][t[0]])
#else:
#    print('Predicted Edible, Actual ' + ['Edible', 'Poisonous'][t[0]])
#
