# based on mush_predict but can have variable hidden layer widths and depths
import matplotlib
import random
import cupy as cp
matplotlib.use('Agg')

import chainer
import chainer.functions as F
import chainer.links as L
from chainer import datasets
from chainer import training
from chainer.training import extensions

import numpy as np

n_units=44
n_hidden=4

mushroomsfile = 'mushrooms.csv'

data_array = np.genfromtxt(
    mushroomsfile, delimiter=',', dtype=str, skip_header=1)
for col in range(data_array.shape[1]):
    data_array[:, col] = np.unique(data_array[:, col], return_inverse=True)[1]

X = data_array[:, 1:].astype(np.float32)
Y = data_array[:, 0].astype(np.int32)[:, None]
#train, test = datasets.split_dataset_random(
#    datasets.TupleDataset(X, Y), int(data_array.shape[0] * .7))

class TupleDatasetGPU(chainer.datasets.TupleDataset):
    def set_device(self,device=0):
        self.device = device
        return self
    def __getitem__(self,index):
        batch = chainer.datasets.TupleDataset.__getitem__(self,index)
        with cp.cuda.Device(self.device):
            if isinstance(batch,list):
                ret = [tuple([cp.asarray(b) for b in batch_]) for batch_ in
                        batch]
            else:
                ret = tuple([cp.asarray(b) for b in batch])
        return ret

gpu_id=0

# sort data randomly
rand_idcs = np.arange(data_array.shape[0])
random.shuffle(rand_idcs)
train_idcs = rand_idcs[:int(data_array.shape[0] * .7)]
test_idcs = rand_idcs[int(data_array.shape[0] * .7):]
test = TupleDatasetGPU(X[test_idcs],Y[test_idcs]).set_device(gpu_id)
train = TupleDatasetGPU(X[train_idcs],Y[train_idcs]).set_device(gpu_id)

class SerialIteratorGPU(chainer.iterators.SerialIterator):
    def set_device(self,device=0):
        self.device=device
    def __next__(self):
        batch = chainer.iterators.SerialIterator.__next__(self)
        with cp.cuda.Device(self.device):
            return cp.asarray(batch)

train_iter = chainer.iterators.SerialIterator(train, 100)
test_iter = chainer.iterators.SerialIterator(
    test, 100, repeat=False, shuffle=False)


## Network definition
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

# Network definition
class MLP(chainer.Chain):
    def __init__(self, n_in, n_units, n_out, n_hidden,device=None):
        super(MLP, self).__init__()
        # TODO: See why this works (https://www.python.org/dev/peps/pep-0343)
        with self.init_scope():
            self.layers=[]
            self.layers.append(L.Linear(n_in,n_units))
            if device:
                self.layers[-1].to_gpu(device)
            for n in range(n_hidden):
                l = L.Linear(n_units,n_units)
                if device:
                    l.to_gpu(device)
                self.add_link('l%d' % (n,), l)
                self.layers.append(l)
            self.layers.append(L.Linear(n_units,n_out))
            if device:
                self.layers[-1].to_gpu(device)
            # the input size to each layer inferred from the layer before
            #self.l1 = L.Linear(n_units)  # n_in -> n_units
            #self.l2 = L.Linear(n_units)  # n_units -> n_units
            #self.l3 = L.Linear(n_out)  # n_units -> n_out

    def __call__(self, x):
        for l in self.layers[:-1]:
            x = F.relu(l(x))
        return self.layers[-1](x)


model = L.Classifier(
    #MLP(44, 1),
    MLP(X.shape[1],n_units, 1,n_hidden),
    lossfun=F.sigmoid_cross_entropy, accfun=F.binary_accuracy)

if gpu_id >= 0:
    model.to_gpu(gpu_id)    

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

predict = model.predictor(x[None]).data
predict = predict[0][0]

if predict >= 0:
    print('Predicted Poisonous, Actual ' + ['Edible', 'Poisonous'][t[0]])
else:
    print('Predicted Edible, Actual ' + ['Edible', 'Poisonous'][t[0]])

