# This works I guess but doesn't learn XOR super well. TODO: Why?
import chainer
from chainer import datasets, functions, links, training
from chainer.training import extensions
import itertools
import numpy as np

# number of examples of each case
num_examples = 1000
# number of units in each hidden layer
n_units=2
# number of hidden layers
n_hidden=1

# Generate examples
def xor_example_gen(num_examples=8):
    full_ds=[]
    for x in itertools.product([0,1],[0,1]):
        full_ds += itertools.repeat(x,num_examples)

    ds_x=np.array(full_ds).astype(np.int32)
    ds_y=np.array(ds_x[:,0] ^ ds_x[:,1]).astype(np.int32)[:,None]
    #ds_x = np.random.uniform(size=(num_examples*4,2))
    #ds_y=np.random.randint(2,size=(num_examples*4,1)).astype(np.int32)
    return (ds_x.astype(np.float32),ds_y)

X, Y = xor_example_gen(num_examples)

# 70% used for training
train, test = datasets.split_dataset_random(
    datasets.TupleDataset(X, Y), int(X.shape[0] * .7))

train_iter = chainer.iterators.SerialIterator(train, min(100,len(train)))
test_iter = chainer.iterators.SerialIterator( test, min(100,len(test)), repeat=False, shuffle=False)

## Network definition
#class MLP(chainer.Chain):
#    def __init__(self, n_in, n_units, n_out, n_hidden):
#        super(MLP, self).__init__()
#        # TODO: See why this works (https://www.python.org/dev/peps/pep-0343)
#        with self.init_scope():
#            self.layers=[]
#            self.layers.append(links.Linear(n_in,n_units))
#            for n in range(n_hidden):
#                self.layers.append(links.Linear(n_units,n_units))
#            self.layers.append(links.Linear(n_units,n_out))
#            # the input size to each layer inferred from the layer before
#            #self.l1 = links.Linear(n_units)  # n_in -> n_units
#            #self.l2 = links.Linear(n_units)  # n_units -> n_units
#            #self.l3 = links.Linear(n_out)  # n_units -> n_out
#
#    def __call__(self, x):
#        for l in self.layers[:-1]:
#            x = functions.relu(l(x))
#        return self.layers[-1](x)

import chainer
import chainer.functions as F
import chainer.links as L

class MLP(chainer.Chain):

    def __init__(self, n_in, n_hidden, n_out):
        super(MLP, self).__init__()
        with self.init_scope():
            self.layer1 = L.Linear(n_in, n_hidden)
            self.layer2 = L.Linear(n_hidden, n_hidden)
            self.layer3 = L.Linear(n_hidden, n_out)

    def __call__(self, x):
        # Forward propagation
        h1 = F.relu(self.layer1(x))
        h2 = F.relu(self.layer2(h1))
        #h1 = F.tanh(self.layer1(x))
        #h2 = F.tanh(self.layer2(h1))

        return self.layer3(h2)


# model instance
model = links.Classifier(
        #MLP(X.shape[1],n_units,1,n_hidden),
        MLP(X.shape[1],n_units,Y.shape[1]),
        lossfun=functions.sigmoid_cross_entropy, accfun=functions.binary_accuracy)

# Setup an optimizer
optimizer = chainer.optimizers.SGD(lr=1e-3)
optimizer.setup(model)

# Create the updater, using the optimizer
updater = training.StandardUpdater(train_iter, optimizer, device=-1)

# Set up a trainer
trainer = training.Trainer(updater, (1000, 'epoch'), out='xor_training_result')

# Evaluate the model with the test dataset for each epoch
trainer.extend(extensions.Evaluator(test_iter, model, device=-1))

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

