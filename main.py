import torch as th
import time
import numpy as np

import argparse

from model import ModelBlockDiag, ModelEnsemble

parser = argparse.ArgumentParser()
parser.add_argument(
        '--hid-size',
        type=int,
        default=16,
        )
parser.add_argument(
        '--num-layers',
        type=int,
        default=3,
        )
parser.add_argument(
        '--num-parallel',
        type=int,
        default=10,
        )
parser.add_argument(
        '--model-type',
        type=str,
        default="ensemble",
        )
args = parser.parse_args()

device = th.device("cpu")

data = th.rand(50000, 12)

batch_size = 256

data_size = data.size()[0]
input_size = data.size()[1]
num_samples = data_size // batch_size + 1

if args.model_type == "ensemble":
    m = ModelEnsemble(
            input_size=input_size,
            output_size=input_size,
            hid_size=args.hid_size,
            num_layers=args.num_layers,
            n_parallel=args.num_parallel,
            )
elif args.model_type == "diag":
    m = ModelBlockDiag(
            input_size=input_size,
            output_size=input_size,
            hid_size=args.hid_size,
            num_layers=args.num_layers,
            n_parallel=args.num_parallel,
            )
print(m)
m.to(device)

opt = th.optim.Adam(m.parameters(), lr=0.0001)
loss_function = th.nn.L1Loss()

epochs = 10

t_t_start = time.time()
for e in range(epochs):
    t_e_start = time.time()
    for i in range(num_samples):
        batch_idxs = np.random.permutation(data_size)[:batch_size]
        batch_data = data[batch_idxs, :].to(device)

        result = m(batch_data)
        p_output = result["output-parallel"]
        p_input = result["input-parallel"]

        loss = loss_function(p_output, p_input)
        m.zero_grad()
        loss.backward()
        opt.step()
    t_e_diff = time.time() - t_e_start
    print('EPOCH TIME: {}'.format(t_e_diff))

t_t_diff = time.time() - t_t_start
print('TOTAL TIME: {}'.format(t_t_diff))
print('\t ...per epoch: {}'.format(t_t_diff/epochs))
