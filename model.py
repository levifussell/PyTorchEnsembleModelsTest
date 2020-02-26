import torch as th
import torch.nn as nn
import torch.functional as F

class Model(nn.Module):

    def __init__(self, input_size, output_size, hid_size, num_layers):
        super().__init__()

        assert num_layers > 0

        layers = []
        layers.append(nn.Linear(input_size, hid_size))
        layers.append(nn.ReLU())
        for i in range(num_layers-1):
            layers.append(nn.Linear(hid_size, hid_size))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hid_size, output_size))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        y = self.model(x)
        return y

class ModelEnsemble(nn.Module):

    def __init__(self, input_size, output_size, hid_size, num_layers, n_parallel):
        super().__init__()

        self.ensemble = nn.ModuleList()
        self.n_parallel = n_parallel
        for _ in range(n_parallel):
            self.ensemble.append(
                    Model(input_size, output_size, hid_size, num_layers)
                    )

    def forward(self, x):
        x_p = x.repeat(1, self.n_parallel)
        y_p = []
        for e in self.ensemble:
            y = e(x)
            y_p.append(y)
        y_p = th.cat(y_p, 1)
        result = {
                "output-parallel"   : y_p,
                "input-parallel"    : x_p,
                }
        return result

class ModelBlockDiag(nn.Module):

    def __init__(self, input_size, output_size, hid_size, num_layers, n_parallel):
        super().__init__()

        assert num_layers > 0

        self.n_parallel = n_parallel
        p_input_size = n_parallel * input_size
        p_hid_size = n_parallel * hid_size
        p_output_size = n_parallel * output_size

        layers = []
        layers.append(nn.Linear(p_input_size, p_hid_size))
        layers.append(nn.ReLU())
        for i in range(num_layers-1):
            layers.append(nn.Linear(p_hid_size, p_hid_size))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(p_hid_size, p_output_size))

        self.model = nn.Sequential(*layers)

        # block diagonal all the weights.
        count = 0
        for idx,l in enumerate(self.model):
            if str(l.__class__.__name__) == "Linear":
                in_size = input_size if idx == 0 else hid_size
                out_size = output_size if idx == len(self.model)-1 else hid_size
                weight_mask = th.zeros(l.out_features, l.in_features)
                #bias_mask = th.zeros_like(l.bias)
                #bias_mask[count*out_size:(count+1)*out_size] = 1.0
                for i in range(n_parallel):
                    weight_mask[i*out_size:(i+1)*out_size, i*in_size:(i+1)*in_size] = 1.0

                l.weight.data.mul_(weight_mask)
                #l.bias.data.mul_(bias_mask)
                #count += 1

    def forward(self, x):
        x_p = x.repeat(1, self.n_parallel)
        y_p = self.model(x_p)
        result = {
                "output-parallel"   : y_p,
                "input-parallel"    : x_p,
                }
        return result
