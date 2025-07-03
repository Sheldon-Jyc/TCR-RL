import torch
import numpy as np
from torch import nn

class ssm_LTCCell(nn.Module):
    def __init__(
            self,
            config,
            wiring,
            in_features=None,
            input_mapping="affine",
            output_mapping="affine",
            ode_unfolds=6,
            epsilon=1e-8,implicit_param_constraints=False,
            **kwargs,
    ):
        super(ssm_LTCCell, self).__init__()
        if in_features is not None:
            wiring.build(config.intermediate_size)
        if not wiring.is_built():
            raise ValueError(
                "Wiring error! Unknown number of input features. Please pass the parameter 'in_features' or call the 'wiring.build()'."
            )
        self.make_positive_fn = (
            nn.Softplus() if implicit_param_constraints else nn.Identity()
        )
        self._implicit_param_constraints = implicit_param_constraints
        self._init_ranges = {
            "gleak": (0.001, 1.0),
            "vleak": (-0.2, 0.2),
            "cm": (0.4, 0.6),
            "w": (0.001, 1.0),
            "sigma": (3, 8),
            "mu": (0.3, 0.8),
            "sensory_w": (0.001, 1.0),
            "sensory_sigma": (3, 8),
            "sensory_mu": (0.3, 0.8),
        }
        self._wiring = wiring
        self._input_mapping = input_mapping
        self._output_mapping = output_mapping
        self._ode_unfolds = ode_unfolds
        self._epsilon = epsilon
        self._clip = torch.nn.ReLU()
        self.use_motorcell = config.use_motorcell
        self.inter_size = config.intermediate_size
        self.out_seq = config.out_seq
        self._allocate_parameters()
        if config.activation == 'softmax':
            self.act = nn.Softmax(dim=-1)
        elif config.activation == 'relu':
            self.act = nn.ReLU()
        elif config.activation == 'silu':
            self.act = nn.SiLU()
        else:
            self.act = nn.Tanh()

        self.in_proj2 = nn.Linear(in_features, self.inter_size + self.state_size)
        self.c_proj = nn.Linear(self.inter_size, self.state_size )
        self.use_bias = config.use_bias
        if self.use_motorcell == 'out_proj':
            self.out_proj = nn.Linear(self.state_size, self.motor_size, bias=self.use_bias)
        self.out_proj = nn.Linear(self.state_size, self.motor_size, bias=self.use_bias) #meiyong

    def forward(self, inputs, ode_state, elapsed_time=1.0):
        batch_size, seq_len, _ = inputs.shape
        proj_state = self.act(self.in_proj2(inputs))
        hidden_state, gate = torch.split(
            proj_state, [self.inter_size, self.state_size], dim=-1
        )
        gate = torch.softmax(gate, dim=1)
        C = self.c_proj(hidden_state)
        fused_out_seq = []
        for t in range(seq_len):
            h_t = hidden_state[:, t, :]
            C_t = C[:, t, :]
            g_t = gate[:, t, :]
            ode_state = self._ode_solver(h_t, ode_state, elapsed_time)
            fused = ode_state * C_t
            fused = fused * torch.sigmoid(g_t)
            fused_out_seq.append(fused)
        fused_out_seq = torch.stack(fused_out_seq, dim=1)
        if self.out_seq is False:
            fused_out = fused_out_seq.mean(dim=1)
        else:
            fused_out = fused_out_seq

        if self.use_motorcell=='use_map':
            outputs = self._map_outputs(fused_out)
        elif self.use_motorcell == 'out_proj':
            outputs = self.out_proj(fused_out)
        else:
            outputs = fused_out
        return outputs , ode_state, fused_out

    def add_weight(self, name, init_value, requires_grad=True):
        param = torch.nn.Parameter(init_value, requires_grad=requires_grad)
        self.register_parameter(name, param)
        return param

    def _get_init_value(self, shape, param_name):
        minval, maxval = self._init_ranges[param_name]
        if minval == maxval:
            return torch.ones(shape) * minval
        else:
            return torch.rand(*shape) * (maxval - minval) + minval

    def _allocate_parameters(self):
        self._params = {}
        self._params["gleak"] = self.add_weight(
            name="gleak", init_value=self._get_init_value((self.state_size,), "gleak")
        )
        self._params["vleak"] = self.add_weight(
            name="vleak", init_value=self._get_init_value((self.state_size,), "vleak")
        )
        self._params["cm"] = self.add_weight(
            name="cm", init_value=self._get_init_value((self.state_size,), "cm")
        )
        self._params["sigma"] = self.add_weight(
            name="sigma",
            init_value=self._get_init_value(
                (self.state_size, self.state_size), "sigma"
            ),
        )
        self._params["mu"] = self.add_weight(
            name="mu",
            init_value=self._get_init_value((self.state_size, self.state_size), "mu"),
        )
        self._params["w"] = self.add_weight(
            name="w",
            init_value=self._get_init_value((self.state_size, self.state_size), "w"),
        )
        self._params["erev"] = self.add_weight(
            name="erev",
            init_value=torch.Tensor(self._wiring.erev_initializer()),
        )
        self._params["sensory_sigma"] = self.add_weight(
            name="sensory_sigma",
            init_value=self._get_init_value(
                (self.sensory_size, self.state_size), "sensory_sigma"
            ),
        )
        self._params["sensory_mu"] = self.add_weight(
            name="sensory_mu",
            init_value=self._get_init_value(
                (self.sensory_size, self.state_size), "sensory_mu"
            ),
        )
        self._params["sensory_w"] = self.add_weight(
            name="sensory_w",
            init_value=self._get_init_value(
                (self.sensory_size, self.state_size), "sensory_w"
            ),
        )
        self._params["sensory_erev"] = self.add_weight(
            name="sensory_erev",
            init_value=torch.Tensor(self._wiring.sensory_erev_initializer()),
        )

        self._params["sparsity_mask"] = self.add_weight(
            "sparsity_mask",
            torch.Tensor(np.abs(self._wiring.adjacency_matrix)),
            requires_grad=False,
        )
        self._params["sensory_sparsity_mask"] = self.add_weight(
            "sensory_sparsity_mask",
            torch.Tensor(np.abs(self._wiring.sensory_adjacency_matrix)),
            requires_grad=False,
        )

        if self._input_mapping in ["affine", "linear"]:
            self._params["input_w"] = self.add_weight(
                name="input_w",
                init_value=torch.ones((self.sensory_size,)),
            )
        if self._input_mapping == "affine":
            self._params["input_b"] = self.add_weight(
                name="input_b",
                init_value=torch.zeros((self.sensory_size,)),
            )

        if self._output_mapping in ["affine", "linear"]:
            self._params["output_w"] = self.add_weight(
                name="output_w",
                init_value=torch.ones((self.motor_size,)),
            )
        if self._output_mapping == "affine":
            self._params["output_b"] = self.add_weight(
                name="output_b",
                init_value=torch.zeros((self.motor_size,)),
            )

    def _sigmoid(self, v_pre, mu, sigma):
        if len(v_pre.shape) == 2:
            v_pre = torch.unsqueeze(v_pre, -1)  # For broadcasting
        mues = v_pre - mu
        x = sigma * mues
        return torch.sigmoid(x)

    def _ode_solver(self, inputs, state, elapsed_time=1.0):
        v_pre = state
        sensory_w_activation = self.make_positive_fn(self._params["sensory_w"]) * self._sigmoid(
            inputs, self._params["sensory_mu"], self._params["sensory_sigma"]
        )
        sensory_w_activation = (
                sensory_w_activation * self._params["sensory_sparsity_mask"]
        )
        sensory_rev_activation = sensory_w_activation * self._params["sensory_erev"]
        w_numerator_sensory = torch.sum(sensory_rev_activation, dim=1)
        w_denominator_sensory = torch.sum(sensory_w_activation, dim=1)
        cm_t = self.make_positive_fn(self._params["cm"]) / (
                elapsed_time / self._ode_unfolds
        )
        w_param = self.make_positive_fn(self._params["w"])
        for t in range(self._ode_unfolds):
            w_activation = w_param * self._sigmoid(
                v_pre, self._params["mu"], self._params["sigma"]
            )
            w_activation = w_activation * self._params["sparsity_mask"]
            rev_activation = w_activation * self._params["erev"]
            w_numerator = torch.sum(rev_activation, dim=1) + w_numerator_sensory
            w_denominator = torch.sum(w_activation, dim=1) + w_denominator_sensory
            gleak = self.make_positive_fn(self._params["gleak"])
            numerator = cm_t * v_pre + gleak * self._params["vleak"] + w_numerator
            denominator = cm_t + gleak + w_denominator
            v_pre = numerator / (denominator + self._epsilon)
        return v_pre

    def _map_outputs(self, state):
        output = state
        if self.motor_size < self.state_size:
            output = output[:, 0: self.motor_size]  # slice
        if self._output_mapping in ["affine", "linear"]:
            output = output * self._params["output_w"]
        if self._output_mapping == "affine":
            output = output + self._params["output_b"]
        return output

    def apply_weight_constraints(self):
        if not self._implicit_param_constraints:
            self._params["w"].data = self._clip(self._params["w"].data)
            self._params["sensory_w"].data = self._clip(self._params["sensory_w"].data)
            self._params["cm"].data = self._clip(self._params["cm"].data)
            self._params["gleak"].data = self._clip(self._params["gleak"].data)

class ssm_LTC(nn.Module):
    def __init__(
        self,
        config,
        input_size: int,
        units, # wiring or int
        return_sequences: bool = True,
        batch_first: bool = True,
        input_mapping="affine",
        output_mapping="affine",
        ode_unfolds=6,
        epsilon=1e-8,
        implicit_param_constraints=True,
    ):
        super(ssm_LTC, self).__init__()
        self.input_size = input_size
        self.wiring_or_units = units
        self.batch_first = batch_first
        self.return_sequences = return_sequences
        wiring = units

        self.rnn_cell = ssm_LTCCell(
            config=config,
            wiring=wiring,
            in_features=input_size,
            input_mapping=input_mapping,
            output_mapping=output_mapping,
            ode_unfolds=ode_unfolds,
            epsilon=epsilon,
            implicit_param_constraints=implicit_param_constraints,
        )
        self._wiring = wiring

    def forward(self, input, hx=None,):
        device = input.device
        is_batched = input.dim() == 3
        batch_dim = 0 if self.batch_first else 1
        seq_dim = 1 if self.batch_first else 0
        if not is_batched:
            input = input.unsqueeze(batch_dim)
        batch_size, seq_len = input.size(batch_dim), input.size(seq_dim)
        if hx is None:
            h_state = torch.zeros((batch_size, self.state_size), device=device)
        else:
            h_state = hx
            if is_batched:
                if h_state.dim() != 2:
                    msg = (
                        "For batched 2-D input, hx and cx should " f"also be 2-D but got ({h_state.dim()}-D) tensor"
                    )
                    raise RuntimeError(msg)
            else:
                if h_state.dim() != 1:
                    msg = (
                        "For unbatched 1-D input, hx and cx should "
                        f"also be 1-D but got ({h_state.dim()}-D) tensor"
                    )
                    raise RuntimeError(msg)
                h_state = h_state.unsqueeze(0)
        h_out, h_state, out_state = self.rnn_cell.forward(input, h_state,)

        return h_out, h_state, out_state

