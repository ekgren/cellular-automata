import os
import sys

import torch
from torch import nn
import torch.nn.functional as F

from ca.runner import Runner
from ca.utils import set_seed, setup_logging, CfgNode as CN


def unfold(input: torch.Tensor, kernel_size: int = 3, pad: int = 1, stride: int = 1) -> torch.Tensor:
    """ Unfold a 2d tensor into patches of size (kernel_size * kernel_size) with a stride of stride. """
    B, C, H, W = input.shape
    input = F.pad(input, (pad, pad, pad, pad), "circular")
    input = input.unfold(2, kernel_size, stride)
    input = input.unfold(3, kernel_size, stride)
    input = input.reshape(B, C, H, W, kernel_size * kernel_size)
    return input


class NeuronalCA:
    """ Neuronal Cellular Automata model. """

    @staticmethod
    def get_default_config():
        C = CN()
        C.model_type = 'NeuronalCA'
        return C

    def __init__(self, config: CN) -> None:
        # Parameters
        self.device = config.device
        self.stdp = config.stdp
        self.kernel_size = config.kernel_size
        self.pad = config.pad
        self.threshold = config.threshold
        self.decay = config.decay
        self.activity_delta = config.activity_delta
        self.drop_p = config.drop_p

        # Initialize model
        dtype = torch.int32
        self.activations = torch.zeros([1, 1, config.board_size, config.board_size], dtype=dtype, device=self.device)
        self.integrations = torch.zeros([1, 1, config.board_size, config.board_size], dtype=dtype, device=self.device)

        self.kernel = torch.ones(self.kernel_size ** 2, dtype=dtype, device=self.device)
        self.kernel[(self.kernel_size ** 2) // 2] = 0

        connectome_shape = (1, 1, config.board_size, config.board_size, self.kernel_size ** 2)
        connectome_init = (torch.rand(connectome_shape, device=self.device) > config.connectome_init_p)
        self.connectome = torch.ones(connectome_shape, dtype=dtype, device=self.device) * self.threshold
        self.connectome[connectome_init] = 0

    def step(self) -> None:
        # Decay activations over time.
        self.activations = (self.activations - self.decay).clamp(min=0)

        # Decay integrations over time
        # Need to figure out how to decay integrations over time
        # integration = (integration - decay).clamp(min=0)

        # Get neighbors and apply kernel
        activations_neighbors = unfold(self.activations.clone(), self.kernel_size, self.pad)
        activations_neighbors[:, :, :, :, (self.kernel_size ** 2) // 2] = 0

        # Check for active neighboring neurons
        neighbors_over_threshold = activations_neighbors >= self.threshold
        activations_neighbors[neighbors_over_threshold] = 1
        activations_neighbors[~neighbors_over_threshold] = 0

        # Update connectome with Spike Timing Dependent Plasticity (STDP).
        # Under the STDP process, if an input spike to a neuron tends, on average,
        # to occur immediately before that neuron's output spike, then that particular
        # input is made somewhat stronger. If an input spike tends, on average,
        # to occur immediately after an output spike.
        # https://en.wikipedia.org/wiki/Spike-timing-dependent_plasticity
        if self.stdp:
            # Long term potentiation
            # Neighbor is active and target integration is 0 < target integration < threshold
            neighbors_over_threshold_and_integrations = neighbors_over_threshold & \
                                        (0 < self.integrations.unsqueeze(4).repeat(1, 1, 1, 1, self.kernel_size ** 2) < self.threshold)
            self.connectome[neighbors_over_threshold_and_integrations] += self.activity_delta

            # Long term depression
            # Neighbor is active and target activation is 0 < target activation < threshold
            neighbors_over_threshold_and_activations = neighbors_over_threshold & \
                                        (0 < self.activations.unsqueeze(4).repeat(1, 1, 1, 1, self.kernel_size ** 2) < self.threshold)
            self.connectome[neighbors_over_threshold_and_activations] -= self.activity_delta

            # Limit range of connectome values
            self.connectome.clamp_(min=0, max=self.threshold * 2)

        # Randomly drop some neighboring activations
        if self.drop_p > 0:
            activations_neighbors *= (torch.rand(self.connectome.shape, device=self.connectome.device) < self.drop_p)

        # Sum neighboring activations.
        # This value is between 0 and kernel_size ** 2 - 1
        activations_neighbors *= (self.connectome >= self.threshold)
        activations_neighbors = activations_neighbors.sum(dim=-1)

        # Add sum of activations to integration if activation < 1.
        neuron_not_active_or_refractory = (self.activations < 1)
        self.integrations += activations_neighbors * neuron_not_active_or_refractory

        # If integration value over threshold set activation value to
        # threshold + activity_delta and reset integration value.
        integration_over_threshold = self.integrations > self.threshold
        self.activations[integration_over_threshold] = self.threshold + self.activity_delta
        self.integrations[integration_over_threshold] = 0


# -----------------------------------------------------------------------------
if __name__ == '__main__':

    def get_config():

        C = CN()

        # system
        C.system = CN()
        C.system.seed = 1337
        C.system.work_dir = './out/NeuronalCA'

        # model
        C.model = NeuronalCA.get_default_config()
        C.model.board_size = 256
        C.model.channels = 3
        C.model.kernel_size = 21
        C.model.pad = 10
        C.model.mod = 10 + 1
        C.model.threshold = 10
        C.model.decay = 1
        C.model.activity_delta = 1
        C.model.drop_p = 0.8
        C.model.connectome_init_p = 0.05

        # runner
        C.runner = Runner.get_default_config()

        # video
        C.video = CN()
        C.video.scale = 2.
        C.video.fps = 30
        C.video.show_nth_frame = 1
        C.video.duration = 5  # seconds
        C.size_v = (int(C.video.scale * C.model.board_size),) * 2
        C.video.iterations = C.video.fps * C.video.duration * C.video.show_nth_frame

        return C

# -----------------------------------------------------------------------------

    # get default config
    config = get_config()
    print(config)
    setup_logging(config)
    set_seed(config.system.seed)

    model = NeuronalCA(config.model)

    # construct the trainer object
    runner = Runner(config.trainer, model)

    # iteration callback
    def batch_end_callback(runner: Runner):

        if runner.iter_num % 10 == 0:
            print(f"iter_dt {runner.iter_dt * 1000:.2f}ms; iter {runner.iter_num}: train loss {runner.loss.item():.5f}")

        if runner.iter_num % 200 == 0:
            pass
            # evaluate both the train and test score
            # model.eval()
            # revert model to training mode
            # model.train()

    runner.set_callback('on_batch_end', batch_end_callback)

    # run the optimization
    runner.run()

