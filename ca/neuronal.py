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
        self.kernel_size = config.kernel_size
        self.pad = config.pad
        self.stride = config.stride
        self.threshold = config.threshold
        self.decay = config.decay
        self.activity_delta = config.activity_delta
        self.drop_p = config.drop_p

        # Initialize model
        self.activations = torch.zeros([1, 1, config.board_size, config.board_size], dtype=torch.int32)
        self.integrations = torch.zeros([1, 1, config.board_size, config.board_size], dtype=torch.int32)

        self.kernel = torch.ones(config.kernel_size ** 2, dtype=torch.int32)
        self.kernel[(config.kernel_size ** 2) // 2] = 0

        connectome_shape = (1, 1, config.board_size, config.board_size, config.kernel_size ** 2)
        self.connectome = config.threshold * torch.ones(connectome_shape, dtype=torch.int32)
        self.connectome *= (torch.rand(connectome_shape) < config.connectome_init_p)

    def step(self):
        # Decay activations over time.
        self.activations = (self.activations - self.decay).clamp(min=0)

        # Decay integrations over time
        # Need to figure out how to decay integrations over time
        # integration = (integration - decay).clamp(min=0)

        # Get neighbors and apply kernel
        activations_neighbors = unfold(self.activations.clone(), self.kernel_size, self.pad, self.stride)
        # TODO: replace kernel with just zero out middle element of activation neighbors
        # TODO: self.kernel[(self.kernel_size ** 2) // 2] = 0 but for activation neighbors
        activations_neighbors *= self.kernel

        # Check for active neighboring neurons
        activations_neighbors = (activations_neighbors >= self.threshold).int()  # Do we really need to convert to int?

        # Update connections
        # TODO: replace this with different weight updating method
        self.connectome += activations_neighbors * 2 - 1
        self.connectome = self.connectome.clamp(0, self.threshold * 2)

        # So what is going on here?
        condition_1 = (self.connectome > self.threshold)  # What the hell is this?
        value_1 = 1  # (mask - threshold).clamp(min=0) # And what the hell is this?
        activations_neighbors *= condition_1 * value_1

        # Randomly drop some neighboring activations
        activations_neighbors *= (torch.rand(self.connectome.shape, device=self.activations.device) < self.drop_p)

        # Sum neighboring activations.
        activations_neighbors = activations_neighbors.sum(dim=-1)  # This is the surrounding activation

        # Add sum of activations to integration.
        neuron_not_active_or_refractory = (self.activations < 1)
        self.integrations += activations_neighbors * neuron_not_active_or_refractory

        # If integration value over threshold set activation value to
        # threshold + activity_delta and reset integration value.
        self.activations[self.integrations > self.threshold] = self.threshold + self.activity_delta
        self.integrations[self.integrations > self.threshold] = 0


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

