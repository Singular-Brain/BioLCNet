from typing import Union, Tuple, Dict

import torch
from torch.nn.modules.utils import _pair

import collections
from tqdm import tqdm

from bindsnet.network.monitors import Monitor
from monitors import RewardMonitor

from learning import PostPre
from bindsnet.learning import MSTDP, NoOp

from bindsnet.network.nodes import LIFNodes, AdaptiveLIFNodes
from bindsnet.network.nodes import Input
from bindsnet.network.network import Network
from bindsnet.network.topology import Connection
from bindsnet.encoding import PoissonEncoder

from locally_connected_multi_chan import LocalConnection2D
from visualization import (
    plot_convergence_and_histogram,
    plot_locally_connected_output_weights,
    plot_locally_connected_feature_maps,
)

from monitors import RewardMonitor
import matplotlib.pyplot as plt

import seaborn as sn
from sklearn.metrics import confusion_matrix


class BioLCNet(Network):
    def __init__(
        self,
        n_classes: int,
        neuron_per_c: int,
        in_channels: int,
        n_channels_lc: int,
        filter_size: int,
        stride: int,
        time: int,
        reward_fn,
        n_neurons: int,
        pre_observation: bool,
        has_decision_period: bool,
        nu_LC: Union[float, Tuple[float, float]],
        nu_Output: float,
        dt: float,
        crop_size: int,
        inh_type_FC,
        inh_factor_LC: float,
        inh_factor_FC: float,
        wmin: float,
        wmax: float,
        theta_plus: float,
        tc_theta_decay: float,
        tc_trace: int,
        norm_factor_LC,
        load_path,
        save_path,
        LC_weights_path=None,
        trace_additive: bool = False,
        confusion_matrix: bool = False,
        lc_weights_vis: bool = False,
        out_weights_vis: bool = False,
        lc_convergence_vis: bool = False,
        out_convergence_vis: bool = False,
        online_rewarding: bool = False,
        gpu: bool = False,
        **kwargs,
    ) -> None:

        """
        Constructor for class `BioLCNet`.
        """

        super().__init__(dt=dt, reward_fn=None)

        if reward_fn is not None:
            self.reward_fn = reward_fn(**kwargs, dt=dt)
            self.reward_fn.network = self
            self.reward_fn.dt = self.dt
        else:
            self.reward_fn = None

        kwargs["dt"] = dt
        kwargs["n_labels"] = n_classes
        kwargs["neuron_per_c"] = neuron_per_c

        self.dt = dt
        self.gpu = gpu
        self.reward_fn = reward_fn(**kwargs)
        self.reward_fn.network = self
        self.reward_fn.dt = self.dt
        self.n_classes = n_classes
        self.neuron_per_class = neuron_per_c
        self.save_path = save_path
        self.load_path = load_path
        self.time = time
        self.crop_size = crop_size
        self.filter_size = filter_size
        self.clamp_intensity = kwargs.get("clamp_intensity", None)
        self.pre_observation = pre_observation
        self.has_decision_period = has_decision_period
        self.confusion_matrix = confusion_matrix
        self.lc_weights_vis = lc_weights_vis
        self.out_weights_vis = out_weights_vis
        self.lc_convergence_vis = lc_convergence_vis
        self.out_convergence_vis = out_convergence_vis
        self.in_channels = in_channels
        self.n_channels_lc = n_channels_lc
        self.convergences = {}
        self.norm_factor_LC = norm_factor_LC
        self.wmin = wmin
        self.wmax = wmax
        self.online = online_rewarding

        if kwargs["variant"] == "scalar":
            assert (
                self.has_decision_period == True
            ), "Decision period is necessary for scalar variant"

        if self.online == False:
            assert (
                self.has_decision_period == True
            ), "Decision period is necessary for offline learning"

        if self.has_decision_period == True:
            assert (
                self.online == False
            ), "Decision period is not compatible with online learning."
            self.observation_period = kwargs["observation_period"]
            assert self.observation_period >= 0, ""
            self.decision_period = kwargs["decision_period"]
            assert (
                self.decision_period > 0
            ), "Decision period should be greater than zero"
            self.learning_period = (
                self.time - self.observation_period - self.decision_period
            )

        elif self.pre_observation == True:
            self.observation_period = kwargs["observation_period"]
            assert self.observation_period >= 0, "Observation period cannot be negative"
            self.learning_period = self.time - self.observation_period
            self.decision_period = self.time - self.observation_period

        else:
            self.observation_period = 0
            self.decision_period = self.time
            self.learning_period = self.time

        ### nodes
        inp = Input(
            shape=[in_channels, crop_size, crop_size],
            traces=True,
            tc_trace=tc_trace,
            traces_additive=trace_additive,
        )
        self.add_layer(inp, name="input")

        ## Hidden layer
        compute_size = lambda inp_size, k, s: int((inp_size - k) / s) + 1
        conv_size = compute_size(crop_size, filter_size, stride)
        main = AdaptiveLIFNodes(
            shape=[n_channels_lc, conv_size, conv_size],
            traces=True,
            tc_trace=tc_trace,
            traces_additive=trace_additive,
            tc_theta_decay=tc_theta_decay,
            theta_plus=theta_plus,
        )

        self.add_layer(main, name="main")

        ### connections
        LC = LocalConnection2D(
            inp,
            main,
            filter_size,
            stride,
            in_channels,
            n_channels_lc,
            input_shape=(crop_size, crop_size),
            nu=_pair(nu_LC),
            update_rule=PostPre,
            wmin=wmin,
            wmax=wmax,
            norm=norm_factor_LC,
        )

        if LC_weights_path:
            a = torch.load(
                LC_weights_path, map_location=torch.device("cuda" if gpu else "cpu")
            )
            LC.w.data = a["state_dict"]["input_to_main1.w"]
            LC.nu = [0, 0]
            print("LC pre-trained weights loaded ...")
        else:
            print(
                "LC pre-trained weights not loaded. Training will be end-to-end and will take more time!"
            )

        self.add_connection(LC, "input", "main")
        self.convergences["lc"] = []

        ### LC inhibition
        main_width = compute_size(crop_size, filter_size, stride)
        w_inh_LC = torch.zeros(
            n_channels_lc, main_width, main_width, n_channels_lc, main_width, main_width
        )
        for c in range(n_channels_lc):
            for w1 in range(main_width):
                for w2 in range(main_width):
                    w_inh_LC[c, w1, w2, :, w1, w2] = -inh_factor_LC
                    w_inh_LC[c, w1, w2, c, w1, w2] = 0

        w_inh_LC = w_inh_LC.reshape(main.n, main.n)

        LC_recurrent_inhibition = Connection(
            source=main,
            target=main,
            w=w_inh_LC,
        )
        self.add_connection(LC_recurrent_inhibition, "main", "main")

        self.final_connection_source_name = "main"
        self.final_connection_source = main

        ### main to output
        out = LIFNodes(
            n=n_neurons,
            traces=True,
            traces_additive=trace_additive,
            tc_trace=tc_trace,
            tc_theta_decay=tc_theta_decay,
            theta_plus=theta_plus,
        )

        self.add_layer(out, "output")

        last_main_out = Connection(
            self.final_connection_source,
            out,
            nu=nu_Output,
            update_rule=MSTDP,
            wmin=wmin,
            wmax=wmax,
        )

        self.add_connection(last_main_out, self.final_connection_source_name, "output")
        self.convergences["last_main_out"] = []

        ### Inhibitory connection in the decoding layer
        if inh_type_FC == "between_layers":
            w = -inh_factor_FC * torch.ones(out.n, out.n)
            for c in range(n_classes):
                ind = slice(c * neuron_per_c, (c + 1) * neuron_per_c)
                w[ind, ind] = 0

            out_recurrent_inhibition = Connection(
                source=out,
                target=out,
                w=w,
                wmin=-inh_factor_FC,
                wmax=0,
            )
            self.add_connection(out_recurrent_inhibition, "output", "output")

        elif inh_type_FC == "one_2_all":
            w = -inh_factor_FC * (torch.ones(out.n, out.n) - torch.eye(out.n, out.n))
            out_recurrent_inhibition = Connection(
                source=out,
                target=out,
                w=w,
                wmin=-inh_factor_FC,
                wmax=0,
            )
            self.add_connection(out_recurrent_inhibition, "output", "output")

        # Directs network to self.self.gpu
        if self.gpu:
            self.to("cuda")

    def run(
        self, inputs: Dict[str, torch.Tensor], time: int, one_step=True, **kwargs
    ) -> None:
        """
        Simulate network for given inputs and time.

        :param inputs: Dictionary of ``Tensor``s of shape ``[time, *input_shape]`` or
                    ``[time, batch_size, *input_shape]``.
        :param time: Simulation time.
        :param one_step: Whether to run the network in "feed-forward" mode, where inputs
            propagate all the way through the network in a single simulation time step.
            Layers are updated in the order they are added to the network.
        """

        # Check input type
        assert type(inputs) == dict, (
            "'inputs' must be a dict of names of layers "
            + f"(str) and relevant input tensors. Got {type(inputs).__name__} instead."
        )
        # Parse keyword arguments.
        clamps = kwargs.get("clamp", {})
        unclamps = kwargs.get("unclamp", {})
        masks = kwargs.get("masks", {})
        injects_v = kwargs.get("injects_v", {})
        self.true_label = kwargs.get("true_label", None)
        kwargs["pred_label"] = None
        kwargs["local_rewarding"] = False
        kwargs["neuron_per_class"] = self.neuron_per_class

        # Compute reward.
        kwargs["give_reward"] = False
        if self.reward_fn is not None and self.learning == True:
            kwargs["reward"] = self.reward_fn.compute(**kwargs)

        # Dynamic setting of batch size.
        if inputs != {}:
            for key in inputs:
                # goal shape is [time, batch, n_0, ...]
                if len(inputs[key].size()) == 1:
                    # current shape is [n_0, ...]
                    # unsqueeze twice to make [1, 1, n_0, ...]
                    inputs[key] = inputs[key].unsqueeze(0).unsqueeze(0)
                elif len(inputs[key].size()) == 2:
                    # current shape is [time, n_0, ...]
                    # unsqueeze dim 1 so that we have
                    # [time, 1, n_0, ...]
                    inputs[key] = inputs[key].unsqueeze(1)

            for key in inputs:
                # batch dimension is 1, grab this and use for batch size
                if inputs[key].size(1) != self.batch_size:
                    self.batch_size = inputs[key].size(1)

                    for l in self.layers:
                        self.layers[l].set_batch_size(self.batch_size)

                    for m in self.monitors:
                        self.monitors[m].reset_state_variables()

                break

        # Effective number of timesteps.
        timesteps = int(self.time / self.dt)

        # Simulate network activity for `time` timesteps.
        for t in range(timesteps):

            # Make a decision and compute reward
            if self.online == False:
                if (
                    self.has_decision_period
                    and t == self.observation_period + self.decision_period
                ):
                    out_spikes = (
                        self.spikes["output"]
                        .get("s")
                        .view(t, self.n_classes, self.neuron_per_class)
                    )
                    self.sum_spikes = (
                        out_spikes[self.observation_period : t, :, :].sum(0).sum(1)
                    )
                    kwargs["pred_label"] = torch.argmax(self.sum_spikes)
                    kwargs["true_label"] = self.true_label
                    kwargs["give_reward"] = True
                    # TODO: if you want per spike modulation, pls calculate rew_base and punish_base
                    kwargs["target_spikes"] = self.sum_spikes[kwargs["true_label"]]
                    kwargs["pred_spikes"] = self.sum_spikes[kwargs["pred_label"]]
                    kwargs["sum_spikes"] = self.sum_spikes
                    assert (
                        kwargs["variant"] == "scalar"
                        or kwargs["variant"] == "per_spike"
                        or kwargs["variant"] == "per_spike_target"
                    ), "the variant must be scalar or per_spike"
                    if self.learning == True:
                        kwargs["reward"] = self.reward_fn.compute(**kwargs)

            # Get input to all layers (synchronous mode).
            current_inputs = {}
            if not one_step:
                current_inputs.update(self._get_inputs())

            for l in self.layers:
                # Update each layer of nodes.
                if l in inputs:
                    if l in current_inputs:
                        current_inputs[l] += inputs[l][t]
                    else:
                        current_inputs[l] = inputs[l][t]

                if one_step:
                    # Get input to this layer (one-step mode).
                    current_inputs.update(self._get_inputs(layers=[l]))

                if l in current_inputs:
                    self.layers[l].forward(x=current_inputs[l])
                else:
                    self.layers[l].forward(x=torch.zeros(self.layers[l].s.shape))

                # Clamp neurons to spike.
                clamp = clamps.get(l, None)
                if clamp is not None:
                    if clamp.ndimension() == 1:
                        self.layers[l].s[:, clamp] = 1
                    else:
                        self.layers[l].s[:, clamp[t]] = 1

                # Clamp neurons not to spike.
                unclamp = unclamps.get(l, None)
                if unclamp is not None:
                    if unclamp.ndimension() == 1:
                        self.layers[l].s[:, unclamp] = 0
                    else:
                        self.layers[l].s[:, unclamp[t]] = 0

                # Inject voltage to neurons.
                inject_v = injects_v.get(l, None)
                if inject_v is not None:
                    if inject_v.ndimension() == 1:
                        self.layers[l].v += inject_v
                    else:
                        self.layers[l].v += inject_v[t]

            # Run synapse updates.
            for c in self.connections:
                if t < self.time - self.learning_period and c[1].startswith("output"):
                    self.connections[c].update(
                        mask=masks.get(c, None), learning=False, **kwargs
                    )
                else:
                    kwargs["target_name"] = c[1]
                    self.connections[c].update(
                        mask=masks.get(c, None), learning=self.learning, **kwargs
                    )

            # # Get input to all layers.
            # current_inputs.update(self._get_inputs())

            if (
                self.reward_fn is not None
                and self.online == True
                and t >= self.time - self.learning_period
                and self.learning == True
            ):
                kwargs["reward"] = self.reward_fn.online_compute(**kwargs)
            # Record state variables of interest.
            for m in self.monitors:
                if type(self.monitors[m]) != RewardMonitor:
                    self.monitors[m].record()
                else:
                    self.monitors[m].record(**kwargs)

        # Re-normalize connections.
        for c in self.connections:
            self.connections[c].normalize()

    def fit(
        self,
        dataloader,
        val_loader,
        reward_hparams,
        label=None,
        hparams=None,
        online_validate=True,
        n_train=10000,
        n_val=250,
        val_interval=250,
        running_window_length=250,
        verbose=True,
        **kwargs,
    ):

        self.verbose = verbose
        self.label = label
        # add Monitors
        reward_monitor = RewardMonitor(time=self.time)
        self.add_monitor(reward_monitor, name="reward")

        acc_hist = collections.deque([], running_window_length)

        self.spikes = {}
        for layer in set(self.layers):
            self.spikes[layer] = Monitor(
                self.layers[layer], state_vars=["s"], time=None
            )
            self.add_monitor(self.spikes[layer], name="%s_spikes" % layer)
            self.dopaminergic_layers = self.layers["output"]

        val_acc = 0.0
        acc = 0.0

        reward_history = []

        ### Load a previous model
        if self.load_path:
            self.model_params = torch.load(self.load_path)
            self.load_state_dict(torch.load(self.load_path)["state_dict"])
            iteration = self.model_params["iteration"]
            hparams = self.model_params["hparams"]
            train_accs = self.model_params["train_accs"]
            val_accs = self.model_params["val_accs"]
            acc_rewards = self.model_params["acc_rewards"]
            print(
                f"Previous model loaded! Resuming training from iteration {iteration}..., last running training accuracy: {train_accs[-1]}\n"
            ) if self.verbose else None
        else:
            print(
                f"Previous model not found! Training from the beginning...\n"
            ) if self.verbose else None
            val_accs = []
            train_accs = []
            acc_rewards = []
        from tqdm.notebook import tqdm

        pbar = tqdm(total=n_train)
        self.reset_state_variables()

        for (i, datum) in enumerate(dataloader):
            if self.load_path:
                if i < iteration:
                    n_train += 1
                    continue

            if i >= n_train:
                break

            image = datum["encoded_image"]
            if self.label is None:
                label = datum["label"]

            # Run the network on the input.
            if self.gpu:
                inputs = {
                    "input": image.cuda().view(
                        self.time, 1, self.in_channels, self.crop_size, self.crop_size
                    )
                }
            else:
                inputs = {
                    "input": image.view(
                        self.time, 1, self.in_channels, self.crop_size, self.crop_size
                    )
                }

            ### Spike clamping (baseline activity)
            clamp = {}
            if self.clamp_intensity is not None:
                encoder = PoissonEncoder(time=self.time, dt=self.dt)
                clamp["output"] = encoder.enc(
                    datum=torch.rand(self.layers["output"].n) * self.clamp_intensity,
                    time=self.time,
                    dt=self.dt,
                )

            self.run(
                inputs=inputs,
                time=self.time,
                **reward_hparams,
                one_step=True,
                true_label=label.int().item(),
                dopaminergic_layers=self.dopaminergic_layers,
                clamp=clamp,
            )

            # Get voltage recording.
            reward_history.append(reward_monitor.get())

            # Add to spikes recording.
            predicted_label = torch.argmax(self.sum_spikes)

            if predicted_label == label:
                acc_hist.append(1)
            else:
                acc_hist.append(0)

            w_lc = self.connections[("input", "main")].w
            w_last_main_out = self.connections[
                (self.final_connection_source_name, "output")
            ].w

            convg_lc1 = 1 - torch.mean((w_lc - self.wmin) * (self.wmax - w_lc))
            convg_out = 1 - torch.mean(
                (w_last_main_out - self.wmin) * (self.wmax - w_last_main_out)
            )
            if self.norm_factor_LC is not None:
                mean_norm_factor_lc = self.norm_factor_LC / w_lc.shape[-1]
                convg_lc1 = 1 - (
                    torch.mean((w_lc - self.wmin) * (self.wmax - w_lc))
                    / (
                        (mean_norm_factor_lc - self.wmin)
                        * (self.wmax - mean_norm_factor_lc)
                    )
                )

            self.convergences["lc"].append((convg_lc1 * 10 ** 4).round() / (10 ** 4))
            self.convergences["last_main_out"].append(
                (convg_out * 10 ** 4).round() / (10 ** 4)
            )

            print(
                "\routput",
                self.sum_spikes,
                "pred_label:",
                predicted_label.item(),
                "GT:",
                label.item(),
                end="",
            )

            acc = 100 * sum(acc_hist) / len(acc_hist)
            self.reward_fn.update(
                accumulated_reward=sum(reward_monitor.get()),
                ema_window=reward_hparams["ema_window"],
            )

            if online_validate and i % val_interval == 0 and i != 0:
                self.reset_state_variables()
                val_acc = self.evaluate(val_loader, n_val, **reward_hparams)
                # tensorboard.writer.add_scalars("accuracy", {"train": acc, "val" : val_acc}, i)
                train_accs.append(acc)
                val_accs.append(val_acc)
                # acc_rewards.append(sum(reward_monitor.get()))
                if self.save_path is not None:
                    model_params = {
                        "state_dict": self.state_dict(),
                        "hparams": hparams,
                        "iteration": i,
                        "val_accs": val_accs,
                        "train_accs": train_accs,
                        "acc_rewards": acc_rewards,
                    }
                    torch.save(model_params, self.save_path)

            print(
                "\nRunning accuracy: "
                + "{:.2f}".format(acc)
                + "%"
            )
            self.reset_state_variables()  # Reset state variables.
            pbar.set_description_str(
                "Running accuracy: "
                + "{:.2f}".format(acc)
                + "%"
            )
            pbar.update()
        if val_acc > 0:
            print("Test accuracy: "+"{:.2f}".format(val_acc))
        else:
            print("Training is complete!")

    def evaluate(self, val_loader, n_val, **kwargs):

        acc_hist_val = collections.deque([], n_val)

        self.train(False)
        self.learning = False

        GT, y_pred = [], []
        for (i, datum) in enumerate(val_loader):
            if i >= n_val:
                break

            image = datum["encoded_image"]
            if self.label is None:
                label = datum["label"]
            else:
                label = self.label

            # Run the network on the input.
            if self.gpu:
                inputs = {
                    "input": image.cuda().view(
                        self.time, 1, self.in_channels, self.crop_size, self.crop_size
                    )
                }
            else:
                inputs = {
                    "input": image.view(
                        self.time, 1, self.in_channels, self.crop_size, self.crop_size
                    )
                }

            self.run(
                inputs=inputs,
                time=self.time,
                **kwargs,
                one_step=True,
                true_label=label.int().item(),
                dopaminergic_layers=self.dopaminergic_layers,
            )

            predicted_label = torch.argmax(self.sum_spikes)

            if predicted_label == label:
                acc_hist_val.append(1)
            else:
                acc_hist_val.append(0)

            GT.append(label)
            y_pred.append(predicted_label)
            if self.testing != True and self.verbose:
                print(
                    "\rSaving the model (if save path is specified)...",
                    end="",
                )
            else:
                print("\r*Test: output",
                self.sum_spikes,
                "predicted_label:",
                predicted_label.item(),
                "GT:",
                label.item(),
                end="",)

            self.reset_state_variables()  # Reset state variables.

        self.train(True)
        self.learning = True

        if self.confusion_matrix:
            self.plot_confusion_matrix(GT, y_pred)

        if self.lc_weights_vis:
            plot_locally_connected_feature_maps(
                self.connections[("input", "main1")].w,
                self.n_channels1,
                self.in_channels,
                0,
                self.crop_size,
                self.filter_size1,
                self.layers["main1"].shape[1],
            )
            plt.show()

        if self.lc_convergence_vis:
            plot_convergence_and_histogram(
                self.connections[("input", "main1")].w, self.convergences["lc1"]
            )
            plt.show()
        if self.out_convergence_vis:
            plot_convergence_and_histogram(
                self.connections[(self.final_connection_source_name, "output")].w,
                self.convergences["last_main_out"],
            )
            plt.show()

        if self.out_weights_vis:
            plot_locally_connected_output_weights(
                self.connections[("input", "main1")].w,
                self.connections[(self.final_connection_source_name, "output")].w,
                0,
                0,
                self.neuron_per_class,
                self.n_channels1,
                self.in_channels,
                0,
                self.crop_size,
                self.filter_size1,
                self.layers["main1"].shape[1],
            )
            plt.show()
            plot_locally_connected_output_weights(
                self.connections[("input", "main1")].w,
                self.connections[(self.final_connection_source_name, "output")].w,
                0,
                1,
                self.neuron_per_class,
                self.n_channels1,
                self.in_channels,
                0,
                self.crop_size,
                self.filter_size1,
                self.layers["main1"].shape[1],
            )
            plt.show()
        val_acc = 100 * sum(acc_hist_val) / len(acc_hist_val)
        return val_acc

    @staticmethod
    def plot_confusion_matrix(GT, y_predicted):
        cm = confusion_matrix(GT, y_predicted)
        plt.figure(figsize=(10, 7))
        sn.heatmap(cm, annot=True)
        plt.xlabel("Predicted")
        plt.ylabel("Truth")
        plt.show()

    def one_step(self, datum, label=None):
        self.reset_state_variables()

        image = datum["encoded_image"]
        if label is None:
            label = datum["label"]

        if self.self.gpu:
            inputs = {
                "input": image.cuda().view(
                    self.time, 1, self.in_channels, self.crop_size, self.crop_size
                )
            }
        else:
            inputs = {
                "input": image.view(
                    self.time, 1, self.in_channels, self.crop_size, self.crop_size
                )
            }

        clamp = {}
        if self.clamp_intensity is not None:
            encoder = PoissonEncoder(time=self.time, dt=self.dt)
            clamp["output"] = encoder.enc(
                datum=torch.rand(self.layers["output"].n) * self.clamp_intensity,
                time=self.time,
                dt=self.dt,
            )

        self.run(
            inputs=inputs,
            time=self.time,
            **self.reward_hparams,
            one_step=True,
            true_label=label.int().item(),
            dopaminergic_layers=self.dopaminergic_layers,
        )
