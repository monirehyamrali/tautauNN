# coding: utf-8

from __future__ import annotations

import os
import io
import gc
from typing import Callable, Any

import psutil
import numpy as np
import tensorflow as tf
from tensorflow.experimental import numpy as tnp
from tensorflow.python.keras.engine import compile_utils
from keras.src.utils.io_utils import print_msg
import sklearn.metrics
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tautaunn.util import plot_confusion_matrix, plot_class_outputs


debug_layer = tf.autograph.experimental.do_not_convert


def get_device(device: str = "cpu", num_device: int = 0) -> tf.device:
    if device == "gpu":
        gpus = tf.config.experimental.list_physical_devices("GPU")
        if gpus:
            selected_gpu = None
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
                if gpu.name.endswith(f":{num_device}"):
                    # tf.config.set_logical_device_configuration(
                    #     gpu,
                    #     [tf.config.LogicalDeviceConfiguration(memory_limit=1024 * 45)],  # unit is MB
                    # )
                    selected_gpu = tf.device(f"/device:GPU:{num_device}")
            if selected_gpu is not None:
                return selected_gpu
        print(f"no gpu with the requested number {num_device} found, falling back to cpu:0")
        num_device = 0

    return tf.device(f"/device:CPU:{num_device}")


def fig_to_image_tensor(fig) -> tf.Tensor:
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    image = tf.image.decode_png(buf.getvalue(), channels=4)
    image = tf.expand_dims(image, 0)
    return image


class ClassificationModelWithValidationBuffers(tf.keras.Model):
    """
    Custom model that saves labels and predictions during validation and resets them before starting a new round.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # buffers for labels, predictions and weights that are filled after validation
        self.buffer_y = self._create_validation_buffer(self.output_shape[-1])
        self.buffer_y_pred = self._create_validation_buffer(self.output_shape[-1])
        self.buffer_y_empty = self._create_validation_buffer(self.output_shape[-1])
        self.buffer_weight = self._create_validation_buffer(1)
        self.buffer_weight_empty = self._create_validation_buffer(1)

    def _create_validation_buffer(self, dim: int) -> tf.Variable:
        return tf.Variable(
            tnp.empty((0, dim), dtype=tf.float32),
            shape=[None, dim],
            trainable=False,
        )

    def _reset_validation_buffer(self) -> None:
        self.buffer_y.assign(self.buffer_y_empty)
        self.buffer_y_pred.assign(self.buffer_y_empty)
        self.buffer_weight.assign(self.buffer_weight_empty)

    def _extend_validation_buffer(self, y: tf.Tensor, y_pred: tf.Tensor, weight: tf.Tensor) -> None:
        self.buffer_y.assign(tf.concat([self.buffer_y, y], axis=0))
        self.buffer_y_pred.assign(tf.concat([self.buffer_y_pred, y_pred], axis=0))
        self.buffer_weight.assign(tf.concat([self.buffer_weight, weight], axis=0))

    def test_on_batch(self, *args, **kwargs):
        self._reset_validation_buffer()
        return super().test_on_batch(*args, **kwargs)

    def evaluate(self, *args, **kwargs):
        self._reset_validation_buffer()
        return super().evaluate(*args, **kwargs)

    def test_step(self, data):
        x, y, sample_weight = tf.keras.utils.unpack_x_y_sample_weight(data)
        y_pred = self(x, training=False)

        self._extend_validation_buffer(y, y_pred, sample_weight)

        self.compute_loss(x, y, y_pred, sample_weight)
        return self.compute_metrics(x, y, y_pred, sample_weight)


class FadeInLayer(tf.keras.layers.Layer):

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

        self.factor = None

    def build(self, input_shape):
        self.factor = self.add_weight(
            shape=(),
            dtype=tf.float32,
            initializer=tf.keras.initializers.Constant(0.0),
            name="fadein_factor",
            trainable=False,
        )

        return super().build(input_shape)

    def call(self, inputs):
        return inputs * self.factor

    def get_config(self):
        config = super().get_config()
        return config


class L2Metric(tf.keras.metrics.Metric):

    def __init__(
        self,
        model: tf.keras.Model,
        select_layers: Callable[[tf.keras.Model], list[tf.keras.layers.Layer]] | None = None,
        name: str = "l2",
        **kwargs,
    ) -> None:
        super().__init__(name=name, **kwargs)

        # store kernels and l2 norms of dense layers
        self.kernels: list[tf.Tensor] = []
        self.norms: list[np.ndarray] = []
        for layer in (select_layers if callable(select_layers) else self._select_layers)(model):
            self.kernels.append(layer.kernel)
            self.norms.append(layer.kernel_regularizer.l2)

        # book the l2 metric
        self.l2: tf.Variable = self.add_weight(name="l2", initializer="zeros")

    def _select_layers(self, model: tf.keras.Model) -> list[tf.keras.layers.Layer]:
        return [
            layer for layer in model.layers
            if isinstance(layer, tf.keras.layers.Dense) and layer.kernel_regularizer is not None
        ]

    def update_state(
        self,
        y_true: tf.Tensor | None,
        y_pred: tf.Tensor | None,
        sample_weight: tf.Tensor | None = None,
    ) -> None:
        if self.kernels and self.norms:
            self.l2.assign(tf.add_n([tf.reduce_sum(k**2) * n for k, n in zip(self.kernels, self.norms)]))

    def result(self) -> tf.Tensor:
        return self.l2

    def reset_states(self) -> None:
        self.l2.assign(0.0)


class MetricMeta(type(tf.keras.metrics.Metric)):

    def __new__(meta_cls, cls_name, bases, cls_dict):
        return super().__new__(meta_cls, cls_dict.get("_cls_name", cls_name), bases, cls_dict)


def metric_class_factory(
    base: type[tf.keras.metrics.Metric],
    name: str | None = None,
) -> type[tf.keras.metrics.Metric]:

    class CustomMetric(base, metaclass=MetricMeta):

        _cls_name = name or base.__name__

        def __init__(self, name: str, output_name: str | None = None, **kwargs) -> None:
            super().__init__(name=name, **kwargs)
            self.output_name = output_name

        def update_state(
            self,
            x: dict[str, tf.Tensor],
            y_true: dict[str, tf.Tensor],
            y_pred: dict[str, tf.Tensor],
            sample_weight: tf.Tensor | None = None,
            **kwargs,
        ) -> None:
            y_true = y_true[self.output_name] if self.output_name else y_true
            y_pred = y_pred[self.output_name] if self.output_name else y_pred
            super().update_state(y_true, y_pred, sample_weight)

    return CustomMetric


class CustomMetricSum(tf.keras.callbacks.Callback):
    def __init__(self, name: str, metrics: list[str], **kwargs):
        super().__init__(**kwargs)
        self.name = name
        self.metrics = metrics

    def on_epoch_end(self, epoch: int, logs: dict[str, Any] | None = None) -> None:
        logs = logs or {}
        logs[self.name] = sum(logs.get(metric, 0.0) for metric in self.metrics)


class CycleLR(tf.keras.callbacks.Callback):
    # mostly taken from https://github.com/titu1994/keras-one-cycle/blob/master/clr.py
    def __init__(
            self,
            steps_per_epoch: int,
            epoch_per_cycle: int = 5,
            policy: str = "triangular2",
            lr_range: list = [1e-5, 3e-3],
            final_lr: float | None = None,
            invert: bool = True,
            monitor: str = "val_ce",
            reduce_on_end: bool = False,
            lr_patience: int = 10,
            lr_factor: float = 0.5,
            mode: str = "min",
            es_patience: int = 10,
            max_cycles: int = 10,
            min_delta: float = 1.0e-5,
            repeat_func: Callable[[ReduceLRAndStop, dict[str, Any]], None] | None = None,
            verbose: int = 0,
            **kwargs,
    ):
        super().__init__(**kwargs)
        # some checks
        if policy not in ["triangular", "triangular2"]:
            raise ValueError(f"{self.__class__.__name__} received unknown policy ({policy})")
        if mode not in ["min", "max"]:
            raise ValueError(f"{self.__class__.__name__} received unknown mode ({mode})")
        if es_patience < 0:
            raise ValueError(f"{self.__class__.__name__} received es_patience < 0 ({es_patience})")

        # set attributes
        self.invert = invert
        self.lr_range = lr_range
        self.final_lr = final_lr
        if not self.lr_range[-1] > self.lr_range[0]:
            raise ValueError(
                f"{self.__class__.__name__}: Upper bound of LR Range must be larger than lower."
                " If inverted policy is desired, set inverted to True",
            )
        if self.invert:
            self.lr_range = self.lr_range[::-1]
            print(f"LR Range: {self.lr_range}")

        self.reduce_on_end = reduce_on_end
        self.monitor = monitor
        self.policy = policy
        self.mode = mode
        self.es_patience = int(es_patience)
        self.lr_patience = int(lr_patience)
        self.verbose = int(verbose)
        self.steps_per_epoch = steps_per_epoch
        self.epoch_per_cycle = epoch_per_cycle
        self.cycle_width = self.lr_range[1] - self.lr_range[0]
        self.steps = self.steps_per_epoch * self.epoch_per_cycle
        self.half_life = int(self.steps / 2.)
        self.min_delta = abs(float(min_delta))
        self.step_size = self.cycle_width / self.half_life
        self.lr_min = self.lr_range[np.argmin(self.lr_range)]
        self.repeat_func = repeat_func
        self.max_cycles = max_cycles
        self.lr_factor = float(lr_factor)

        # state
        self.history = {}
        self.wait: int = 0
        self.best_epoch: int = -1
        self.best_weights: tuple[tf.Tensor, ...] | None = None
        self.best_metric: float = np.nan
        self.monitor_op: Callable[[float, float], bool] | None = None
        self.repeat_counter: int = 0
        self.cycle_step: int = 0
        self.cycle_count: int = 0
        self.reduce_lr_and_stop: bool = False
        self.lr_counter: int = 0

        self._reset()

    def _reset(self) -> None:
        self.wait = 0
        self.lr_counter = 0
        self.skip_lr_monitoring = False
        self.repeat_counter = 0
        self.cycle_step = 0

        self._reset_best()

    def _reset_best(self) -> None:
        self.best_epoch = -1
        self.best_weights = None
        if self.mode == "min":
            self.best_metric = np.inf
            self.best_metric_with_previous_lr = np.inf
            self.monitor_op = lambda cur, best: (best - cur) > self.min_delta
        else:  # "max"
            self.best_metric = -np.inf
            self.best_metric_with_previous_lr = -np.inf
            self.monitor_op = lambda cur, best: (cur - best) > self.min_delta

    def calc_lr(self):
        if self.cycle_step < self.half_life:
            new_lr = self.lr_range[0] + (self.cycle_step * self.step_size)
            return new_lr
        else:
            new_lr = self.lr_range[1] - ((self.cycle_step - self.half_life) * self.step_size)
            return new_lr

    def on_train_begin(self, logs={}):
        logs = logs or {}
        self._reset()
        print(f"Starting CycleLR with {self.policy} policy and {self.lr_range} range.")
        tf.keras.backend.set_value(self.model.optimizer.lr, self.calc_lr())

    def on_batch_end(self, epoch, logs=None):
        logs = logs or {}
        if not self.reduce_lr_and_stop:
            new_lr = self.calc_lr()
            self.cycle_step += 1
            # setdefault() adds the second argument in case the key doesn't exist
            # if it does already exist, it does nothing
            self.history.setdefault("lr", []).append(
                tf.keras.backend.get_value(self.model.optimizer.lr))
            tf.keras.backend.set_value(self.model.optimizer.lr, new_lr)
            for k, v in logs.items():
                self.history.setdefault(k, []).append(v)

    def _reset_before_new_cycle(self) -> None:
        self.cycle_step = 0
        # for triangular policy, the LR range stays the same so nothing to be done here
        if self.cycle_count > 0:
            if self.policy == "triangular2":
                if np.max(self.lr_range) > 4 * np.min(self.lr_range):
                    # reduce the top of lr_range to half it's value
                    self.lr_range[np.argmax(self.lr_range)] /= 2.
                    self.cycle_width = self.lr_range[1] - self.lr_range[0]
                    self.step_size = self.cycle_width / self.half_life
                    # new_lr = self.calc_lr()
                    # tf.keras.backend.set_value(self.model.optimizer.lr, new_lr)
                else:
                    return

    def _reset_for_fine_tuning(self) -> None:
        self.wait = 0
        self.repeat_counter = 0

    def on_epoch_end(self, epoch, logs):
        if not self.reduce_lr_and_stop:
            if self.cycle_step == self.steps:
                self.cycle_count += 1
                self._reset_before_new_cycle()

            # add the current learning rate to the logs
            logs = logs or {}
            logs["lr"] = tf.keras.backend.get_value(self.model.optimizer.lr)

            # do nothing when no metric is available yet
            value = self.get_monitor_value(logs)
            if value is None:
                return

            # helper to get a newline only for the first invocation
            nls = {"nl": "\n"}
            nl = lambda: nls.pop("nl", "")
            # new best value?
            if self.best_metric is None or self.monitor_op(value, self.best_metric):
                self.best_metric = value
                self.best_weights = self.model.get_weights()
                self.best_epoch = epoch
                self.wait = 0
                if self.verbose >= 2:
                    print_msg(f"{nl()}{self.__class__.__name__}: recorded new best value of {value:.5f}")
                logs["last_best"] = 0
                return
            logs["last_best"] = int(epoch - self.best_epoch)

            # no improvement, increase wait counter
            self.wait += 1

            # run at least one full cycle
            if self.cycle_count < 1:
                return

            if self.cycle_count == self.max_cycles:
                if self.reduce_on_end:
                    self.reduce_lr_and_stop = True
                    # set the lr to the mid of the current lr range
                    if self.final_lr is None:
                        self.final_lr = (self.lr_range[0] + self.lr_range[1]) / 2.
                    tf.keras.backend.set_value(self.model.optimizer.lr, self.final_lr)
                    self.wait = 0
                    return
                else:
                    self.model.stop_training = True
                    if self.verbose >= 1:
                        print_msg(f"{nl()}{self.__class__.__name__}: early stopping triggered")

            # repeat cycling?
            if callable(self.repeat_func) and self.repeat_func(self, logs):
                # yes, repeat
                self.repeat_counter += 1
                self._reset_before_repeat()
                if self.verbose >= 1:
                    print_msg(
                        f"{nl()}{self.__class__.__name__}: repeat_func triggered repitition of lr and es cycle",
                    )
                return

            # within patience?
            if self.wait <= self.es_patience:
                return

            # stop training completely
            self.model.stop_training = True
            if self.verbose >= 1:
                print_msg(f"{nl()}{self.__class__.__name__}: early stopping triggered")
        else:
            # add the current learning rate to the logs
            logs = logs or {}
            logs["lr"] = tf.keras.backend.get_value(self.model.optimizer.lr)

            # do nothing when no metric is available yet
            value = self.get_monitor_value(logs)
            if value is None:
                return

            # helper to get a newline only for the first invocation
            nls = {"nl": "\n"}
            nl = lambda: nls.pop("nl", "")

            # new best value?
            if self.best_metric is None or self.monitor_op(value, self.best_metric):
                self.best_metric = value
                self.best_weights = self.model.get_weights()
                self.best_epoch = epoch
                self.wait = 0
                if self.verbose >= 2:
                    print_msg(f"{nl()}{self.__class__.__name__}: recorded new best value of {value:.5f}")
                logs["last_best"] = 0
                return
            logs["last_best"] = int(epoch - self.best_epoch)

            # no improvement, increase wait counter
            self.wait += 1

            #
            # lr monitoring
            #

            if self.wait <= self.lr_patience:
                return
            # drop?
            if (
                self.best_metric_with_previous_lr is None or
                self.monitor_op(self.best_metric, self.best_metric_with_previous_lr)
            ):
                # yes, drop
                logs["lr"] *= self.lr_factor
                tf.keras.backend.set_value(self.model.optimizer.lr, logs["lr"])
                self.lr_counter += 1
                self.wait = 0
                if self.verbose >= 1:
                    msg = (
                        f"{nl()}{self.__class__.__name__}: reducing learning rate to {logs['lr']:.2e} "
                        f"(reduction {self.lr_counter}), best metric is {self.best_metric:.5f}"
                    )
                    if self.best_metric_with_previous_lr not in (None, np.inf, -np.inf):
                        msg += f", was {self.best_metric_with_previous_lr:.5f} with previous learning rate"
                    print_msg(msg)
                self.best_metric_with_previous_lr = self.best_metric
            else:
                # last drop had already no effect, stop monitoring for lr drops
                self.skip_lr_monitoring = True
                self.wait = 0
                if self.verbose >= 1:
                    print_msg(
                        f"{nl()}{self.__class__.__name__}: last learning rate reduction had no effect, "
                        f"from now on checking early stopping with patience {self.es_patience}",
                    )

            # within patience?
            if self.wait <= self.es_patience:
                return

            # stop training completely
            self.model.stop_training = True
            if self.verbose >= 1:
                print_msg(f"{nl()}{self.__class__.__name__}: early stopping triggered")

    def on_epoch_begin(self, epoch, logs=None):
        if self.cycle_step == 0 and self.cycle_count > 0:
            if not self.reduce_lr_and_stop:
                print(f"\n Cycle {self.cycle_count} Finished after epoch: {epoch}")
                print(f"\n Starting new cycle with lr_range: {self.lr_range}")
        if self.cycle_count == self.max_cycles:
            if self.reduce_on_end:
                print(f"\n Cycle {self.cycle_count} Finished after epoch: {epoch}")
                print(f"\n Initiating ReduceLRandStop after epoch: {epoch}")
                print(f"\n Switching to: {self.final_lr}\n")
                # add 1 to cycle count to stop from printing this message again
                self.cycle_count += 1

    def on_train_end(self, logs: dict[str, Any] | None = None) -> None:
        self.restore_best_weights()

    def restore_best_weights(self) -> bool:
        if self.best_weights is None:
            return False
        self.model.set_weights(self.best_weights)
        if self.verbose >= 1:
            print_msg(
                f"{self.__class__.__name__}: recovered best weights from epoch {self.best_epoch + 1}, "
                f"best metric was {self.best_metric:.5f}",
            )

    def get_monitor_value(self, logs: dict[str, Any]) -> float | int:
        logs = logs or {}
        value = logs.get(self.monitor)
        if value is None:
            print_msg(f"{self.__class__.__name__}: metric '{self.monitor}' not available, found {','.join(list(logs))}")
        return value


class ReduceLRAndStop(tf.keras.callbacks.Callback):

    def __init__(
        self,
        monitor: str = "val_loss",
        min_delta: float = 1.0e-5,
        mode: str = "min",
        lr_start_epoch: int = 0,
        lr_patience: int = 10,
        lr_factor: float = 0.1,
        es_start_epoch: int = 0,
        es_patience: int = 1,
        repeat_func: Callable[[ReduceLRAndStop, dict[str, Any]], None] | None = None,
        verbose: int = 0,
        **kwargs,
    ):
        super().__init__(**kwargs)

        # some checks
        if mode not in ["min", "max"]:
            raise ValueError(f"{self.__class__.__name__} received unknown mode ({mode})")
        if lr_patience < 0:
            raise ValueError(f"{self.__class__.__name__} received lr_patience < 0 ({lr_patience})")
        if lr_factor >= 1.0:
            raise ValueError(f"{self.__class__.__name__} received lr_factor >= 1 ({lr_factor})")
        if es_patience < 0:
            raise ValueError(f"{self.__class__.__name__} received es_patience < 0 ({es_patience})")

        # set attributes
        self.monitor = monitor
        self.min_delta = abs(float(min_delta))
        self.mode = mode
        self.lr_start_epoch = int(lr_start_epoch)
        self.lr_patience = int(lr_patience)
        self.lr_factor = float(lr_factor)
        self.es_start_epoch = int(es_start_epoch)
        self.es_patience = int(es_patience)
        self.repeat_func = repeat_func
        self.verbose = int(verbose)

        # state
        self.wait: int = 0
        self.lr_counter: int = 0
        self.skip_lr_monitoring: bool = False
        self.best_epoch: int = -1
        self.best_weights: tuple[tf.Tensor, ...] | None = None
        self.best_metric: float = np.nan
        self.best_metric_with_previous_lr: float = np.nan
        self.monitor_op: Callable[[float, float], bool] | None = None
        self.repeat_counter: int = 0

        self._reset()

    def _reset(self) -> None:
        self.wait = 0
        self.lr_counter = 0
        self.skip_lr_monitoring = False
        self.repeat_counter = 0

        self._reset_best()

    def _reset_before_repeat(self) -> None:
        self.wait = 0
        self.lr_counter = 0

    def _reset_best(self) -> None:
        self.wait = 0
        self.best_epoch = -1
        self.best_weights = None

        if self.mode == "min":
            self.best_metric = np.inf
            self.best_metric_with_previous_lr = np.inf
            self.monitor_op = lambda cur, best: (best - cur) > self.min_delta
        else:  # "max"
            self.best_metric = -np.inf
            self.best_metric_with_previous_lr = -np.inf
            self.monitor_op = lambda cur, best: (cur - best) > self.min_delta

    def _reset_for_fine_tuning(self) -> None:
        self.wait = 0
        self.skip_lr_monitoring = False
        self.repeat_counter = 0

    def on_train_begin(self, logs: dict[str, Any] | None = None) -> None:
        self._reset()

    def on_epoch_end(self, epoch: int, logs: dict[str, Any] | None = None) -> None:
        # add the current learning rate to the logs
        logs = logs or {}
        logs["lr"] = tf.keras.backend.get_value(self.model.optimizer.lr)

        # do nothing when no metric is available yet
        value = self.get_monitor_value(logs)
        if value is None:
            return

        # helper to get a newline only for the first invocation
        nls = {"nl": "\n"}
        nl = lambda: nls.pop("nl", "")

        # new best value?
        if self.best_metric is None or self.monitor_op(value, self.best_metric):
            self.best_metric = value
            self.best_weights = self.model.get_weights()
            self.best_epoch = epoch
            self.wait = 0
            if self.verbose >= 2:
                print_msg(f"{nl()}{self.__class__.__name__}: recorded new best value of {value:.5f}")
            logs["last_best"] = 0
            return
        logs["last_best"] = int(epoch - self.best_epoch)

        # no improvement, increase wait counter
        self.wait += 1

        #
        # lr monitoring
        #

        if not self.skip_lr_monitoring:
            # do nothing if lr is not yet to be monitored
            if epoch < self.lr_start_epoch:
                self.wait = 0
                return

            # within patience?
            if self.wait <= self.lr_patience:
                return

            # drop?
            if (
                self.best_metric_with_previous_lr is None or
                self.monitor_op(self.best_metric, self.best_metric_with_previous_lr)
            ):
                # yes, drop
                logs["lr"] *= self.lr_factor
                tf.keras.backend.set_value(self.model.optimizer.lr, logs["lr"])
                self.lr_counter += 1
                self.wait = 0
                if self.verbose >= 1:
                    msg = (
                        f"{nl()}{self.__class__.__name__}: reducing learning rate to {logs['lr']:.2e} "
                        f"(reduction {self.lr_counter}), best metric is {self.best_metric:.5f}"
                    )
                    if self.best_metric_with_previous_lr not in (None, np.inf, -np.inf):
                        msg += f", was {self.best_metric_with_previous_lr:.5f} with previous learning rate"
                    print_msg(msg)
                self.best_metric_with_previous_lr = self.best_metric
            else:
                # last drop had already no effect, stop monitoring for lr drops
                self.skip_lr_monitoring = True
                self.wait = 0
                if self.verbose >= 1:
                    print_msg(
                        f"{nl()}{self.__class__.__name__}: last learning rate reduction had no effect, "
                        f"from now on checking early stopping with patience {self.es_patience}",
                    )
            return

        #
        # early stopping monitoring
        #

        # do nothing if es is not yet to be monitored
        if epoch < self.es_start_epoch:
            self.wait = 0
            return

        # within patience?
        if self.wait <= self.es_patience:
            return

        # repeat cycling?
        if callable(self.repeat_func) and self.repeat_func(self, logs):
            # yes, repeat
            self.repeat_counter += 1
            self._reset_before_repeat()
            if self.verbose >= 1:
                print_msg(
                    f"{nl()}{self.__class__.__name__}: repeat_func triggered repitition of lr and es cycle",
                )
            return

        # stop training completely
        self.model.stop_training = True
        if self.verbose >= 1:
            print_msg(f"{nl()}{self.__class__.__name__}: early stopping triggered")

    def on_train_end(self, logs: dict[str, Any] | None = None) -> None:
        self.restore_best_weights()

    def restore_best_weights(self) -> bool:
        if self.best_weights is None:
            return False
        self.model.set_weights(self.best_weights)
        if self.verbose >= 1:
            print_msg(
                f"{self.__class__.__name__}: recovered best weights from epoch {self.best_epoch + 1}, "
                f"best metric was {self.best_metric:.5f}",
            )

    def get_monitor_value(self, logs: dict[str, Any]) -> float | int:
        logs = logs or {}
        value = logs.get(self.monitor)
        if value is None:
            print_msg(f"{self.__class__.__name__}: metric '{self.monitor}' not available, found {','.join(list(logs))}")
        return value


class LivePlotWriter(tf.keras.callbacks.Callback):

    def __init__(
        self,
        log_dir: str,
        class_names: list[str],
        validate_every: int = 1,
        name="confusion",
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        # attributes
        self.log_dir: str = log_dir
        self.class_names: list[str] = class_names
        self.validate_every: int = validate_every

        # state
        self.file_writer: tf.summary.SummaryWriter = tf.summary.create_file_writer(os.path.join(log_dir, "validation"))
        self.counter: int = 0

    def on_test_end(self, logs: dict[str, Any] | None = None) -> None:
        self.counter += 1

        if (
            getattr(self.model, "buffer_y", None) is None or
            getattr(self.model, "buffer_y_pred", None) is None or
            getattr(self.model, "buffer_weight", None) is None
        ):
            if self.counter == 1:
                print_msg(
                    f"\n{self.__class__.__name__} requires model.buffer_y, model.buffer_y_pred and buffer_weight to be "
                    "set, not writing summary images",
                )
            return

        # get data
        y = self.model.buffer_y.numpy()
        y_pred = self.model.buffer_y_pred.numpy()
        weight = self.model.buffer_weight[:, 0].numpy()

        # confusion matrix
        true_classes = np.argmax(y, axis=1)
        pred_classes = np.argmax(y_pred, axis=1)
        cm = sklearn.metrics.confusion_matrix(true_classes, pred_classes, sample_weight=weight, normalize="true")
        cm_image = fig_to_image_tensor(plot_confusion_matrix(cm, self.class_names, colorbar=False)[0])

        # output distributions
        out_imgs = [
            fig_to_image_tensor(plot_class_outputs(y_pred, y, i, self.class_names)[0])
            for i in range(len(self.class_names))
        ]

        with self.file_writer.as_default():
            step = self.counter * self.validate_every
            tf.summary.image("epoch_confusion_matrix", cm_image, step=step)
            for i, img in enumerate(out_imgs):
                tf.summary.image(f"epoch_output_distribution_{self.class_names[i]}", img, step=step)
            self.file_writer.flush()


class CollectGarbage(tf.keras.callbacks.Callback):

    def on_test_end(self, logs: dict[str, Any] | None = None) -> None:
        print(psutil.Process(os.getpid()).memory_info().rss)
        gc.collect()
        print(psutil.Process(os.getpid()).memory_info().rss)


class EmbeddingEncoder(tf.keras.layers.Layer):

    def __init__(self, expected_inputs, keys_dtype=tf.int32, values_dtype=tf.int32, **kwargs):
        super().__init__(**kwargs)

        self.expected_inputs = expected_inputs
        self.keys_dtype = keys_dtype
        self.values_dtype = values_dtype

        self.n_inputs = len(expected_inputs)
        self.tables = []

    def get_config(self):
        config = super().get_config()
        config["expected_inputs"] = self.expected_inputs
        config["keys_dtype"] = self.keys_dtype
        config["values_dtype"] = self.values_dtype
        return config

    def build(self, input_shape):
        for i, keys in enumerate(self.expected_inputs):
            keys = tf.constant(keys, dtype=self.keys_dtype)
            offset = sum(map(len, self.expected_inputs[:i]))
            values = tf.constant(list(range(len(keys))), dtype=self.values_dtype) + offset
            table = tf.lookup.StaticHashTable(tf.lookup.KeyValueTensorInitializer(keys, values), -1)
            self.tables.append(table)

        return super().build(input_shape)

    def call(self, x):
        return tf.concat(
            [
                self.tables[i].lookup(x[..., i:i + 1])
                for i in range(self.n_inputs)
            ],
            axis=1,
        )


class BetterModel(tf.keras.Model):

    def compile(self, **kwargs):
        metrics = kwargs.pop("metrics", None)
        weighted_metrics = kwargs.pop("weighted_metrics", None)
        from_serialized = kwargs.get("from_serialized", False)

        # metrics must be lists or tuples when set
        if metrics is not None:
            if not isinstance(metrics, (list, tuple)):
                raise TypeError(
                    "Type of `metrics` argument not understood. "
                    "Expected a list or tuple, found: {}".format(type(metrics)),
                )
            metrics = list(metrics)
        if weighted_metrics is not None:
            if not isinstance(weighted_metrics, (list, tuple)):
                raise TypeError(
                    "Type of `weighted_metrics` argument not understood. "
                    "Expected a list or tuple, found: {}".format(type(weighted_metrics)),
                )
            weighted_metrics = list(weighted_metrics)

        # let super do the rest
        super().compile(**kwargs)

        # reset the metrics container
        self.compiled_metrics = CustomMetricsContainer(
            metrics,
            weighted_metrics,
            output_names=self.output_names,
            from_serialized=from_serialized,
            # mesh=None if self._layout_map is None else self._layout_map.get_default_mesh(),
        )

    def compute_metrics(self, x, y, y_pred, sample_weight):
        self.compiled_metrics.update_state(x, y, y_pred, sample_weight)
        return self.get_metrics_result()

    # def compute_loss(self, x=None, y=None, y_pred=None, sample_weight=None):
    #     pass


class CustomMetricsContainer(compile_utils.MetricsContainer):

    def build(self, x, y_true, y_pred):
        # this is fine as long as there is single inheritance
        compile_utils.Container.build(self, y_pred)
        self._built = False

        # setup metrics, optionally set names, etc,
        self._metrics = tuple(self._user_metrics or ())
        self._weighted_metrics = tuple(self._user_weighted_metrics or ())

        # finalize as done in super class
        self._create_ordered_metrics()
        self._built = True

    def update_state(self, x, y_true, y_pred, sample_weight=None):
        if not self._built:
            self.build(x, y_true, y_pred)

        # let metrics update their state
        for metric in self._metrics:
            metric.update_state(x, y_true, y_pred, sample_weight=None)
        for metric in self._weighted_metrics:
            metric.update_state(x, y_true, y_pred, sample_weight=sample_weight)

    def _create_ordered_metrics(self):
        self._metrics_in_order = list(self._metrics) + list(self._weighted_metrics)
