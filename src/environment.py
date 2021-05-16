import types
from typing import Optional

import tensorflow as tf
import tensorflow_probability as tfp
from tf_agents.bandits.environments import bandit_tf_environment as bte
from tf_agents.specs import tensor_spec
from tf_agents.trajectories import time_step
from tf_agents.utils import eager_utils
from tf_agents.typing import types
import numpy as np
from data import obd

tfd = tfp.distributions


class ClassificationRewardsBanditEnvironment(bte.BanditTFEnvironment):
    """An environment based on an arbitrary classification problem."""

    def __init__(
        self,
        dataset: tf.data.Dataset,
        reward_distribution: types.Distribution,
        batch_size: types.Int,
        label_dtype_cast: Optional[tf.DType] = None,
        shuffle_buffer_size: Optional[types.Int] = None,
        repeat_dataset: Optional[bool] = True,
        prefetch_size: Optional[types.Int] = None,
        seed: Optional[types.Int] = None,
    ):
        """Initialize `ClassificationRewardsBanditEnvironment`.

        Args:
          dataset: a `tf.data.Dataset` consisting of two `Tensor`s, [inputs, labels]
            where inputs can be of any shape, while labels are integer class labels.
            The label tensor can be of any rank as long as it has 1 element.
          reward_distribution: a `tfd.Distribution` with event_shape
            `[num_classes, num_actions]`. Entry `[i, j]` is the reward for taking
            action `j` for an instance of class `i`.
          batch_size: if `dataset` is batched, this is the size of the batches.
          label_dtype_cast: if not None, casts dataset labels to this dtype.
          shuffle_buffer_size: If None, do not shuffle.  Otherwise, a shuffle buffer
            of the specified size is used in the environment's `dataset`.
          repeat_dataset: Makes the environment iterate on the `dataset` once
            avoiding `OutOfRangeError:  End of sequence` errors when the environment
            is stepped past the end of the `dataset`.
          prefetch_size: If None, do not prefetch.  Otherwise, a prefetch buffer
            of the specified size is used in the environment's `dataset`.
          seed: Used to make results deterministic.
          name: The name of this environment instance.
        Raises:
          ValueError: if `reward_distribution` does not have an event shape with
            rank 2.
        """

        # Computing `action_spec`.
        event_shape = reward_distribution.event_shape
        if len(event_shape) != 2:
            raise ValueError(
                "reward_distribution must have event shape of rank 2; "
                "got event shape {}".format(event_shape)
            )
        _, num_actions = event_shape
        action_spec = tensor_spec.BoundedTensorSpec(
            shape=(), dtype=tf.int32, minimum=0, maximum=num_actions - 1, name="action"
        )
        output_shapes = tf.compat.v1.data.get_output_shapes(dataset)

        # Computing `time_step_spec`.
        if len(output_shapes) != 3:
            raise ValueError(
                "Dataset must have exactly two outputs; got {}".format(
                    len(output_shapes)
                )
            )
        context_shape = output_shapes[0]
        context_dtype, lbl_dtype, reward_dtype = tf.compat.v1.data.get_output_types(
            dataset
        )
        if label_dtype_cast:
            lbl_dtype = label_dtype_cast
        observation_spec = tensor_spec.TensorSpec(
            shape=context_shape, dtype=context_dtype
        )
        time_step_spec = time_step.time_step_spec(observation_spec)

        super(ClassificationRewardsBanditEnvironment, self).__init__(
            action_spec=action_spec,
            time_step_spec=time_step_spec,
            batch_size=batch_size,
        )

        if shuffle_buffer_size:
            dataset = dataset.shuffle(
                buffer_size=shuffle_buffer_size,
                seed=seed,
                reshuffle_each_iteration=True,
            )
        if repeat_dataset:
            dataset = dataset.repeat()
        dataset = dataset.batch(batch_size, drop_remainder=True)
        if prefetch_size:
            dataset = dataset.prefetch(prefetch_size)
        self._data_iterator = eager_utils.dataset_iterator(dataset)
        self._current_label = tf.compat.v2.Variable(
            tf.zeros(batch_size, dtype=lbl_dtype)
        )
        self._current_reward = tf.compat.v2.Variable(
            tf.zeros(batch_size, dtype=reward_dtype)
        )
        self._previous_label = tf.compat.v2.Variable(
            tf.zeros(batch_size, dtype=lbl_dtype)
        )
        self._previous_reward = tf.compat.v2.Variable(
            tf.zeros(batch_size, dtype=reward_dtype)
        )
        self._reward_distribution = reward_distribution
        self._label_dtype = lbl_dtype

        reward_means = self._reward_distribution.mean()
        self._optimal_action_table = tf.argmax(
            reward_means, axis=1, output_type=self._action_spec.dtype
        )
        self._optimal_reward_table = tf.reduce_max(reward_means, axis=1)

    def _observe(self) -> types.NestedTensor:
        context, lbl, reward = eager_utils.get_next(self._data_iterator)
        self._previous_label.assign(self._current_label)
        self._previous_reward.assign(self._current_reward)
        self._current_label.assign(
            tf.reshape(tf.cast(lbl, dtype=self._label_dtype), shape=[self._batch_size])
        )
        self._current_reward.assign(
            tf.reshape(
                tf.cast(reward, dtype=self._label_dtype), shape=[self._batch_size]
            )
        )
        return tf.reshape(
            context, shape=[self._batch_size] + self._time_step_spec.observation.shape
        )

    def _apply_action(self, action: types.NestedTensor) -> types.NestedTensor:
        action = tf.reshape(action, shape=[self._batch_size] + self._action_spec.shape)
        rewards = tf.cast(self._current_reward, tf.int32) * tf.cast(
            self._current_label == action, tf.int32
        )
        return tf.cast(rewards, tf.float32)

    def compute_optimal_action(self) -> types.NestedTensor:
        return self._previous_label

    def compute_optimal_reward(self) -> types.NestedTensor:
        return tf.cast(self._previous_reward, tf.float32)


def make_env(
    obd_path,
    output_shape,
    batch_size,
    sample_rate=1.0,
    campaigns={"random": ["men"]},
    nactions=33,
    load_batch_size=50000,
):
    def sample(*x):
        return tf.random.uniform([], maxval=1.0) <= tf.constant(sample_rate)

    obd_ds = obd.get_dataset(obd_path, campaigns, load_batch_size)

    def f(*x):
        context = tf.cast(tf.concat((x[2:3], x[9:43]), axis=0), tf.float32)
        context = tf.cast(
            tf.concat((context, x[5], x[6], x[7], x[8], x[43], x[44]), axis=0),
            tf.float32,
        )
        context.set_shape((output_shape))
        arm = x[1]
        reward = x[3] * 10
        arm.set_shape(())
        reward.set_shape(())
        return context, arm, reward

    obd_labeled_ds = obd_ds.map(f).filter(sample)

    rewards_dist = np.eye(nactions, dtype=np.float32) * 5
    reward_distribution = tfd.Independent(
        tfd.Deterministic(rewards_dist), reinterpreted_batch_ndims=2
    )
    environment = ClassificationRewardsBanditEnvironment(
        obd_labeled_ds, reward_distribution, batch_size
    )
    return environment
