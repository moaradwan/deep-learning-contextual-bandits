import tensorflow as tf
from tf_agents.bandits.specs import utils as bandit_spec_utils
from tf_agents.metrics import tf_metric
from tf_agents.utils import common


class CumulativeRewardMetric(tf_metric.TFStepMetric):
    """Computes the cumulative reward"""

    def __init__(self, dtype: float = tf.float32, name="CumulativeRewardMetric"):
        self.dtype = dtype
        self.cumulative_reward = common.create_variable(
            initial_value=0, dtype=self.dtype, shape=(), name="cumulative_reward"
        )
        super(CumulativeRewardMetric, self).__init__(name=name)

    def call(self, trajectory):

        trajectory_reward = trajectory.reward
        if isinstance(trajectory.reward, dict):
            trajectory_reward = trajectory.reward[bandit_spec_utils.REWARD_SPEC_KEY]
        self.cumulative_reward.assign_add(tf.reduce_sum(trajectory_reward))
        return self.cumulative_reward

    def result(self):
        return tf.identity(self.cumulative_reward, name=self.name)
