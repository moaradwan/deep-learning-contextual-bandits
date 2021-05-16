from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import functools

import tensorflow_probability as tfp

from tf_agents.bandits.environments import environment_utilities as env_util
from tf_agents.bandits.metrics import tf_metrics as tf_bandit_metrics
from tf_agents.bandits.agents.examples.v2 import trainer

from metrics import CumulativeRewardMetric

tfd = tfp.distributions


def train(agent, environment, training_loops, steps_per_loop, output_path):
    optimal_reward_fn = functools.partial(
        env_util.compute_optimal_reward_with_classification_environment,
        environment=environment,
    )

    optimal_action_fn = functools.partial(
        env_util.compute_optimal_action_with_classification_environment,
        environment=environment,
    )
    regret_metric = tf_bandit_metrics.RegretMetric(optimal_reward_fn)
    cumulative_reward = CumulativeRewardMetric()
    suboptimal_arms_metric = tf_bandit_metrics.SuboptimalArmsMetric(optimal_action_fn)
    metrics = [regret_metric, suboptimal_arms_metric, cumulative_reward]
    from datetime import datetime

    t1 = datetime.now()
    trainer.train(
        root_dir=output_path,
        agent=agent,
        environment=environment,
        training_loops=training_loops,
        steps_per_loop=steps_per_loop,  # 452950//batch_size,
        additional_metrics=metrics,
    )
    t2 = datetime.now()
    print("Training time in minutes:")
    print((t2 - t1).total_seconds() / 60)
