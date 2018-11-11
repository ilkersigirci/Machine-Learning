from ExperimentSuite import ExperimentSuite
import tensorflow as tf

DEFAULT_EPOCH = 75
DEFAULT_LAYERS = (512,)
DEFAULT_ACTIVATION = tf.nn.relu
DEFAULT_LOSS = "categorical_hinge"


if __name__ == "__main__":
    # TODO: Make your experiments here
    es = ExperimentSuite()
