from ExperimentSuite import ExperimentSuite
import tensorflow as tf
from Vectorizer import Vectorizer
from datetime import datetime

DEFAULT_EPOCH = 75
DEFAULT_LAYERS = (512,)
DEFAULT_ACTIVATION = tf.nn.relu
DEFAULT_LOSS = "categorical_hinge"


if __name__ == "__main__":
    # TODO: Make your experiments here
    es = ExperimentSuite()
    v=Vectorizer(max_df=0.97,min_df=0.5)
    print datetime.now()
    v1=v.fit_transform(es.train_contents,"existance")
    print v1
    print datetime.now()