import tensorflow as tf 

@tf.function
def test_graph():
    x = tf.reshape((), (0, ))
    return x

b = test_graph()
b
breakpoint()
