import tensorflow as tf
import numpy as np
import os


class Qnet(object):
    def __init__(self,name_scope,summaries_dir=None):
        assert name_scope=="eval_net" or "target_net"
        self.name_scope=name_scope
        self.summary_writer=None
        with tf.variable_scope(name_scope):
            self._build_model()
            if summaries_dir:
                summary_dir = os.path.join(summaries_dir, "summaries_{}".format(name_scope))
                if not os.path.exists(summary_dir):
                    os.makedirs(summary_dir)
                self.summary_writer = tf.summary.FileWriter(summary_dir)

    def _build_model(self):

        self.X=tf.placeholder(shape=[None,84,84,4],dtype=tf.uint8,name="X")  #Input State
        self.Y=tf.placeholder(shape=[None],dtype=tf.float32,name="Y")  #TD_target_value
        self.action=tf.placeholder(shape=[None],dtype=tf.int32,name="actions")  #Actions_ID

        X = tf.to_float(self.X) / 255.0
        batch_size = tf.shape(self.X)[0]

        conv1 = tf.layers.conv2d(
            X, 32, 8, 4, activation=tf.nn.relu)
        conv2 = tf.layers.conv2d(
            conv1, 64, 4, 2, activation=tf.nn.relu)
        conv3 = tf.layers.conv2d(
            conv2, 64, 3, 1, activation=tf.nn.relu)

        # Fully connected layers
        flattened = tf.layers.flatten(conv3)
        fc1 = tf.contrib.layers.fully_connected(flattened, 512)
        self.predictions = tf.contrib.layers.fully_connected(fc1,4)

        gather_indices = tf.range(batch_size) * tf.shape(self.predictions)[1] + self.action
        self.action_predictions = tf.gather(tf.reshape(self.predictions, [-1]), gather_indices)

        losses=tf.squared_difference(self.Y,self.action_predictions)
        self.loss=tf.reduce_mean(losses)

        if self.name_scope=="eval_net":
            with tf.variable_scope("opt"):
                self.optimizer=tf.train.RMSPropOptimizer(0.00025,0.99,0.0,1e-6)
                self.train_op=self.optimizer.minimize(loss=self.loss,global_step=tf.train.get_global_step())

        self.summaries = tf.summary.merge([
            tf.summary.scalar("loss", self.loss),
            tf.summary.histogram("loss_hist", losses),
            tf.summary.histogram("q_values_hist", self.predictions),
            tf.summary.scalar("max_q_value", tf.reduce_max(self.predictions))
        ])

    def prediction(self,sess,s):
        return sess.run(self.predictions,feed_dict={self.X:s})

    def choose_action(self,sess,s,epsilon):
        if self.name_scope=="eval_net":
            A=np.ones(4,dtype=float)*epsilon/4.0
            s=np.expand_dims(s,0)
            q_value=self.prediction(sess,s)[0]
            act=np.argmax(q_value)
            A[act]+=(1-epsilon)
            action=np.random.choice(len(A),p=A)

            return action
        else:
            return None

    def learn(self,sess,y,x,a):
        summaries, global_step, _, loss = sess.run(
            [self.summaries, tf.train.get_global_step(), self.train_op, self.loss],
            feed_dict={self.X: x, self.Y: y, self.action: a})

        return loss

