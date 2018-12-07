import numpy as np
import tensorflow as tf
from collections import namedtuple
import random


class utils(object):
    def __init__(self,sess,memory_size,batch_size):
        self.sess=sess
        self.memory_size=memory_size
        self.Transition = namedtuple("Transition", ["state", "action", "reward", "next_state", "done"])
        self.replay_memo=[]
        self.batch_size=batch_size


    # def Preprocess(self,observation):
    #     #with tf.variable_scope("state_processor"):
    #     self.input_state = tf.placeholder(shape=[210, 160, 3], dtype=tf.uint8)
    #     self.output = tf.image.rgb_to_grayscale(self.input_state)
    #     self.output = tf.image.crop_to_bounding_box(self.output, 34, 0, 160, 160)
    #     self.output = tf.image.resize_images(
    #         self.output, [84, 84], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    #     self.output = tf.squeeze(self.output)
    #
    #     return self.sess.run(self.output,feed_dict={self.input_state:observation})


    def store_transition(self,s,a,r,s_,done):
        if len(self.replay_memo)==self.memory_size:
            self.replay_memo.pop(0)
        self.replay_memo.append(self.Transition(s,a,r,s_,done))

    def sample_memory(self):
        batch_memo=random.sample(self.replay_memo,self.batch_size)
        states_batch, action_batch, reward_batch, next_states_batch, done_batch = map(np.array, zip(*batch_memo))

        return states_batch,action_batch,reward_batch,next_states_batch,done_batch

    def replace_net_param(self,sess,target_net,eval_net):
        print("\nreplace the param")
        eval_net_param=[t for t in tf.trainable_variables() if t.name.startswith(eval_net.name_scope)]
        sorted(eval_net_param,key=lambda v:v.name)
        target_net_param = [t for t in tf.trainable_variables() if t.name.startswith(target_net.name_scope)]
        sorted(target_net_param, key=lambda v: v.name)

        update_op=[]
        for e_param,t_param in zip(eval_net_param,target_net_param):
            op=t_param.assign(e_param)
            update_op.append(op)

        sess.run(update_op)

class StateProcessor():
    """
    Processes a raw Atari images. Resizes it and converts it to grayscale.
    """
    def __init__(self):
        # Build the Tensorflow graph
        with tf.variable_scope("state_processor"):
            self.input_state = tf.placeholder(shape=[210, 160, 3], dtype=tf.uint8)
            self.output = tf.image.rgb_to_grayscale(self.input_state)
            self.output = tf.image.crop_to_bounding_box(self.output, 34, 0, 160, 160)
            self.output = tf.image.resize_images(
                self.output, [84, 84], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            self.output = tf.squeeze(self.output)

    def process(self, sess, state):
        """
        Args:
            sess: A Tensorflow session object
            state: A [210, 160, 3] Atari RGB State
        Returns:
            A processed [84, 84] state representing grayscale values.
        """
        return sess.run(self.output, { self.input_state: state })