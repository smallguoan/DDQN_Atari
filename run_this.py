import gym
from gym.wrappers import Monitor
import numpy as np
import tensorflow as tf
from utils import utils
from utils import StateProcessor
from model import Qnet
import sys
import itertools

#Initial all objects
tf.reset_default_graph()
sess=tf.Session()
target_net=Qnet(name_scope="target_net")
eval_net=Qnet(name_scope="eval_net",summaries_dir="./summaries")
state_processor=StateProcessor()
tools=utils(sess,500000,32)
total_step=tf.Variable(0,trainable=False,name="global_step")
sess.run(tf.global_variables_initializer())
env=gym.envs.make("Breakout-v0")
epsilon_decay=np.linspace(1,0.1,500000)

saver = tf.train.Saver()
# Load a previous checkpoint if we find one
latest_checkpoint = tf.train.latest_checkpoint("./check_point")
if latest_checkpoint:
    print("Loading model checkpoint {}...\n".format(latest_checkpoint))
    saver.restore(sess, latest_checkpoint)

#Initial the memory
observation=env.reset()
observation=state_processor.process(sess=sess,state=observation)
state=np.stack([observation]*4,axis=2)
#print(state.shape)

for i in range(1,50000):
    action=eval_net.choose_action(sess,state,1.0)
    state_,reward,done,_=env.step(action)
    state_ = state_processor.process(sess,state_)
    state_ = np.append(state[:, :, 1:], np.expand_dims(state_, 2), axis=2)
    tools.store_transition(state,action,reward,state_,done)
    if done:
        observation = env.reset()
        observation = state_processor.process(sess,observation)
        state = np.stack([observation] * 4, axis=2)
    else:
        state=state_
    print("\rInit the memo {}/{}".format(i,50000),end="")
    sys.stdout.flush()

env = Monitor(env,
                  directory="./exp",
                  resume=True,
                  video_callable=lambda count: count % 50 ==0)

#Main loop
total_t=sess.run(tf.train.get_global_step())
for episode_i in range(1,10000):
    saver.save(sess, "./check_point/check_point.ckpt")
    observation=env.reset()
    observation=state_processor.process(sess,observation)
    state=np.stack([observation]*4,axis=2)
    ep_reward=0
    loss=None
    for i in itertools.count():
        epsilon = epsilon_decay[min(total_t, 500000 - 1)]
        episode_summary = tf.Summary()
        episode_summary.value.add(simple_value=epsilon, tag="epsilon")
        eval_net.summary_writer.add_summary(episode_summary, total_t)
        # choose action
        #epsilon=epsilon_decay[min(total_t,500000-1)]
        action=eval_net.choose_action(sess,state,epsilon=epsilon)
        # step
        next_state,reward,done,_=env.step(action)
        next_state=state_processor.process(sess,next_state)
        next_state = np.append(state[:, :, 1:], np.expand_dims(next_state, 2), axis=2)
        # Store
        tools.store_transition(state,action,reward,next_state,done)
        # Learn (DDQN method)
        # Before learn, confirm whether we should replace target_net
        if total_t%10000==0:
            tools.replace_net_param(sess,target_net,eval_net)
        states_batch, action_batch, reward_batch, next_states_batch, done_batch=tools.sample_memory()

        q_values_next = eval_net.prediction(sess, next_states_batch)
        best_actions = np.argmax(q_values_next, axis=1)
        q_values_next_target=target_net.prediction(sess,next_states_batch)
        q_selected=q_values_next_target[np.arange(32),best_actions]
        q_target=reward_batch+np.invert(done).astype(np.float32)*0.99*q_selected
        loss=eval_net.learn(sess,q_target,states_batch,action_batch)

        # For next step
        if done:
            break
        ep_reward+=reward
        state=next_state
        total_t+=1
        print("\r episode_i: {}/{},Reward: {},total_step: {} epsilon: {} loss: {}".format(episode_i, 10000, ep_reward,
                                                                                          total_t, epsilon, loss),end="")
        sys.stdout.flush()


