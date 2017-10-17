import argparse
import os
import sys

import threading
import multiprocessing
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.contrib.slim as slim
import scipy.signal
from AC_Network import AC_Network
from Worker import Worker

from random import choice
from time import sleep
from time import time

import gym


parser = argparse.ArgumentParser(description="Run commands")
parser.add_argument('-w', '--num-workers', default=1, type=int,
                    help="Number of workers")

parser.add_argument('-e', '--env-id', type=str, default="vizdoom",
                    help="Environment id")

#parser.add_argument('-m', '--load-model', type=bool, default=False,
#                    help="Load old model")

# parser.add_argument('-l', '--log-dir', type=str, default="/tmp/pong",
#                    help="Log directory path")
# parser.add_argument('-n', '--dry-run', action='store_true',
#                    help="Print out commands rather than executing them")

# Add visualise tag
parser.add_argument('--visualise', action='store_true',
                    help="Visualise the gym environment by running env.render() between each timestep")

def run():
    
    tf.reset_default_graph()
    
    
# Create a directory to save episode playback gifs to
if not os.path.exists('./frames'):
    os.makedirs('./frames')

with tf.device("/cpu:0"):
    model_path = './model'
    if not os.path.exists(model_path):  # create model if it doesn't exist
        os.makedirs(model_path)
    
    args = parser.parse_args()
    print(args)
    env = gym.make(args.env_id)
    s_size = env.observation_space.shape # observation space
    print(s_size)
    a_size = env.action_space.n  # action space
    print(a_size)
    #s_size = 100800 # Observations are greyscale frames of 84 * 84 * 1
    #a_size = 3  # Agent can move Left, Right, or Fire
    global_episodes = tf.Variable(0, dtype=tf.int32, name='global_episodes', trainable=False)
    trainer = tf.train.AdamOptimizer(learning_rate=1e-4)
    master_network = AC_Network(s_size, a_size, 'global', None)  # Generate global network
    # num_workers = multiprocessing.cpu_count() # Set workers ot number of available CPU threads
    num_workers = args.num_workers
    workers = []
    # Create worker classes
    for i in range(num_workers):
        workers.append(Worker(env,i,s_size,a_size,trainer,model_path,global_episodes))
        saver = tf.train.Saver(max_to_keep=5)


with tf.Session() as sess:
    
    load_model = False

    max_episode_length = 300
    gamma = .99  # discount rate for advantage estimation and reward discounting
    
    coord = tf.train.Coordinator()
    if load_model == True:
        print ('Loading Model...')
        ckpt = tf.train.get_checkpoint_state(model_path)
        saver.restore(sess,ckpt.model_checkpoint_path)
    else:
        sess.run(tf.global_variables_initializer())
        
    # This is where the asynchronous magic happens.
    # Start the "work" process for each worker in a separate threat.
    worker_threads = []
    for worker in workers:
        worker_work = lambda: worker.work(max_episode_length, gamma, sess, coord, saver)
        t = threading.Thread(target=(worker_work))
        t.start()
        sleep(0.5)
        worker_threads.append(t)
    coord.join(worker_threads)
   


if __name__ == "__main__":
    run()
