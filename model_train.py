# -*- coding: utf-8 -*-
"""
Created on Fri Apr 27 10:59:27 2018

@author: zhewei
"""

import tensorflow as tf
from dataLoad import loadDataset,getBatches
from model import Seq2SeqModel
from tqdm import tqdm
import math
import os

# http://blog.csdn.net/leiting_imecas/article/details/72367937
# tf定义了tf.app.flags，用于支持接受命令行传递参数，相当于接受argv。
'''
tf.app.flags.DEFINE_integer('rnn_size', 1024, 'Number of hidden units in each layer')
tf.app.flags.DEFINE_integer('num_layers', 2, 'Number of layers in each encoder and decoder')
tf.app.flags.DEFINE_integer('embedding_size', 1024, 'Embedding dimensions of encoder and decoder inputs')
tf.app.flags.DEFINE_float('learning_rate', 0.0001, 'Learning rate')
tf.app.flags.DEFINE_integer('batch_size', 128, 'Batch size')
tf.app.flags.DEFINE_integer('numEpochs', 30, 'Maximum # of training epochs')
tf.app.flags.DEFINE_integer('steps_per_checkpoint', 100, 'Save model checkpoint every this iteration')
tf.app.flags.DEFINE_string('model_dir', 'model/', 'Path to save model checkpoints')
tf.app.flags.DEFINE_string('model_name', 'chatbot.ckpt', 'File name used for model checkpoints')
FLAGS = tf.app.flags.FLAGS
'''
model_dir = os.path.join('model')
model_name = 'chatbot.ckpt'
rnn_size = 256
num_layers = 2
embedding_size = 64

learning_rate = 0.001
batch_size = 256
numEpochs = 5
steps_per_checkpoint = 1000



data_path = os.path.join('processed_data', 'trainFile.pkl')
#model_dir = 'model'
word2id, id2word, trainSamples = loadDataset(data_path)


with tf.Session() as sess:
    model = Seq2SeqModel(rnn_size=256,
                         num_layers=2,
                         embedding_size=64,
                         learning_rate=0.001,
                         word_to_idx=word2id,
                         mode='train',
                         use_attention=True,
                         beam_search=False,
                         beam_size=5,
                         max_gradient_norm=5.0
                         )

    # 如果存在已经保存的模型的话，就继续训练，否则，就重新开始
    
    ckpt = tf.train.get_checkpoint_state(model_dir)  
    if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
        pass     
        print('Reloading model parameters..')
        model.saver.restore(sess, ckpt.model_checkpoint_path)
    else:
        print('Created new model parameters..')
        sess.run(tf.global_variables_initializer())

    current_step = 0
    summary_writer = tf.summary.FileWriter(model_dir, graph=sess.graph)
    checkpoint_path = os.path.join('model', 'model.ckpt')
    
    for e in range(numEpochs):
        print("----- Epoch {}/{} -----".format(e + 1, numEpochs))
        batches = getBatches(trainSamples, batch_size)
        
        # Tqdm 是一个快速，可扩展的Python进度条，可以在 Python 长循环中添加一个进度提示信息，用户只需要封装任意的迭代器 tqdm(iterator)。
        for nextBatch in tqdm(batches, desc="Training"):
            loss, summary = model.train(sess, nextBatch)
            current_step += 1
            # save model every steps_per_checkpoint
            if current_step % steps_per_checkpoint == 0:
                perplexity = math.exp(float(loss)) if loss < 300 else float('inf')
                tqdm.write("----- Step %d -- Loss %.2f -- Perplexity %.2f" % (current_step, loss, perplexity))
                summary_writer.add_summary(summary, current_step)
        model.saver.save(sess, checkpoint_path)
                