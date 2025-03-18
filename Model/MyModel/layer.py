import numpy as np
import tensorflow as tf
import math

def attn_head(seq, seq_v, dis, activation, in_drop=0.0, coef_drop=0.0, rho=math.sqrt(math.pi), sigma=1/math.sqrt(2),residual=False):
    with tf.name_scope('my_attn'):
        if in_drop != 0.0:
            seq = tf.nn.dropout(seq, 1.0 - in_drop)

        #seq_fts = tf.layers.conv1d(seq, out_sz, 1, use_bias=False)

        # simplest self-attention possible
        logits = tf.nn.relu(tf.matmul(seq, tf.transpose(seq_v, [1,0])))
        dis_sqrt = tf.abs(tf.sqrt(dis))
        log_norm = (rho*dis_sqrt/((sigma*math.sqrt(2*math.pi))))*tf.pow(dis,1/(2*np.square(sigma)))
        # log_norm_shape = tf.shape(log_norm)
        # logits_shape = tf.shape(logits)
        logits_2 = tf.multiply(log_norm,logits)
        coefs = tf.nn.softmax(logits)
        coefs_v = tf.nn.softmax(tf.transpose(logits_2,[1,0]))
        #coefs = bias_mat * coefs
        if coef_drop != 0.0:
            coefs = tf.nn.dropout(coefs, 1.0 - coef_drop)
        if in_drop != 0.0:
            seq_fts = tf.nn.dropout(seq_fts, 1.0 - in_drop)

        ret = tf.matmul(coefs, seq_v)
        ret_v = tf.matmul(coefs_v, seq)
        # residual connection
        if residual:
            ret = ret + seq
            ret_v = ret_v + seq_v

        return activation(ret), activation(ret_v)  # activation
