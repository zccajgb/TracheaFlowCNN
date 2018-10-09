from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import tensorflow as tf
import image_gen_v2 as dg
import argparse
import sys
import os
from datetime import datetime as dt
from itertools import chain,repeat,cycle
import cnn_obj as net
from network_shape import gen_shape,gen_dense_shape,disc_shape,disc_dense_shape
from tensorflow.python import debug as tf_debug
#from memory_profiler import profile
#from pprint import pprint
parser = argparse.ArgumentParser()
parser.parse_args()
FLAGS=None
''' 
Define variables and initialise  data adwritefile
'''
Num_trachea=100
#Num_trachea=20
Num_points=32
y_points=64
BATCH_SIZE=100
#BATCH_SIZE=10
BUFFER_SIZE=200
EPOCHS=50000	
EPOCHS=5000
EPOCHS=1	
readout=100
LEARNING_RATE=0.0001
DROPOUT=0.2
n_batches=Num_trachea//BATCH_SIZE
#write file
writefile=open('cloop.txt','w')
now=str(dt.now())
writefile.write(now)
writefile.write('\n')
writefile.close()
#initialise data
P0=10
filtfilt=300

z1,x1,_=dg.scalar_bernoulli(Num_trachea,Num_points,poly_order=P0) #generates data
z1=np.expand_dims(z1,-1)
z2,x2,_=dg.scalar_bernoulli(Num_trachea,Num_points,poly_order=P0) #generates data	
z2=np.expand_dims(z2,-1)
y1=np.zeros(Num_trachea)
z_test,x_test,_=dg.scalar_bernoulli(10,Num_points,poly_order=P0)
z_test=np.expand_dims(z_test,-1)
gen_loss_vector=[]
disc_loss_vector=[]
#batch_size=tf.placeholder(tf.int64,name="batch_size")
'''
Define Network Shapes
'''

#G_Conv_shape=[[32,4,stride1,'valid','channels_last',1,tf.nn.leaky_relu,True,None,tf.zeros_initializer(),None,None,None,None,None,True,"gen_conv1"],[32,4,stride1,'valid','channels_last',1,tf.nn.leaky_relu,True,None,tf.zeros_initializer(),None,None,None,None,None,True,"gen_conv2"]]
#G_Dense=[[32,tf.nn.leaky_relu,True,None,tf.zeros_initializer(),None,None,None,None,None,True,'gen_Dense_1',None]]
#G_Deconv_shape=[[[4,32,1],[BATCH_SIZE,1,32],stride2,'VALID','NWC','gen_Deconv1']]
#G_Deconv_out=[[4,1,32],[BATCH_SIZE,Num_points,1],stride1,'VALID','NWC','Generator_Output']
#D_Conv_shape=[[32,4,2,'valid','channels_last',1,tf.nn.leaky_relu,True,None,tf.zeros_initializer(),None,None,None,None,None,True,"disc_conv1",None],[32,4,2,'valid','channels_last',1,tf.nn.leaky_relu,True,None,tf.zeros_initializer(),None,None,None,None,None,True,"disc_conv2",None]]
#D_Dense=[[32,tf.nn.leaky_relu,True,None,tf.zeros_initializer(),None,None,None,None,None,True,'disc_Dense_1',None]]
#Readout=[1,tf.nn.sigmoid,True,None,tf.zeros_initializer(),None,None,None,None,None,True,'disc_Readout',None]
G_Conv_shape,G_Deconv_shape,G_Deconv_out=gen_shape(num_layers=2,filters=filtfilt,strides=1,BATCH_SIZE=BATCH_SIZE,kernel_size=4,Num_points=1)
G_Dense=gen_dense_shape(num_layers=1,Units=filtfilt)
D_Conv_shape=disc_shape(num_layers=2,filters=filtfilt,kernel_size=3,strides=1)
D_Dense,Readout=disc_dense_shape(num_layers=1,Units=filtfilt)
'''
Create Network
'''
generator=net.network('g',Num_points,False,y_points)
xf,zf=net.build_gen(generator,G_Conv_shape,G_Dense,DROPOUT,G_Deconv_shape,G_Deconv_out)

generator.train(LEARNING_RATE,zf)
''' Define Saver'''
saver=tf.train.Saver()
''' Run Session'''

with tf.Session() as sess:
	options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
	run_metadata=tf.RunMetadata()
	writer=tf.summary.FileWriter('logdir',sess.graph)
	sess.run(tf.global_variables_initializer())#initialise variables
	#sess = tf_debug.LocalCLIDebugWrapperSession(sess)
	#sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)
	#TRAIN MODEL
	sess.run(generator.dataset_init_op,feed_dict={generator.x_:x1,generator.batch_size:BATCH_SIZE,generator.y_:z1})#initialise iterator
	#SAVE MODEL
	for i in range(EPOCHS):#loop over epochs
		gen_loss=0
		for _ in range(n_batches):
			_,loss_value=sess.run([generator.train_op,generator.loss],feed_dict={generator.training:True})
			gen_loss+=loss_value
		gen_loss_vector.append(gen_loss)

		if i%readout==0:
	#		print("Iter:{},Disc_Loss:{:.4f},Gen_Loss:{:.4f}".format(i,disc_loss/n_batches,gen_loss/n_batches))
			
			writefile=open('cloop.txt','a')
			writefile.write("Iter:{},Gen_Loss:{:.4f}\n".format(i,gen_loss/n_batches))
			writefile.close()
			#print(sess.run(y_est,feed_dict={training:False}))
			tf.summary.scalar('Gen_Loss',gen_loss)
			merged=tf.summary.merge_all()
			summary=sess.run(merged,feed_dict={generator.training:False},options=options,run_metadata=run_metadata)
			writer.add_run_metadata(run_metadata, str(i),global_step=i)
			writer.add_summary(summary,i)

	#SAVE MODEL
#	for var in tf.trainable_variables():
#		tf.summary.histogram(var.name,var)
	#print("Iter:{},Disc_Loss:{:.4f},Gen_Loss:{:.4f}".format(i,disc_loss/n_batches,gen_loss/n_batches))
	writefile=open('cloop.txt','a')
	writefile.write("Iter:{},Gen_Loss:{:.8f}\n".format(i,gen_loss/n_batches))
	writefile.close()
	saver.save(sess,'./save')

	
	#TEST MODEL
	sess.run(generator.dataset_init_op,feed_dict={generator.x_:x_test,generator.y_:z_test,generator.batch_size:x_test.shape[0]})
	#print('Test Loss:{:4f}'.format(sess.run(generator.gen_loss,feed_dict={generator.training:False,discriminator.training:False})))
	z_calc=sess.run(zf,feed_dict={generator.training:False})
	#error=np.mean(y_calc-y_test)
	#print(error)	

writefile=open('cloop.txt','a')
writefile.write("Done")
writefile.close()

Nettype='CNN Scalar'
Loss_Type='MSE'
Optimiser='ADAM'


writefile=open('Network_data.tsv','w')
string='ID,'+Nettype+'\t'+str(Num_points)+'\t'+str(True)+'\t'+str(True)+'\t'+str(True)+'\t'+str(EPOCHS)+'\t'+str(Num_trachea)+'\t'+str(BUFFER_SIZE)+'\t'+str(BATCH_SIZE)+'\t'+str(G_Conv_shape)+'\t'+str(G_Dense)+'\t'+str(DROPOUT)+'\t'+str(Readout)+'\t'+Loss_Type+'\t'+str(gen_loss)
writefile.write(string)
writefile.close()
	
