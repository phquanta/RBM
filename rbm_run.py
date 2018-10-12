
import rbm
from imp import reload

import numpy as np
#import input_data
import tensorflow.examples.tutorials.mnist.input_data as input_data
#import Image
from PIL import Image
from utils import tile_raster_images
import random
import sys
import scipy.io as sio
import os
#from deap import base
#from deap import creator
#from deap import tools
#import deap as deap


reload(rbm)

alpha = 0.01
batchsize = 100
n_h=340
n_v=784
eachNLog=5000
eachNErr=50
wDecay=0.0001
hDecay=0.005
Nepochs=500


mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
trX, trY, teX, teY = mnist.train.images, mnist.train.labels, mnist.test.images,\
    mnist.test.labels



statCnt=[]

rbm_m1=rbm.rbm(batchsize,alpha,wDecay,hDecay, n_h,n_v)
#rbm_m2=rbm.rbm(batchsize,alpha, n_h,n_v, 'scope2')

#rbm_m2=rbm.rbm(batchsize,alpha,wDecay,hDecay, n_h,n_v, 'scope55')

#print (rbm_m1.sess.run(rbm_m1.err_sum,  feed_dict={rbm_m1.X: trX} ))    

#rbm_m1.X=trX
print (rbm_m1.err_sum(trX))    

#trX=(trX>0).astype(float)


#rbm_m1.load_weights("C:\\Users\\Andrei\\Documents\\UoGuelphResearch\\MileStones_MRR\\Runs\\GA_ES_DiffOnly_guided_R16_100iterPerGA\\weights-25000")

#print("Loaded")

#w=rbm_m1.sess.run(rbm_m1.rbm_w)
#hb=rbm_m1.sess.run(rbm_m1.rbm_hb)
#vb=rbm_m1.sess.run(rbm_m1.rbm_vb)
#sio.savemat('MRR1.mat', {'vishid':w,'visbiases':vb,'hidbiases':hb},dtype='double') 
#sio.savemat('MRR2.mat', {'vishid':np.array(w, dtype='float64'),'visbiases':np.array(vb, dtype='float64'),'hidbiases':np.array(hb, dtype='float64')})


#sys.exit("Exiting")




f= open('statistics.dat','w')
        


#rbm_m1.load_weights("C:\\Users\\Andrei\\Documents\\UoGuelphResearch\\May22_2018_GA_guided\\weights-8573")

#print("Loaded")

#sys.exit("Exiting")


i=0
#rbm_m1.save_pcd_fantasy(0)
for epochs in range(0,Nepochs):
    rbm_m1.it=0 
    #rbm_m1.save_pcd_fantasy(rbm_m1.i)
    for start, end in zip(
        range(0, len(trX), batchsize), range(batchsize, len(trX), batchsize)):
     
     
     batch = trX[start:end]
     rbm_m1.it=rbm_m1.it+1    
     
     n_w, _, _ =rbm_m1.batch_train_with_deps(batch)
 #    error = rbm_m1.sess.run(merged_summary_op,feed_dict={rbm_m1.X:trX})
     
 #    summary_writer.add_summary(error,epochs*len(trX)+i)
 #    np.random.shuffle(trX)
     if i % eachNLog == 0:
#         rbm_m1.save_fantasy_visual(i,batch)
         rbm_m1.save_pcd_fantasy(i)
         rbm_m1.save_weights_visual(i)
         rbm_m1.gen_v1_after_k_steps(trX[0:100],1,i) 
         rbm_m1.gen_v1_after_k_steps(trX[0:100],2,i)
         rbm_m1.gen_v1_after_k_steps(trX[0:100],10,i) 
         rbm_m1.gen_v1_after_k_steps(trX[0:100],20,i) 
         rbm_m1.gen_v1_after_k_steps(trX[0:100],30,i) 
         rbm_m1.gen_v1_after_k_steps(trX[0:100],40,i) 
         rbm_m1.gen_v1_after_k_steps(trX[0:100],50,i) 
         rbm_m1.gen_v1_after_k_steps(trX[0:100],100,i) 
         rbm_m1.gen_v1_after_k_steps(trX[0:100],1000,i) 
         rbm_m1.gen_v1_after_k_steps(trX[0:100],5000,i) 
 #        rbm_m1.save_weights(i)
         
 
         #error = rbm_m1.sess.run(merged_summary_op,feed_dict={rbm_m1.X:trX})
         #summary_writer.add_summary(error,epochs*len(trX)+i)
         #w=rbm_m1.sess.run(rbm_m1.rbm_w)
         #hb=rbm_m1.sess.run(rbm_m1.rbm_hb)
         #vb=rbm_m1.sess.run(rbm_m1.rbm_vb)
         sio.savemat(os.getcwd().split('/')[-1]+'_'+str(i)+'.mat', {'vishid':np.array(rbm_m1.rbm_w, dtype='float64'),'visbiases':np.array(rbm_m1.rbm_vb, dtype='float64'),'hidbiases':np.array(rbm_m1.rbm_hb, dtype='float64')})




     if  i % eachNErr  == 0:
     #if  i  == 200:
                
         
                #err=rbm_m1.compute_error(trX)       
                #rbm_m1.X=trX.copy()
                #err=rbm_m1.compute_error(trX)       
                err=rbm_m1.err_sum(trX)       
                #statCnt.append([i,err,rbm_m1.P1,rbm_m1.P2,rbm_m1.P3,rbm_m1.Nrepls, rbm_m1.AccMvs,rbm_m1.sU])       
                print ("Error,i  ",err,i)
                
                print([i,err], file=f)
                if i % 2 ==0:
                     f.flush()
#                rbm_m1.save_weights_visual((start / 1000))
                #rbm
     i=i+1           

print ('Final Error (trX):',rbm_m1.compute_error(trX))
print ('Final Error (teX):',rbm_m1.compute_error(teX))
sio.savemat(os.getcwd().split('\\')[-1]+'_'+str(i)+'.mat', {'vishid':np.array(rbm_m1.rbm_w, dtype='float64'),'visbiases':np.array(rbm_m1.rbm_vb, dtype='float64'),'hidbiases':np.array(rbm_m1.rbm_hb, dtype='float64')})
#rbm_m1.save_weights(i)
