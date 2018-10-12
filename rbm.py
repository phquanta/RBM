# -*- coding: utf-8 -*-
"""
Created on Sat Mar 10 01:11:19 2018

@author: Andrei
"""


import numpy as np
#import input_data
import tensorflow.examples.tutorials.mnist.input_data as input_data
#import Image
from PIL import Image
from utils import tile_raster_images
import random
from pathlib import Path
import os







class rbm:
    i=0
    np.random.seed(seed=2000)
    def sample_prob(self,probs):
        
        #return np.asarray((np.sign(probs-np.random.uniform(low=0,high=1,size=np.shape(probs)))>0).astype(float),dtype=np.float32)
        return (np.sign(probs-np.random.uniform(low=0,high=1,size=np.shape(probs)))>0).astype(float)
    
    #@np.vectorize
    def sigmoid(self,x):
        #return np.asarray(1. / (1. + np.exp(-x),dtype=np.float32)
        return 1. / (1. + np.exp(-x))
    
    def h_sample(self,X):
        return self.sample_prob(self.sigmoid(np.matmul(X, self.rbm_w) + self.rbm_hb))
    
    def v_sample(self,X):
        return self.sample_prob(self.sigmoid(np.matmul(self.h_sample(X), self.rbm_w.T) + self.rbm_vb))
    
    def err(self,X):
        return X - self.v_sample(X)
    
    def err_sum(self,X):
        return  np.mean(self.err(X) * self.err(X))
    
    def h0(self,X):
        h0probs = self.sigmoid(np.matmul(X, self.rbm_w) + self.rbm_hb)
        return self.sample_prob(h0probs)

    def rbmGibbs(self,v, k):
            #print(v)    
            vOut=v.copy()
            for i in range(0,k-1):
                hOut = self.sample_prob(self.sigmoid(np.matmul(vOut, self.rbm_w) + self.rbm_hb))
                #vOutOld=vOut.copy()
                vOut = self.sample_prob(self.sigmoid(np.matmul(hOut, self.rbm_w.T) + self.rbm_vb))
             #   print(i,k,np.sum(vOut-vOutOld),np.sum(vOut),np.sum(vOutOld))
            hOut = self.sample_prob(self.sigmoid(np.matmul(vOut, self.rbm_w) + self.rbm_hb))
                #vOutOld=vOut.copy()
            vOut = self.sigmoid(np.matmul(hOut, self.rbm_w.T) + self.rbm_vb)


            return vOut,hOut
            #return vv, hh, count+1, k
    
    def compute_error(self, X):   
        #error=
        self.X=X.copy()
        error= self.err_sum()
        return error

    def compute_error_whole(self,trX):
           g=0
           error=0.
           batchsize1=int(len(trX)/5.)
           for start, end in zip(range(0, len(trX), batchsize1), range(batchsize1, len(trX), batchsize1)):
                           batch = trX[start:end]
                           #nonlocal error    
                   #        nonlocal error
                           error=error+self.compute_error(batch)
                           g=g+1    
        
           error=error/g

           #with tf.variable_scope(self.scope):
             #error= self.sess.run(self.err_sum, feed_dict={self.X: trX}) 
     #    return self.sess.run(self.err_sum, feed_dict={self.X: trX}) 
           return error,
 

    def updateAll(self,X):
        
        h0=self.h0(X)
        
        [v_fantasy,h_fantasy]=self.rbmGibbs(X,self.k)
        #vOut = self.sample_prob(self.sigmoid(np.matmul(h0, self.rbm_w.T) + self.rbm_vb))
        #hOut = self.sample_prob(self.sigmoid(np.matmul(vOut, self.rbm_w) + self.rbm_hb))
        #vOut = self.sample_prob(self.sigmoid(np.dot(h0,self.rbm_w.T) + self.rbm_vb))
        #hOut = self.sample_prob(self.sigmoid(np.dot(vOut,self.rbm_w) + self.rbm_hb))

                #vOutOld=vOut.copy()
        #[v_fantasy,h_fantasy]=[vOut,hOut]

        
        self.vArr=v_fantasy.copy()
        w_plus=np.matmul(X.T,h0)        
        
        #self.w_negative_grad_old = tf.matmul(tf.transpose(self.v2), self.h2)
        w_minus=np.matmul(v_fantasy.T,h_fantasy)
        self.rbm_w=self.rbm_w+self.alpha*(w_plus-w_minus)/np.shape(X)[0]-self.rbm_w*self.wDecay*self.alpha
        self.rbm_vb=self.rbm_vb+self.alpha*np.mean(X-v_fantasy,axis=0)
        #print(np.shape(v_fantasy), np.sum(v_fantasy),np.sum(h0))
        self.rbm_hb=self.rbm_hb+self.alpha*np.mean(h0-h_fantasy,axis=0)-self.alpha * self.hDecay*self.rbm_hb
        
        
        


        
    
    def __init__(self, batchsize,alpha,decay, hDecay, n_h,n_v, k=10):
        
        
        self.alpha = alpha
        self.batchsize = batchsize
        self.n_h=n_h
        self.n_v=n_v

        self.k = k
        self.counter=0
        self.wDecay=decay
        self.hDecay=hDecay
        self.hArr=np.random.randint(2,size=(self.batchsize,self.n_h)).astype(float)
        self.vArr=np.random.randint(2,size=(self.batchsize,self.n_v)).astype(float)
        #self.rbm_w=np.random.rand(self.n_v,self.n_h)/1000.
        self.rbm_w=np.random.normal(0.0, 0.01, [self.n_v, self.n_h])
        
        self.rbm_vb=np.zeros(self.n_v)
        self.rbm_hb=np.zeros(self.n_h)
        self.w_plus=np.zeros(np.shape(self.rbm_w),dtype='float32')
        self.w_minus=np.zeros(np.shape(self.rbm_w),dtype='float32')
        
        
        
    def gen_v1_after_k_steps(self,X,k,ind):
    
    #       [self.vN0,self.hN0,_,_]=tf.while_loop(self.lessThanK, self.rbmGibbs, [self.v1orig, self.h1orig, self.ct, k], parallel_iterations=1, back_prop=False)
            [v_fantasy,h_fantasy]=self.rbmGibbs(X,k)
            #print(np.shape(h_fantasy))
            #v= np.asarray(self.sigmoid(np.matmul(h_fantasy, self.rbm_w.T) + self.rbm_vb),dtype=np.float32)
            
            v= self.sigmoid(np.matmul(h_fantasy, self.rbm_w.T) + self.rbm_vb)
            #print(np.shape(v))
            #print(v)
    #       self.vN = tf.nn.sigmoid(tf.matmul(self.hN0, tf.transpose(self.rbm_w)) + self.rbm_vb)
    #       v=self.sess.run(self.vN,feed_dict={self.X:X})       
    
            
            image = Image.fromarray(
                          tile_raster_images(
                              X=v,
                              img_shape=(28, 28),
                              tile_shape=(25, 20),
                              tile_spacing=(1, 1)
    
                              )
                          )
                                                                
                                                                
                                                                        
            image.save("vN_after_%d_steps_step_%d.png" % (k,ind))



    
        


        
        

        

    

    def batch_train_with_deps(self,X):


                 #self.X=X.copy()
                 self.updateAll(X)
                 
                 ############### CD ##########################
                 #self.vArr=X.copy()
                 #w, v_b, h_b=self.sess.run([self.updateW,self.updateVb,self.updateHb], 
                 #                            feed_dict={self.X: X,self.vv1: self.vArr})
                 ##self.vArr=vArr1.copy()
                 ####################################################
                 
                 
                 ############### PCD ##########################
                 #vArr1,w, v_b, h_b=self.sess.run([self.ch,self.updateW,self.updateVb,self.updateHb], 
                 #                            feed_dict={self.X: X,self.vv1: self.vArr})
                 #self.vArr=vArr1.copy()
                 #####################################################
                 return self.rbm_w, self.rbm_vb,self.rbm_hb
       
  
    def save_fantasy_visual(self,ind,trX):
#              with tf.control_dependencies([dep for dep in self.dependencies]):
                  [v1,v2]=self.sess.run([self.v1,self.v2],feed_dict={self.X:trX})

                  
          
                  image = Image.fromarray(
                        tile_raster_images(
                                X=v1,
                                img_shape=(28, 28),
                                tile_shape=(25, 20),
                                tile_spacing=(1, 1)
                                )
                        )

                  image1 = Image.fromarray(
                        tile_raster_images(
                                X=v2,
                                img_shape=(28, 28),
                                tile_shape=(25, 20),
                                tile_spacing=(1, 1)
                                )
                        )
                  image2 = Image.fromarray(
                        tile_raster_images(
                                X=v2-v1,
                                img_shape=(28, 28),
                                tile_shape=(25, 20),
                                tile_spacing=(1, 1)
                                )
                        )


                
                  image.save("v0_%d.png" % ind)
                  image1.save("vlast_%d.png" % ind)
                  image2.save("diff_%d.png" % ind)
        

    def save_pcd_fantasy(self,ind):
          
                  image = Image.fromarray(
                        tile_raster_images(
                                X=self.vArr,
                                img_shape=(28, 28),
                                tile_shape=(25, 20),
                                tile_spacing=(1, 1)
                                )
                        )

                
                  image.save("pcd_fantasy_step_%d.png" % ind)


    def save_weights_visual(self,ind):
            
        
               image = Image.fromarray(
                        tile_raster_images(
                                X=(self.rbm_w).T,
                                img_shape=(28, 28),
                                tile_shape=(25, 20),
                                tile_spacing=(1, 1)
                                )
                        )
                
                
               image.save("rbm_%d.png" % ind)


    def sample_from_rbm(self,trX,g):
      ind1=random.randint(0,len(trX))
      ind2=ind1+10
      n_digits=ind2-ind1
      v_sampled=[]
      for i in range(0,10):
          v_sampled.append(self.sess.run(self.v_sample,feed_dict={self.X:trX[ind1:ind2]}))
    
    

      v_sampled=np.squeeze(np.asarray(v_sampled))
      v_sampled=np.append(v_sampled[0,:,:],trX[ind1:ind2],axis=0)

      vsampled = Image.fromarray(
                     tile_raster_images(
                                X=v_sampled,
                                img_shape=(28, 28),
                                tile_shape=(25, 20),
                                #tile_spacing=(1, 1)
                                )
                        )
                
      vsampled.show()                
      vsampled.save("vsampled_"+str(g)+"_all.png")


    def sample1_from_rbm(self,trX):
      ind1=random.randint(0,len(trX))
     
      v_sampled=[]
      for i in range(0,10):
          v_sampled.append(self.sess.run(self.v_sample,feed_dict={self.X:trX[ind1:ind1+1]}))



      v_sampled=np.squeeze(np.asarray(v_sampled))
      v_sampled=np.append(v_sampled[:,:],trX[ind1:ind1+1],axis=0)

      vsampled = Image.fromarray(
                     tile_raster_images(
                                X=v_sampled,
                                img_shape=(28, 28),
                                tile_shape=(25, 20),
                                #tile_spacing=(1, 1)
                                )
                        )

      vsampled.show()
      vsampled.save("vsampled_all.png")


    def sample_from_rbm(self,trX,g):
      ind1=random.randint(0,len(trX))
      ind2=ind1+10
      n_digits=ind2-ind1
      v_sampled=[]
      for i in range(0,10):
          v_sampled.append(self.sess.run(self.v_sample,feed_dict={self.X:trX[ind1:ind2]}))
    
    

      v_sampled=np.squeeze(np.asarray(v_sampled))
      v_sampled=np.append(v_sampled[0,:,:],trX[ind1:ind2],axis=0)

      vsampled = Image.fromarray(
                     tile_raster_images(
                                X=v_sampled,
                                img_shape=(28, 28),
                                tile_shape=(25, 20),
                                #tile_spacing=(1, 1)
                                )
                        )
                
      vsampled.show()                
      vsampled.save("vsampled_"+str(g)+"_all.png")


    def sample_from_rbm_ind(self,trX,g,ind):
      ind1=0
      ind2=ind1+100
      n_digits=ind2-ind1
      v_sampled=[]
      for i in range(0,10):
          v_sampled.append(self.sess.run(self.v_sample,feed_dict={self.X:trX[ind1:ind2]}))
    
    

      v_sampled=np.squeeze(np.asarray(v_sampled))
      v_sampled=np.append(v_sampled[0,:,:],trX[ind1:ind2],axis=0)

      vsampled = Image.fromarray(
                     tile_raster_images(
                                X=v_sampled,
                                img_shape=(28, 28),
                                tile_shape=(25, 20),
                                #tile_spacing=(1, 1)
                                )
                        )
                
      vsampled.show()                
      vsampled.save("vsampled_"+str(g)+"_all.png")



    
      
        

        
        

    
