import numpy as np 
import cv2 
import argparse
import pickle
import os 
from time import time
import matplotlib.pyplot as plt
import joblib

from utils import * 
from ply import *


class SFM(object): 
    def __init__(self, opts): 
        self.opts = opts
        self.point_cloud = np.zeros((0,3))

        #setting up directory stuff..
        self.images_dir = os.path.join(opts.data_dir)
        self.feat_dir = os.path.join(opts.preprocess_dir, 'features')
        self.matches_dir = os.path.join(opts.preprocess_dir, 'matches')
        self.out_cloud_dir = os.path.join(opts.out_dir, 'point-clouds')
        self.out_err_dir = os.path.join(opts.out_dir, 'errors')

        #output directories
        if not os.path.exists(self.out_cloud_dir): 
            os.makedirs(self.out_cloud_dir)

        if (opts.plot_error is True) and (not os.path.exists(self.out_err_dir)): 
            os.makedirs(self.out_err_dir)

        self.image_names = [x.split('.')[0] for x in sorted(os.listdir(self.images_dir)) \
                            if x.split('.')[-1] in ['jpg', 'png', 'ppm']]

        #setting up shared parameters for the pipeline
        self.image_data, self.matches_data, errors = {}, {}, {}


        #self.K = np.array([[2759.48,0,1520.69],[0,2764.16,1006.81],[0,0,1]])
        self.K = load_camera_intrinsics(os.path.join(opts.data_dir,'K.txt'))
        print('K = ', self.K)
        
    def _LoadFeatures(self, name, return_desc=False): 
        kp = joblib.load(os.path.join(self.feat_dir,'{}.kp'.format(name)))
        kp = list2kp(kp)

        if return_desc:
            desc = joblib.load(os.path.join(self.feat_dir,'{}.desc'.format(name)))

            return kp, desc 
        return kp

    def _LoadMatches(self, name1, name2): 
        matches = joblib.load(os.path.join(self.matches_dir,'match_{}_{}.pkl'.format(name1,name2)))
        matches = list2matches(matches)
        return matches

    def _GetAlignedMatches(self,kp1,kp2,matches):
        img1idx = np.array([m.queryIdx for m in matches])
        img2idx = np.array([m.trainIdx for m in matches])

        #filtering out the keypoints that were matched. 
        kp1_filt = (np.array(kp1))[img1idx]
        kp2_filt = (np.array(kp2))[img2idx]

        #retreiving the image coordinates of matched keypoints
        img1pts = np.array([kp.pt for kp in kp1_filt])
        img2pts = np.array([kp.pt for kp in kp2_filt])

        return img1pts, img2pts, img1idx, img2idx

    def _BaselinePoseEstimation(self, name1, name2):

        kp1 = self._LoadFeatures(name1)
        kp2 = self._LoadFeatures(name2)  

        matches = self._LoadMatches(name1, name2)
        matches = sorted(matches, key = lambda x:x.distance)

        img1pts, img2pts, img1idx, img2idx = self._GetAlignedMatches(kp1,kp2,matches)
        
        F,mask = cv2.findFundamentalMat(img1pts,img2pts,method=cv2.FM_RANSAC, 
                                        ransacReprojThreshold=20, confidence=0.99, maxIters=200)
        mask = mask.astype(bool).flatten()

        E = self.K.T @ F @ self.K
        _,R,t,_ = cv2.recoverPose(E,img1pts[mask],img2pts[mask],self.K)

        R1, t1, ref1 = np.eye(3,3), np.zeros((3,1)), np.ones((len(kp1),))*-1
        R2, t2, ref2 = R,t,np.ones((len(kp2),))*-1

        self.matches_data[(name1,name2)] = [matches, img1pts[mask], img2pts[mask], 
                                            img1idx[mask],img2idx[mask]]

        return R1, t1, ref1, R2, t2, ref2

    def _TriangulateTwoViews(self, name1, name2): 

        def __TriangulateTwoViews(img1pts, img2pts, R1, t1, R2, t2): 
            img1ptsHom = cv2.convertPointsToHomogeneous(img1pts)[:,0,:]
            img2ptsHom = cv2.convertPointsToHomogeneous(img2pts)[:,0,:]

            img1ptsNorm = (np.linalg.inv(self.K) @ img1ptsHom.T).T
            img2ptsNorm = (np.linalg.inv(self.K) @ img2ptsHom.T).T

            img1ptsNorm = cv2.convertPointsFromHomogeneous(img1ptsNorm)[:,0,:]
            img2ptsNorm = cv2.convertPointsFromHomogeneous(img2ptsNorm)[:,0,:]

            pts4d = cv2.triangulatePoints(np.hstack((R1,t1)),np.hstack((R2,t2)),
                                            img1ptsNorm.T,img2ptsNorm.T)
            pts3d = cv2.convertPointsFromHomogeneous(pts4d.T)[:,0,:]

            return pts3d

        def _Update3DReference(ref1, ref2, img1idx, img2idx, upp_limit): 

            ref1[img1idx] = np.arange(upp_limit) 
            ref2[img2idx] = np.arange(upp_limit) 

            return ref1, ref2

        R1, t1, ref1, R2, t2, ref2 = self._BaselinePoseEstimation(name1, name2)

        _, img1pts, img2pts, img1idx, img2idx = self.matches_data[(name1,name2)]
        
        new_point_cloud = __TriangulateTwoViews(img1pts, img2pts, R1, t1, R2, t2)

        ref1, ref2 = _Update3DReference(ref1, ref2, img1idx, img2idx,new_point_cloud.shape[0])
        print('pointshap', new_point_cloud.shape)
        return new_point_cloud, ref1, ref2
        
    def _align_point_cloud(self, cloud1, cloud2, name0, name1, name2, ref1, ref2):   
        _, _, _, _, img1idx0 = self.matches_data[(name0,name1 )]
        _, _, _, img1idx2, _ = self.matches_data[(name1,name2)]
        common_matchs=list(set(img1idx0) & set(img1idx2))
        ref1, ref2 = ref1[common_matchs], ref2[common_matchs]
        sub_cloud1, sub_cloud2 = cloud1[ref1[ref1>=0].astype(int)], cloud2[ref2[ref2>=0].astype(int)]
        print(sub_cloud1.shape)
        R, T, rms = ransac_rigid_transform(sub_cloud2.T, sub_cloud1.T,iters=80)
        print(rms)
        cloud2_aligned = (R @ cloud2.T + T).T
        return cloud2_aligned

    def ToPly(self, filename, point_cloud, name, ref):
        
        def _GetColors(): 
            colors = np.zeros_like(point_cloud)
            kp = self._LoadFeatures(name)
            kp = np.array(kp)[ref>=0]
            image_pts = np.array([_kp.pt for _kp in kp])
            image = cv2.imread(os.path.join(self.images_dir, name+ '.jpg'))[:,:,::-1]

            colors[ref[ref>=0].astype(int)] = image[image_pts[:,1].astype(int),
                                                    image_pts[:,0].astype(int)]
            
            return colors.astype('uint8')
        
        colors = _GetColors()
        write_ply(filename, [point_cloud, colors], ['x', 'y', 'z', 'red', 'green', 'blue'])
        #pts2ply(self.point_cloud, colors, filename)

    def _ComputeReprojectionError(self, name): 
        
        def _ComputeReprojections(X,R,t,K): 
            outh = K.dot(R.dot(X.T) + t )
            out = cv2.convertPointsFromHomogeneous(outh.T)[:,0,:]
            return out 

        R, t, ref = self.image_data[name]
        reproj_pts = _ComputeReprojections(self.point_cloud[ref[ref>0].astype(int)], R, t, self.K)

        kp = self._LoadFeatures(name)
        img_pts = np.array([kp_.pt for i, kp_ in enumerate(kp) if ref[i] > 0])
        
        err = np.mean(np.sqrt(np.sum((img_pts-reproj_pts)**2,axis=-1)))

        if self.opts.plot_error: 
            fig,ax = plt.subplots()
            image = cv2.imread(os.path.join(self.images_dir, name+'.jpg'))[:,:,::-1]
            ax = DrawCorrespondences(image, img_pts, reproj_pts, ax)
            
            ax.set_title('reprojection error = {}'.format(err))

            fig.savefig(os.path.join(self.out_err_dir, '{}.png'.format(name)))
            plt.close(fig)
            
        return err
    

    def Run2(self):
        old_point_cloud, _, ref0 = self._TriangulateTwoViews(self.image_names[0], self.image_names[1])
        self.ToPly(os.path.join(self.out_cloud_dir, 'cloud_0_view.ply'), old_point_cloud, self.image_names[1], ref0)

        for i in range(1, len(self.image_names)-1):
            name1, name2 = self.image_names[i], self.image_names[i+1]

            t0, errors = 0, []

            new_point_cloud, ref1, ref2 = self._TriangulateTwoViews(name1, name2)
            t1 = time()
            t = t1-t0
            print('Baseline Cameras {0}, {1}: Baseline Triangulation [time={2:.3}s]'.format(name1, 
                                                                                    name2, t))
            aligned_point_cloud = self._align_point_cloud(old_point_cloud, new_point_cloud, self.image_names[i-1],
                                    self.image_names[i], self.image_names[i+1], ref0, ref1)


            #3d point cloud generation and reprojection error evaluation
            self.ToPly(os.path.join(self.out_cloud_dir, 'cloud_{}_view.ply'.format(i)), aligned_point_cloud, name1, ref1)
            old_point_cloud, ref0 = aligned_point_cloud, ref2

            

def SetArguments(parser): 

    #directory stuff
    parser.add_argument('--data_dir',action='store',type=str,default='dataset/Herz-jesus-P25/images',dest='data_dir',
                        help='root directory containing input data (default: dataset/fountain-P11/images)') 
    parser.add_argument('--out_dir',action='store',type=str,default='Herz-jesus-P25/',dest='out_dir',
                        help='root directory to store results in (default: fountain-P11/)') 
    parser.add_argument('--preprocess_dir',action='store',type=str,default='Herz-jesus-P25/',dest='preprocess_dir',
                        help='root directory where keypoints and mateches are saved (default: fountain-P11/)') 

    #misc
    parser.add_argument('--plot_error',action='store',type=bool,default=False,dest='plot_error')


if __name__=='__main__': 
    parser = argparse.ArgumentParser()
    SetArguments(parser)
    opts = parser.parse_args()

    sfm = SFM(opts)
    sfm.Run2()