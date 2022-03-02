import cv2 
import numpy as np 
import pickle 
import argparse
import os 
from time import time
from tqdm import tqdm
import joblib
from utils import * 

EXT = ['jpg', 'png', 'ppm', 'pgm']



def FeatMatch(opts): 
    
    img_names = sorted(os.listdir(opts.data_dir))
    img_names = [n for n in img_names if n.split('.')[-1] in EXT]
    img_paths = [os.path.join(opts.data_dir, x) for x in img_names]

    print(img_paths)
    if opts.use_mask:
        mask_names = sorted(os.listdir(opts.mask_dir))
        mask_paths = [os.path.join(opts.mask_dir, x) for x in mask_names if \
                    x.split('.')[-1] in EXT]
    
    feat_out_dir = os.path.join(opts.out_dir,'features')
    matches_out_dir = os.path.join(opts.out_dir,'matches')
    if opts.verbose:
        kp_show = os.path.join(opts.out_dir,'kp_show')
        if not os.path.exists(kp_show): 
            os.makedirs(kp_show)
    if not os.path.exists(feat_out_dir): 
        os.makedirs(feat_out_dir)
    if not os.path.exists(matches_out_dir): 
        os.makedirs(matches_out_dir)
    
    data = []
    pbar = tqdm(total=len(img_paths))

    print('Extracting features ..')
    for i, img_path in enumerate(img_paths): 
        img = cv2.imread(img_path)[:,:,::-1]
        img_name = img_names[i].split('.')[0]
        #img = img[:,:,::-1] 
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if opts.use_mask:
            mask = 255-cv2.imread(mask_paths[i],0)
        else: mask=None
        feat = cv2.SIFT_create(nfeatures=30000, nOctaveLayers = 5, edgeThreshold = 50, contrastThreshold=0.02)
        #feat = cv2.SIFT_create()
        kp, desc = feat.detectAndCompute(img,mask)
        data.append((img_path,img_name, kp, desc))

        kp_ = kp2list(kp)
        with open(os.path.join(feat_out_dir, '{}.kp'.format(img_name)),'wb') as out:
            #pickle.dump(kp_, out)
            joblib.dump(kp_, out)
        with open(os.path.join(feat_out_dir, '{}.desc'.format(img_name)),'wb') as out:
            #pickle.dump(desc, out)
            joblib.dump(desc, out)
        if opts.verbose:
            img=cv2.drawKeypoints(img,kp,img, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            cv2.imwrite(os.path.join(kp_show, '{}.jpg'.format(img_name)), img)
        pbar.update(1)

    pbar.close()
    num_done = 0 
    num_matches = int (((len(img_paths)-1) * (len(img_paths))) / 2)
    print('Matching keypoints ..')
    pbar = tqdm(total=num_matches)
    t1 = time()
    for i in range(len(data)): 
        for j in range(i+1, len(data)): 
            img_path1, img_name1, kp1, desc1 = data[i]
            img_path2, img_name2, kp2, desc2 = data[j]

            assert opts.matcher in ['Brute-Force', 'FLANN']
            if opts.matcher == 'Brute-Force':
                # BFMatcher with default params
                matcher = cv2.BFMatcher()
            else:
                # FLANN parameters
                FLANN_INDEX_KDTREE = 0
                index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
                search_params = dict(checks=50)   # or pass empty dictionary

                matcher = cv2.FlannBasedMatcher()

            matches = matcher.knnMatch(desc1,desc2,k=2)
            if opts.ratio_test:
                matches = get_good_matches(matches)
            matches = sorted(matches, key = lambda x:x.distance)
            matches_ = matches2list(matches)

            pickle_path = os.path.join(matches_out_dir, 'match_{}_{}.pkl'.format(img_name1,
                                                                                 img_name2))
            with open(pickle_path,'wb') as out:
                joblib.dump(matches_, out)

            t2 = time()
            matches = sorted(matches, key = lambda x:x.distance)
            # Draw first 40 matches.
            if opts.verbose:
                img1 = cv2.imread(img_path1)[:,:,::-1]
                img2 = cv2.imread(img_path2)[:,:,::-1]
                img3 = cv2.drawMatches(img1,kp1,img2,kp2,matches,None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
                cv2.imwrite(os.path.join(kp_show, '{}_{}.jpg'.format(img_name1, img_name2)), img3)
            pbar.update(1)
            pbar.set_description(f"{len(matches)} kp matched")
    pbar.close()
    print('Matching DONE: [time={:.2f}s]'.format(t2-t1))

            

if __name__=='__main__': 
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir',action='store',type=str,default='dataset/Herz-jesus-P25/images',
                        dest='data_dir',help='directory containing images (default: dataset/fountain-P11/images)') 
    parser.add_argument('--use_mask',action='store_true')                    
    parser.add_argument('--mask_dir',action='store',type=str,default='dataset/bunny_data/silhouettes',
                        dest='mask_dir',help='directory containing images (default: dataset/fountain-P11/images)') 
    parser.add_argument('--out_dir',action='store',type=str,default='Herz-jesus-P25/',
                        dest='out_dir',help='root directory to store results in \
                        (default: ../data/fountain-P11)') 

    #feature matching args
    parser.add_argument('--matcher', choices = ['Brute-Force', 'FLANN'], default='FLANN', \
                        help='[BFMatcher|FlannBasedMatcher] Matching algorithm to use' )
    parser.add_argument('--ratio_test',action='store',type=bool,default=True,)
    parser.add_argument('--verbose',action='store_true')   
    opts = parser.parse_args()

    FeatMatch(opts)