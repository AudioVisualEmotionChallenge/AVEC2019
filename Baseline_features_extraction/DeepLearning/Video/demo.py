#This script is for AVEC 2019 challenges. It provides three sets of deep visual features
#Please check README.txt for more details
#If you have any question, please contact Siyang.Song@nottingham.ac.uk 

import os
import argparse
import csv 
import sys
import shutil
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument('--src_vid_path', default='./videos/', type=str) # define source video directory
parser.add_argument('--openFace_bin_path', default='./FeatureExtraction', type=str) # openFace implementation
parser.add_argument('--orig_out_data_path', default='./processed/', type=str)
parser.add_argument('--dest_out_data_path', default='./feature_extraction/files/', type=str) # define the output path
parser.add_argument('--save_img', default=1) # save the aligned face images and .csv files or not
parser.add_argument('--save_feat', default=0) # save landmarks and feature files or not
parser.add_argument('--Res_feat', default=1) # Extracting Aff-wild ResNet feature
parser.add_argument('--VGG_feat', default=1) # Extracting Aff-wild ResNet feature
parser.add_argument('--Res_reg_feat', default=1) # Extracting Aff-wild ResNet feature
args = parser.parse_args()


vid_list = os.listdir(args.src_vid_path)
vid_list.sort()


for i in range(0, len(vid_list)):
    print(vid_list[i][:-4])

    # implementing OpenFace for face alignment
	
    print(args.openFace_bin_path+' -f '+args.src_vid_path+vid_list[i])
    os.system(args.openFace_bin_path+' -f '+args.src_vid_path+vid_list[i])
    if os.path.exists(args.dest_out_data_path+'aligned_face/'+vid_list[i][:-4]+'_aligned'):
	    shutil.rmtree(args.dest_out_data_path+'aligned_face/'+vid_list[i][:-4]+'_aligned')
    
    face_img_path = args.dest_out_data_path+'aligned_face/'+vid_list[i][:-4]+'_aligned/'
    shutil.move(args.orig_out_data_path + vid_list[i][:-4]+'_aligned', face_img_path)
		
	# Create .CSV file for indexing aligned images path	
    align_imglist = os.listdir(face_img_path)	
    align_imglist.sort()
    csv_path = args.dest_out_data_path+ 'csv_files/' 
	
    print('creating .csv')	
    if not os.path.exists(csv_path):
        os.makedirs(csv_path)
		
    out_csv_path = csv_path + vid_list[i][:-4] + '.csv'	
    with open(out_csv_path, mode='w') as data_file:
        data_writer = csv.writer(data_file, delimiter=',')
        for j in range(0, len(align_imglist)):
            data_writer.writerow([args.dest_out_data_path + 'aligned_face/' + vid_list[i][:-4]+'_aligned/' + align_imglist[j]])
				
    # Conducting deep feature extraction
    print('Feature Extraction!')	
    vgg_feat_path = args.dest_out_data_path + 'Deep_features/VGG/'
    res_feat_path = args.dest_out_data_path + 'Deep_features/RES/'
    res_feat_path_reg = args.dest_out_data_path + 'Deep_features/RES_reg/'
    if not os.path.exists(vgg_feat_path):
        os.makedirs(vgg_feat_path)		
    if not os.path.exists(res_feat_path):
        os.makedirs(res_feat_path)
    if not os.path.exists(res_feat_path_reg):        
        os.makedirs(res_feat_path_reg)			
		
    feat_out_Name_vgg = vgg_feat_path + vid_list[i][:-4] +'_vgg.mat'
    feat_out_Name_res = res_feat_path + vid_list[i][:-4] +'_res.mat'
    feat_out_Name_res_reg = res_feat_path_reg + vid_list[i][:-4] +'_res_reg.mat'
    
    if args.Res_feat == 1:
        os.system('LD_LIBRARY_PATH=/usr2/local/cuda-9.0/lib64:/usr2/local/cuda-9.0/lib64/stubs/:$LD_LIBRARY_PATH srun python ./feature_extraction/feature_extraction_RESNET.py --input_file '+ out_csv_path + ' --save_file '+feat_out_Name_res)
    
    if args.VGG_feat == 1:
	    os.system('LD_LIBRARY_PATH=/usr2/local/cuda-9.0/lib64:/usr2/local/cuda-9.0/lib64/stubs/:$LD_LIBRARY_PATH srun python ./feature_extraction/feature_extraction_VGG.py --input_file '+ out_csv_path + ' --save_file '+ feat_out_Name_vgg)
    
    if args.Res_reg_feat == 1:
	    os.system('LD_LIBRARY_PATH=/usr2/local/cuda-9.0/lib64:/usr2/local/cuda-9.0/lib64/stubs/:$LD_LIBRARY_PATH srun python ./feature_extraction/feature_extraction_RESNET_reg.py --input_file '+ out_csv_path + ' --save_file '+ feat_out_Name_res_reg)
	
    if args.save_img == 0:
        shutil.rmtree(face_img_path)
        os.remove(out_csv_path)
		
		
if args.save_feat == 0:
    shutil.rmtree('./processed/')