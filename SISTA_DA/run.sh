# nohup python SHOT/image_NRC_target_Randconv.py --t 1 --gpu_id 0 &
# nohup python SHOT/image_NRC_target_Randconv.py --t 2 --gpu_id 0 &
# nohup python SHOT/image_NRC_target_Randconv.py --t 3 --gpu_id 1 &

# nohup python SHOT/image_NRC_target_feat_mixup.py --t 1 --gpu_id 0 &
# nohup python SHOT/image_NRC_target_feat_mixup.py --t 2 --gpu_id 0 &
# nohup python SHOT/image_NRC_target_feat_mixup.py --t 3 --gpu_id 1 &#
nohup python SHOT/image_NRC_target.py --variant 'direct_target' --gpu_id 8 &
nohup python SHOT/image_NRC_target.py --variant 'interp_concat' --gpu_id 9 &
nohup python SHOT/image_NRC_target.py --t 2 --gpu_id 8 &
nohup python SHOT/image_NRC_target.py --t 3 --gpu_id 9 &
nohup python SHOT/image_NRC_target.py --train_size 500 --gpu_id 9 &
nohup python SHOT/image_NRC_target.py --train_size 1000 --gpu_id 5 &
nohup python SHOT/image_NRC_target.py --train_size 2000 --gpu_id 8 &
nohup python SHOT/image_NRC_target.py --train_size 5000 --gpu_id 7 &
nohup python SHOT/image_NRC_target.py --train_size 10000 --gpu_id 6 &