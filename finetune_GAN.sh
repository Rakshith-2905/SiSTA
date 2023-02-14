if [ $1 == 'CelebA' ]; then
  python generative_augmentation/GAN_CelebA.py --generate_data_type=synthetic --domain=$2
fi
if [ $1 == 'AFHQ' ]; then
  python generative_augmentation/GAN_AFHQ.py --generate_data_type=synthetic --domain=$2 --class_name=$3
fi
if [ $1 == 'CIFAR-10' ]; then
  python generative_augmentation/GAN_CIFAR-10.py --generate_data_type=synthetic --domain=$2 --no_cls=$3
fi