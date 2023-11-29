# train a non-watermarked clean model
python main.py --config config/cifar100/cifar100_clean.yaml --runname cifar100_clean

# train a watermarked model (from pretrained)
python main.py --config config/cifar100/cifar100_watermark.yaml --runname cifar100_watermark_empty_f --save_trigger

# train a watermarked model (from scratch)
python main.py --config config/cifar100/cifar100_watermark_fs.yaml --runname cifar100_watermark_fs_emptyf --save_trigger

# evaluate a watermarked model
python main.py --action evaluate --dataset cifar100 --arch resnet18 --checkpoint_path /path/to/last.pth

# attack
# hard label extraction
python main.py --action attack --dataset cifar100 --arch resnet18 --epochs 40 --attack_type retrain --runname retrain \
  --victim_path /path/to/last.pth

# knowledge distillation with mobile net
python main.py --action attack --dataset cifar100 --arch mobile --epochs 40 --attack_type distill --runname distill  \
  --victim_path /path/to/last.pth

# knockoff nets
python main.py --action attack --dataset cifar100 --arch resnet18 --epochs 40 --attack_type knockoff --runname knockoff  \
  --victim_path /path/to/last.pth

# model fine-tuning
python main.py --action attack --dataset cifar100 --arch resnet18 --epochs 10 --attack_type rtal --runname rtal \
  --victim_path /path/to/last.pth

python main.py --action attack --dataset cifar100 --arch resnet18 --epochs 10 --attack_type ftal --runname ftal \
  --victim_path /path/to/last.pth

# model compression
python main.py --action attack --dataset cifar100 --arch resnet18 --attack_type quantization --runname quantization \
  --victim_path /path/to/last.pth

python main.py --action attack --dataset cifar100 --arch resnet18 --attack_type prune --runname pruning  \
  --victim_path /path/to/last.pth

# cross architecture hard label extraction
python main.py --action attack --dataset cifar100 --arch VGG16 --epochs 40 --attack_type cross --runname vgg16_cross_retraining  \
  --victim_path /path/to/last.pth

python main.py --action attack --dataset cifar100 --arch senet --epochs 40 --attack_type cross --runname senet_cross_retraining  \
  --victim_path /path/to/last.pth
  