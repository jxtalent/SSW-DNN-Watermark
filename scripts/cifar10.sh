# train a clean model
python main.py --config config/cifar10/cifar10_clean.yaml --runname cifar10_clean

# train a watermarked model (from pretrained)
python main.py --config config/cifar10/cifar10_watermark.yaml --runname cifar10_watermark --save_trigger

# train a watermarked model (from scratch)
python main.py --config config/cifar10/cifar10_watermark_fs.yaml --runname cifar10_watermark_fs --save_trigger

# evaluate a watermarked model
python main.py --action evaluate --dataset cifar10 --arch resnet18 --checkpoint_path /path/to/last.pth

# attack
# hard label extraction
python main.py --action attack --dataset cifar10 --arch resnet18 --epochs 40 --attack_type retrain --runname retrain_1128 \
  --victim_path /path/to/last.pth

# knowledge distillation with mobile net
python main.py --action attack --dataset cifar10 --arch mobile --epochs 40 --attack_type distill --runname distill  \
  --victim_path /path/to/last.pth

# knockoff nets
python main.py --action attack --dataset cifar10 --arch resnet18 --epochs 40 --attack_type knockoff --runname knockoff  \
  --victim_path /path/to/last.pth

# model fine-tuning
python main.py --action attack --dataset cifar10 --arch resnet18 --epochs 10 --attack_type rtal --runname rtal \
  --victim_path /path/to/last.pth

python main.py --action attack --dataset cifar10 --arch resnet18 --epochs 10 --attack_type ftal --runname ftal \
  --victim_path /path/to/last.pth

# model compression
python main.py --action attack --dataset cifar10 --arch resnet18 --attack_type quantization --runname quantization \
  --victim_path /path/to/last.pth

python main.py --action attack --dataset cifar10 --arch resnet18 --attack_type prune --runname pruning  \
  --victim_path /path/to/last.pth

# cross architecture hard label extraction
python main.py --action attack --dataset cifar10 --arch VGG16 --epochs 40 --attack_type cross --runname vgg16_cross_retraining  \
  --victim_path /path/to/last.pth

python main.py --action attack --dataset cifar10 --arch senet --epochs 40 --attack_type cross --runname senet_cross_retraining  \
  --victim_path /path/to/last.pth
