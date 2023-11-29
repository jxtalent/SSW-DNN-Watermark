# train a clean model
python main.py --config config/fashion/fashion_clean.yaml --runname fashion_clean

# train a watermarked model (from pretrained)
python main.py --config config/fashion/fashion_watermark.yaml --runname fashion_watermark --save_trigger

# train a watermarked model (from scratch)
python main.py --config config/fashion/fashion_watermark_fs.yaml --runname fashion_watermark_fs --save_trigger

# evaluate a watermarked model
python main.py --action evaluate --dataset fashion --arch vanilla --checkpoint_path /path/to/last.pth

# attack
python main.py --action attack --dataset fashion --arch vanilla --epochs 50 --attack_type retrain --runname retrain \
  --victim_path /path/to/last.pth

python main.py --action attack --dataset fashion --arch simple --epochs 50 --attack_type distill --runname distill  \
  --victim_path /path/to/last.pth

python main.py --action attack --dataset fashion --arch vanilla --epochs 50 --attack_type knockoff --runname knockoff  \
  --victim_path /path/to/last.pth

python main.py --action attack --dataset fashion --arch vanilla --epochs 10 --attack_type rtal --runname rtal \
  --victim_path /path/to/last.pth

python main.py --action attack --dataset fashion --arch vanilla --epochs 10 --attack_type ftal --runname ftal \
  --victim_path /path/to/last.pth

python main.py --action attack --dataset fashion --arch vanilla --attack_type quantization --runname quantization \
  --victim_path /path/to/last.pth

python main.py --action attack --dataset fashion --arch vanilla --attack_type prune --runname pruning  \
  --victim_path /path/to/last.pth