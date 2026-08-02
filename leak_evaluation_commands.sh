# VANILLA

python main.py --config configs/datasets/cifar10/cifar10.yml configs/datasets/cifar10/cifar10_ood.yml configs/networks/resnet18_32x32.yml configs/pipelines/test/test_leak.yml configs/preprocessors/base_preprocessor.yml configs/postprocessors/leak.yml --network.checkpoint 'results/cifar10_resnet18_32x32_base_e100_lr0.1_default/s0/best.ckpt' --mark=vanilla

# LOGITNORM

python main.py --config configs/datasets/cifar10/cifar10.yml configs/datasets/cifar10/cifar10_ood.yml configs/networks/resnet18_32x32.yml configs/pipelines/test/test_leak.yml configs/preprocessors/base_preprocessor.yml configs/postprocessors/leak.yml --network.checkpoint 'results/cifar10_resnet18_32x32_logitnorm_e100_lr0.1_alpha0.04_default/s0/best.ckpt' --mark=logitnorm

# VOS

python main.py --config configs/datasets/cifar10/cifar10.yml configs/datasets/cifar10/cifar10_ood.yml configs/networks/resnet18_32x32.yml configs/pipelines/test/test_leak.yml configs/preprocessors/base_preprocessor.yml configs/postprocessors/leak.yml --network.checkpoint 'results/cifar10_resnet18_32x32_vos_e100_lr0.1_default/s0/best.ckpt' --mark=vos

# T2FNorm

python main.py --config configs/datasets/cifar10/cifar10.yml configs/datasets/cifar10/cifar10_ood.yml configs/networks/t2fnorm_net.yml configs/pipelines/test/test_leak.yml configs/preprocessors/base_preprocessor.yml configs/postprocessors/leak.yml --network.checkpoint 'results/cifar10_t2fnorm_net_t2fnorm_e100_lr0.1_alpha0.1_default/s11033/best.ckpt' --mark=t2fnorm

# DER

python main.py --config configs/datasets/cifar10/cifar10.yml configs/datasets/cifar10/cifar10_ood.yml configs/networks/resnet18_32x32.yml configs/pipelines/test/test_leak.yml configs/preprocessors/base_preprocessor.yml configs/postprocessors/leak.yml --network.checkpoint 'results/cifar10_resnet18_32x32_der_e100_lr0.005_gamma0.2_beta0.5_clean_loss_included/s0/best.ckpt' --mark=der

# ASCOOD

python main.py --config configs/datasets/cifar10/cifar10.yml configs/datasets/cifar10/cifar10_ood.yml configs/networks/ascood_net.yml configs/pipelines/test/test_leak.yml configs/preprocessors/base_preprocessor.yml configs/postprocessors/leak.yml --network.checkpoint 'results/cifar10_ascood_net_ascood_e100_lr0.1_w1.0_p0.15_otype_shuffle_alpha_10.0_10.0_kl_div_True_default/s0/best.ckpt' --mark=ascood
