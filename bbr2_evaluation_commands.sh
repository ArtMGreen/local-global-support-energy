# VANILLA

python main.py --config configs/datasets/cifar10/cifar10.yml configs/networks/resnet18_32x32.yml configs/pipelines/test/test_bbr.yml configs/preprocessors/base_preprocessor.yml --network.checkpoint 'results/cifar10_resnet18_32x32_base_e100_lr0.1_default/s0/best.ckpt' --mark=vanilla --evaluator.evaluator_args.layers_before_logits=2 --evaluator.evaluator_args.whiten_features=true --evaluator.evaluator_args.whitened_top_n_selection=true

# python main.py --config configs/datasets/cifar10/cifar10.yml configs/networks/resnet18_32x32.yml configs/pipelines/test/test_bbr.yml configs/preprocessors/base_preprocessor.yml --network.checkpoint 'results/cifar10_resnet18_32x32_base_e100_lr0.1_default/s0/best.ckpt' --mark=vanilla --evaluator.evaluator_args.layers_before_logits=2 --evaluator.evaluator_args.whiten_features=true --evaluator.evaluator_args.whitened_top_n_selection=false

python main.py --config configs/datasets/cifar10/cifar10.yml configs/networks/resnet18_32x32.yml configs/pipelines/test/test_bbr.yml configs/preprocessors/base_preprocessor.yml --network.checkpoint 'results/cifar10_resnet18_32x32_base_e100_lr0.1_default/s0/best.ckpt' --mark=vanilla --evaluator.evaluator_args.layers_before_logits=2 --evaluator.evaluator_args.whiten_features=false --evaluator.evaluator_args.whitened_top_n_selection=false

# LOGITNORM

python main.py --config configs/datasets/cifar10/cifar10.yml configs/networks/resnet18_32x32.yml configs/pipelines/test/test_bbr.yml configs/preprocessors/base_preprocessor.yml --network.checkpoint 'results/cifar10_resnet18_32x32_logitnorm_e100_lr0.1_alpha0.04_default/s0/best.ckpt' --mark=logitnorm --evaluator.evaluator_args.layers_before_logits=2 --evaluator.evaluator_args.whiten_features=true --evaluator.evaluator_args.whitened_top_n_selection=true

# python main.py --config configs/datasets/cifar10/cifar10.yml configs/networks/resnet18_32x32.yml configs/pipelines/test/test_bbr.yml configs/preprocessors/base_preprocessor.yml --network.checkpoint 'results/cifar10_resnet18_32x32_logitnorm_e100_lr0.1_alpha0.04_default/s0/best.ckpt' --mark=logitnorm --evaluator.evaluator_args.layers_before_logits=2 --evaluator.evaluator_args.whiten_features=true --evaluator.evaluator_args.whitened_top_n_selection=false

python main.py --config configs/datasets/cifar10/cifar10.yml configs/networks/resnet18_32x32.yml configs/pipelines/test/test_bbr.yml configs/preprocessors/base_preprocessor.yml --network.checkpoint 'results/cifar10_resnet18_32x32_logitnorm_e100_lr0.1_alpha0.04_default/s0/best.ckpt' --mark=logitnorm --evaluator.evaluator_args.layers_before_logits=2 --evaluator.evaluator_args.whiten_features=false --evaluator.evaluator_args.whitened_top_n_selection=false

# VOS

python main.py --config configs/datasets/cifar10/cifar10.yml configs/networks/resnet18_32x32.yml configs/pipelines/test/test_bbr.yml configs/preprocessors/base_preprocessor.yml --network.checkpoint 'results/cifar10_resnet18_32x32_vos_e100_lr0.1_default/s0/best.ckpt' --mark=vos --evaluator.evaluator_args.layers_before_logits=2 --evaluator.evaluator_args.whiten_features=true --evaluator.evaluator_args.whitened_top_n_selection=true

# python main.py --config configs/datasets/cifar10/cifar10.yml configs/networks/resnet18_32x32.yml configs/pipelines/test/test_bbr.yml configs/preprocessors/base_preprocessor.yml --network.checkpoint 'results/cifar10_resnet18_32x32_vos_e100_lr0.1_default/s0/best.ckpt' --mark=vos --evaluator.evaluator_args.layers_before_logits=2 --evaluator.evaluator_args.whiten_features=true --evaluator.evaluator_args.whitened_top_n_selection=false

python main.py --config configs/datasets/cifar10/cifar10.yml configs/networks/resnet18_32x32.yml configs/pipelines/test/test_bbr.yml configs/preprocessors/base_preprocessor.yml --network.checkpoint 'results/cifar10_resnet18_32x32_vos_e100_lr0.1_default/s0/best.ckpt' --mark=vos --evaluator.evaluator_args.layers_before_logits=2 --evaluator.evaluator_args.whiten_features=false --evaluator.evaluator_args.whitened_top_n_selection=false

# T2FNorm

python main.py --config configs/datasets/cifar10/cifar10.yml configs/networks/t2fnorm_net.yml configs/pipelines/test/test_bbr.yml configs/preprocessors/base_preprocessor.yml --network.checkpoint 'results/cifar10_t2fnorm_net_t2fnorm_e100_lr0.1_alpha0.1_default/s11033/best.ckpt' --mark=t2fnorm --evaluator.evaluator_args.layers_before_logits=2 --evaluator.evaluator_args.whiten_features=true --evaluator.evaluator_args.whitened_top_n_selection=true

# python main.py --config configs/datasets/cifar10/cifar10.yml configs/networks/t2fnorm_net.yml configs/pipelines/test/test_bbr.yml configs/preprocessors/base_preprocessor.yml --network.checkpoint 'results/cifar10_t2fnorm_net_t2fnorm_e100_lr0.1_alpha0.1_default/s11033/best.ckpt' --mark=t2fnorm --evaluator.evaluator_args.layers_before_logits=2 --evaluator.evaluator_args.whiten_features=true --evaluator.evaluator_args.whitened_top_n_selection=false

python main.py --config configs/datasets/cifar10/cifar10.yml configs/networks/t2fnorm_net.yml configs/pipelines/test/test_bbr.yml configs/preprocessors/base_preprocessor.yml --network.checkpoint 'results/cifar10_t2fnorm_net_t2fnorm_e100_lr0.1_alpha0.1_default/s11033/best.ckpt' --mark=t2fnorm --evaluator.evaluator_args.layers_before_logits=2 --evaluator.evaluator_args.whiten_features=false --evaluator.evaluator_args.whitened_top_n_selection=false

# DER

python main.py --config configs/datasets/cifar10/cifar10.yml configs/networks/resnet18_32x32.yml configs/pipelines/test/test_bbr.yml configs/preprocessors/base_preprocessor.yml --network.checkpoint 'results/cifar10_resnet18_32x32_der_e100_lr0.005_gamma0.2_beta0.5_clean_loss_included/s0/best.ckpt' --mark=der --evaluator.evaluator_args.layers_before_logits=2 --evaluator.evaluator_args.whiten_features=true --evaluator.evaluator_args.whitened_top_n_selection=true

# python main.py --config configs/datasets/cifar10/cifar10.yml configs/networks/resnet18_32x32.yml configs/pipelines/test/test_bbr.yml configs/preprocessors/base_preprocessor.yml --network.checkpoint 'results/cifar10_resnet18_32x32_der_e100_lr0.005_gamma0.2_beta0.5_clean_loss_included/s0/best.ckpt' --mark=der --evaluator.evaluator_args.layers_before_logits=2 --evaluator.evaluator_args.whiten_features=true --evaluator.evaluator_args.whitened_top_n_selection=false

python main.py --config configs/datasets/cifar10/cifar10.yml configs/networks/resnet18_32x32.yml configs/pipelines/test/test_bbr.yml configs/preprocessors/base_preprocessor.yml --network.checkpoint 'results/cifar10_resnet18_32x32_der_e100_lr0.005_gamma0.2_beta0.5_clean_loss_included/s0/best.ckpt' --mark=der --evaluator.evaluator_args.layers_before_logits=2 --evaluator.evaluator_args.whiten_features=false --evaluator.evaluator_args.whitened_top_n_selection=false

# ASCOOD

python main.py --config configs/datasets/cifar10/cifar10.yml configs/networks/ascood_net.yml configs/pipelines/test/test_bbr.yml configs/preprocessors/base_preprocessor.yml --network.checkpoint 'results/cifar10_ascood_net_ascood_e100_lr0.1_w1.0_p0.15_otype_shuffle_alpha_10.0_10.0_kl_div_True_default/s0/best.ckpt' --mark=ascood --evaluator.evaluator_args.layers_before_logits=2 --evaluator.evaluator_args.whiten_features=true --evaluator.evaluator_args.whitened_top_n_selection=true

# python main.py --config configs/datasets/cifar10/cifar10.yml configs/networks/ascood_net.yml configs/pipelines/test/test_bbr.yml configs/preprocessors/base_preprocessor.yml --network.checkpoint 'results/cifar10_ascood_net_ascood_e100_lr0.1_w1.0_p0.15_otype_shuffle_alpha_10.0_10.0_kl_div_True_default/s0/best.ckpt' --mark=ascood --evaluator.evaluator_args.layers_before_logits=2 --evaluator.evaluator_args.whiten_features=true --evaluator.evaluator_args.whitened_top_n_selection=false

python main.py --config configs/datasets/cifar10/cifar10.yml configs/networks/ascood_net.yml configs/pipelines/test/test_bbr.yml configs/preprocessors/base_preprocessor.yml --network.checkpoint 'results/cifar10_ascood_net_ascood_e100_lr0.1_w1.0_p0.15_otype_shuffle_alpha_10.0_10.0_kl_div_True_default/s0/best.ckpt' --mark=ascood --evaluator.evaluator_args.layers_before_logits=2 --evaluator.evaluator_args.whiten_features=false --evaluator.evaluator_args.whitened_top_n_selection=false
