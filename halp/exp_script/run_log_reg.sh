# python mnist_log_reg.py --n-epochs=100 --batch-size=1 --reg=0.0 --alpha=0.001 --seed=1 --n-classes=10  --solver='sgd' --cuda
# python mnist_log_reg.py --n-epochs=100 --batch-size=100 --reg=0.03 --alpha=0.005 --seed=1 --n-classes=10  --solver='svrg' --cuda -T=600
# python mnist_log_reg.py --n-epochs=100 --batch-size=1 --reg=0.009 --alpha=0.003 --seed=1 --n-classes=10  --solver='lp-sgd' --cuda
# python mnist_log_reg.py --n-epochs=100 --batch-size=100 --reg=0.03 --alpha=0.005 --seed=1 --n-classes=10  --solver='lp-svrg' --cuda -T=600
# python mnist_log_reg.py --n-epochs=100 --batch-size=1 --reg=0.002 --alpha=0.002 --seed=1 --n-classes=10  --solver='bc-sgd' --cuda -T=60000
# python mnist_log_reg.py --n-epochs=100 --batch-size=1 --reg=0.009 --alpha=0.003 --seed=1 --n-classes=10  --solver='bc-sgd' --cuda -T=60000
# python mnist_log_reg.py --n-epochs=100 --batch-size=100 --reg=0.03 --alpha=0.005 --seed=1 --n-classes=10  --solver='bc-svrg' --cuda -T=600


# perform sgd with CA's config
# python mnist_log_reg.py --n-epochs=10 --batch-size=1 --reg=0.009 --alpha=0.003 --seed=1 --n-classes=10  --solver='lp-sgd' --cuda 2>&1 | tee log/ca_sgd_fp16.log

# python mnist_log_reg.py --n-epochs=10 --batch-size=1 --reg=0.002 --alpha=0.002 --seed=1 --n-classes=10  --solver='bc-sgd' -T=60000 --cuda 2>&1 | tee log/ca_bc_sgd_fp16.log

# python mnist_log_reg.py --n-epochs=10 --batch-size=1 --reg=0.03 --alpha=0.005 --seed=1 --n-classes=10  --solver='bc-svrg' -T=60000 --cuda 2>&1 | tee log/ca_bc_svrg_fp16.log


# perform svrg with CA's config
# python mnist_log_reg.py --n-epochs=10 --batch-size=1 --reg=0.009 --alpha=0.003 --seed=1 --n-classes=10  --solver='bc-sgd' -T=60000 --cuda 2>&1 | tee log/cmp_ca_bc_sgd_fp16.log

# python mnist_log_reg.py --n-epochs=10 --batch-size=1 --reg=0.009 --alpha=0.003 --seed=1 --n-classes=10  --solver='bc-svrg' -T=60000 --cuda 2>&1 | tee log/cmp_ca_bc_svrg_fp16.log

# python mnist_log_reg.py --n-epochs=10 --batch-size=1 --reg=0.009 --alpha=0.003 --seed=1 --n-classes=10  --solver='sgd' --cuda 2>&1 | tee log/cmp_ca_sgd_fp32.log

# python mnist_log_reg.py --n-epochs=10 --batch-size=1 --reg=0.000 --alpha=0.001 --seed=1 --n-classes=10  --solver='sgd' --cuda 2>&1 | tee log/cmp_ca_sgd_fp32_second_config.log


# # investigate why stochastic does not bring big improvement of performance
# python mnist_log_reg.py --n-epochs=10 --batch-size=1 --reg=0.00009 --alpha=0.003 --seed=1 --n-classes=10  --solver='bc-svrg' --rounding='stoc' -T=60000 --cuda 2>&1 | tee log/ca_bc_svrg_fp16_cmp_stoc_full_full_grad.log

# python mnist_log_reg.py --n-epochs=10 --batch-size=1 --reg=0.00009 --alpha=0.003 --seed=1 --n-classes=10  --solver='bc-svrg' --rounding='near' -T=60000 --cuda 2>&1 | tee log/ca_bc_svrg_fp16_cmp_near.log

# python mnist_log_reg.py --n-epochs=10 --batch-size=1 --reg=0.00009 --alpha=0.003 --seed=1 --n-classes=10  --solver='bc-svrg' --rounding='stoc' -T=60000 --cuda 2>&1 | tee log/ca_bc_svrg_fp16_cmp_stoc.log

# python mnist_log_reg.py --n-epochs=10 --batch-size=1 --reg=0.00009 --alpha=0.003 --seed=1 --n-classes=10  --solver='bc-svrg' --rounding='void' -T=60000 --cuda 2>&1 | tee log/ca_bc_svrg_fp16_cmp_void.log

# python mnist_log_reg.py --n-epochs=10 --batch-size=1 --reg=0.00009 --alpha=0.003 --seed=1 --n-classes=10  --solver='lp-svrg' --rounding='near' -T=60000 --cuda 2>&1 | tee log/ca_lp_svrg_fp16_cmp_near.log

# python mnist_log_reg.py --n-epochs=10 --batch-size=1 --reg=0.00009 --alpha=0.003 --seed=1 --n-classes=10  --solver='lp-svrg' --rounding='stoc' -T=60000 --cuda 2>&1 | tee log/ca_lp_svrg_fp16_cmp_stoc.log

# python mnist_log_reg.py --n-epochs=10 --batch-size=1 --reg=0.00009 --alpha=0.003 --seed=1 --n-classes=10  --solver='lp-sgd' --rounding='near' -T=60000 --cuda 2>&1 | tee log/ca_lp_sgd_fp16_cmp_near.log

# python mnist_log_reg.py --n-epochs=10 --batch-size=1 --reg=0.00009 --alpha=0.003 --seed=1 --n-classes=10  --solver='bc-sgd' --rounding='near' -T=60000 --cuda 2>&1 | tee log/ca_bc_sgd_fp16_cmp_near.log

# python mnist_log_reg.py --n-epochs=10 --batch-size=1 --reg=0.00009 --alpha=0.003 --seed=1 --n-classes=10  --solver='sgd' --rounding='void' -T=60000 --cuda 2>&1 | tee log/ca_fp_sgd_fp32_cmp_near.log

# python mnist_log_reg.py --n-epochs=10 --batch-size=1 --reg=0.00009 --alpha=0.003 --seed=1 --n-classes=10  --solver='svrg' -T=60000 --cuda 2>&1 | tee log/ca_bc_svrg_fp32_cmp.log


# # python mnist_log_reg.py --n-epochs=10 --batch-size=1 --reg=0.00009 --alpha=0.003 --seed=1 --n-classes=10  --solver='lp-sgd' --rounding='void' -T=60000 --cuda 2>&1 | tee log/ca_lp_sgd_fp16_cmp_void.log



# # python mnist_log_reg.py --n-epochs=10 --batch-size=100 --reg=0.00009 --alpha=0.003 --seed=1 --n-classes=10  --solver='bc-svrg' --rounding='stoc' -T=600 --cuda 2>&1 | tee log/test.log


# # # # bc-sgd with a random setting
# # python mnist_log_reg.py --n-epochs=10 --batch-size=1 --reg=0.002 --alpha=0.002 --seed=1 --n-classes=10  --solver='bc-sgd' --rounding='stoc' -T=60000 --cuda 2>&1 | tee log/ca_bc_sgd_fp16_stoc.log
# # python mnist_log_reg.py --n-epochs=10 --batch-size=1 --reg=0.002 --alpha=0.002 --seed=1 --n-classes=10  --solver='bc-sgd' --rounding='near' -T=60000 --cuda 2>&1 | tee log/ca_bc_sgd_fp16_near.log
# # python mnist_log_reg.py --n-epochs=10 --batch-size=1 --reg=0.002 --alpha=0.002 --seed=1 --n-classes=10  --solver='bc-sgd' --rounding='void' -T=60000 --cuda 2>&1 | tee log/ca_bc_sgd_fp16_void.log
# # python mnist_log_reg.py --n-epochs=10 --batch-size=1 --reg=0.002 --alpha=0.002 --seed=1 --n-classes=10  --solver='lp-sgd' --rounding='void' -T=60000 --cuda 2>&1 | tee log/ca_lp_sgd_fp16_void.log


# timing experiments
# python mnist_log_reg.py --n-epochs=5 --batch-size=1 --reg=0.00009 --alpha=0.003 --seed=1 --n-classes=10  --solver='sgd' --rounding='void' -T=60000 --cuda 2>&1 | tee log/ca_lp_sgd_fp32_timing_batch_1.log
# python mnist_log_reg.py --n-epochs=10 --batch-size=100 --reg=0.00009 --alpha=0.003 --seed=1 --n-classes=10  --solver='sgd' --rounding='void' -T=600 --cuda 2>&1 | tee log/ca_lp_sgd_fp32_timing_batch_100.log

# python mnist_log_reg.py --n-epochs=5 --batch-size=1 --reg=0.00009 --alpha=0.003 --seed=1 --n-classes=10  --solver='lp-sgd' --rounding='void' -T=60000 --cuda 2>&1 | tee log/ca_lp_sgd_fp16_timing_batch_1.log
# python mnist_log_reg.py --n-epochs=10 --batch-size=100 --reg=0.00009 --alpha=0.003 --seed=1 --n-classes=10  --solver='lp-sgd' --rounding='void' -T=60000 --cuda 2>&1 | tee log/ca_lp_sgd_fp16_timing_batch_100.log

# # python mnist_log_reg.py --n-epochs=5 --batch-size=1 --reg=0.00009 --alpha=0.003 --seed=1 --n-classes=10  --solver='bc-svrg' --rounding='near' -T=60000 --cuda 2>&1 | tee log/ca_bc_svrg_fp16_timing_batch_1.log
# python mnist_log_reg.py --n-epochs=10 --batch-size=100 --reg=0.00009 --alpha=0.003 --seed=1 --n-classes=10  --solver='bc-svrg' --rounding='near' -T=600 --cuda 2>&1 | tee log/ca_bc_svrg_fp16_timing_batch_100.log

# # python mnist_log_reg.py --n-epochs=5 --batch-size=1 --reg=0.00009 --alpha=0.003 --seed=1 --n-classes=10  --solver='bc-sgd' --rounding='near' -T=60000 --cuda 2>&1 | tee log/ca_bc_sgd_fp16_timing_batch_1.log
# python mnist_log_reg.py --n-epochs=10 --batch-size=100 --reg=0.00009 --alpha=0.003 --seed=1 --n-classes=10  --solver='bc-sgd' --rounding='near' -T=600 --cuda 2>&1 | tee log/ca_bc_sgd_fp16_timing_batch_100.log

## test run for mnist experiments
# python mnist_log_reg.py --n-epochs=10 --batch-size=128 --reg=5e-4 --alpha=0.01 --seed=1 --n-classes=10  --solver='bc-sgd' --rounding='near' -T=469 --cuda 2>&1

# python mnist_log_reg.py --n-epochs=10 --batch-size=128 --reg=5e-4 --alpha=0.01 --seed=1 --n-classes=10  --solver='lp-sgd' --rounding='near' -T=391 --dataset=cifar10 --model=lenet --cuda 2>&1

# python mnist_log_reg.py --n-epochs=10 --batch-size=128 --reg=5e-4 --alpha=0.01 --seed=1 --n-classes=10  --solver='lp-sgd' --rounding='void' -T=391 --dataset=cifar10 --model=lenet --cuda 2>&1

# LeNet verification runs
# python run_models.py --n-epochs=100 --batch-size=128 --reg=5e-4 --alpha=0.01 --seed=1 --n-classes=10  --solver='sgd' --rounding='void' -T=391 --dataset=cifar10 --model=lenet --cuda 2>&1 | tee test/lenet_fp32_sgd.log
# python run_models.py --n-epochs=100 --batch-size=128 --reg=5e-4 --alpha=0.01 --seed=1 --n-classes=10  --solver='lp-sgd' --rounding='near' -T=391 --dataset=cifar10 --model=lenet --cuda 2>&1 | tee test/lenet_fp16_sgd.log
# python run_models.py --n-epochs=100 --batch-size=128 --reg=5e-4 --alpha=0.01 --seed=1 --n-classes=10  --solver='bc-sgd' --rounding='near' -T=391 --dataset=cifar10 --model=lenet --cuda 2>&1 | tee test/lenet_fp16_bc_sgd_near.log

# python run_models.py --n-epochs=100 --batch-size=128 --reg=5e-4 --alpha=0.01 --seed=1 --n-classes=10  --solver='svrg' --rounding='near' -T=391 --dataset=cifar10 --model=lenet --cuda 2>&1 | tee test/lenet_fp32_svrg.log
# python run_models.py --n-epochs=100 --batch-size=128 --reg=5e-4 --alpha=0.01 --seed=1 --n-classes=10  --solver='lp-svrg' --rounding='near' -T=391 --dataset=cifar10 --model=lenet --cuda 2>&1 | tee test/lenet_fp16_svrg.log
# python run_models.py --n-epochs=100 --batch-size=128 --reg=5e-4 --alpha=0.01 --seed=1 --n-classes=10  --solver='bc-svrg' --rounding='near' -T=391 --dataset=cifar10 --model=lenet --cuda 2>&1 | tee test/lenet_fp16_bc_svrg_near.log


python /dfs/scratch0/zjian/floating_halp/halp/halp/exp_script/run_models.py --n-epochs=100 --batch-size=128 --reg=0.0 --alpha=0.05 --momentum=0.0 --seed=1  --n-classes=10  --solver=svrg  --rounding=void  -T=391  --dataset=cifar10  --model=lenet  --cuda

python /dfs/scratch0/zjian/floating_halp/halp/halp/exp_script/run_models.py --n-epochs=100 --batch-size=128 --reg=0.0 --alpha=0.05 --momentum=0.0 --seed=1  --n-classes=10  --solver=bc-svrg  --rounding=void  -T=391  --dataset=cifar10  --model=lenet  --cuda

python /dfs/scratch0/zjian/floating_halp/halp/halp/exp_script/run_models.py --n-epochs=100 --batch-size=128 --reg=0.5 --alpha=0.05 --momentum=0.0 --seed=1  --n-classes=10  --solver=sgd  --rounding=void  -T=391  --dataset=cifar10  --model=lenet  --cuda

python /dfs/scratch0/zjian/floating_halp/halp/halp/exp_script/run_models.py --n-epochs=100 --batch-size=128 --reg=0.5 --alpha=0.05 --momentum=0.0 --seed=1  --n-classes=10  --solver=bc-sgd  --rounding=void  -T=391  --dataset=cifar10  --model=lenet  --cuda




