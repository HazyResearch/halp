python -m unittest optim/bit_center_optim_test.py
python -m unittest layers/linear_layer_test.py
python -m unittest layers/cross_entropy_test.py
python -m unittest layers/conv_layer_test.py
python -m unittest layers/relu_layer_test.py
python -m unittest layers/pool_layer_test.py
python -m unittest layers/batch_norm_layer_test.py
python -m unittest models/logistic_regression_test.py
python -m unittest models/lenet_test.py
python -m unittest models/resnet_test.py
python -m unittest utils/utils.py

## eager mode test on logreg
printf "\n\n\n eager mode bc svrg comparing to non eager mode bc svrg in double mode on lenet, they should produce very similar values"
python run_models.py --n-epochs=3 --batch-size=128 --reg=5e-4 --alpha=0.01 --seed=1 --n-classes=10  --solver='bc-svrg' --rounding='void' -T=391 --dataset=mnist --model=logreg --cuda --double-debug | grep loss
python run_models.py --n-epochs=3 --batch-size=128 --reg=5e-4 --alpha=0.01 --seed=1 --n-classes=10  --solver='bc-svrg' --rounding='void' -T=391 --dataset=mnist --model=logreg --cuda --double-debug --on-site-compute | grep loss

## eager mode test on resnet
printf "\n\n\n eager mode bc svrg comparing to fp svrg in double mode on resnet, they should produce very similar values"
cd /dfs/scratch0/zjian/floating_halp/halp/halp/exp_script && python /dfs/scratch0/zjian/floating_halp/halp/halp/exp_script/run_models.py --double-debug --n-epochs=3 --batch-size=32 --reg=0.0005 --alpha=1.0 --momentum=0.9 --seed=1  --n-classes=10  --solver=bc-svrg  --rounding=void  -T=391  --dataset=cifar10  --model=resnet  --cuda  --resnet-load-ckpt --resnet-load-ckpt-epoch-id=300 --resnet-save-ckpt-path=/dfs/scratch0/zjian/floating_halp/exp_res/resnet_weight_saving_nov_30_backup/opt_lp-sgd_momentum_0.9_lr_0.1_l2_reg_0.0005_seed_1 --on-site-compute
cd /dfs/scratch0/zjian/floating_halp/halp/halp/exp_script && python /dfs/scratch0/zjian/floating_halp/halp/halp/exp_script/run_models.py --double-debug --n-epochs=3 --batch-size=32 --reg=0.0005 --alpha=1.0 --momentum=0.9 --seed=1  --n-classes=10  --solver=svrg  --rounding=void  -T=391  --dataset=cifar10  --model=resnet  --cuda  --resnet-load-ckpt --resnet-load-ckpt-epoch-id=300 --resnet-save-ckpt-path=/dfs/scratch0/zjian/floating_halp/exp_res/resnet_weight_saving_nov_30_backup/opt_lp-sgd_momentum_0.9_lr_0.1_l2_reg_0.0005_seed_1


## eager mode test on resnet
printf "\n\n\n eager mode bc svrg comparing to fp svrg in single precision debug mode on resnet, they should produce relatively similar values"
cd /dfs/scratch0/zjian/floating_halp/halp/halp/exp_script && python /dfs/scratch0/zjian/floating_halp/halp/halp/exp_script/run_models.py --float-debug --n-epochs=3 --batch-size=32 --reg=0.0005 --alpha=1.0 --momentum=0.9 --seed=1  --n-classes=10  --solver=bc-svrg  --rounding=void  -T=391  --dataset=cifar10  --model=resnet  --cuda  --resnet-load-ckpt --resnet-load-ckpt-epoch-id=300 --resnet-save-ckpt-path=/dfs/scratch0/zjian/floating_halp/exp_res/resnet_weight_saving_nov_30_backup/opt_lp-sgd_momentum_0.9_lr_0.1_l2_reg_0.0005_seed_1 --on-site-compute
cd /dfs/scratch0/zjian/floating_halp/halp/halp/exp_script && python /dfs/scratch0/zjian/floating_halp/halp/halp/exp_script/run_models.py --float-debug --n-epochs=3 --batch-size=32 --reg=0.0005 --alpha=1.0 --momentum=0.9 --seed=1  --n-classes=10  --solver=svrg  --rounding=void  -T=391  --dataset=cifar10  --model=resnet  --cuda  --resnet-load-ckpt --resnet-load-ckpt-epoch-id=300 --resnet-save-ckpt-path=/dfs/scratch0/zjian/floating_halp/exp_res/resnet_weight_saving_nov_30_backup/opt_lp-sgd_momentum_0.9_lr_0.1_l2_reg_0.0005_seed_1


printf "\n\n\n compare svrg and bc svrg results in double model. The following 2 run should generate almost the same test loss and accuracy\n"
python exp_script/run_models.py --n-epochs=2 --batch-size=128 --reg=0.0005 --alpha=0.1 --momentum=0.9 --seed=1  --n-classes=10  --solver=bc-svrg  --rounding=void  -T=391  --dataset=cifar10  --model=resnet  --cuda  --double-debug | grep loss
python exp_script/run_models.py --n-epochs=2 --batch-size=128 --reg=0.0005 --alpha=0.1 --momentum=0.9 --seed=1  --n-classes=10  --solver=svrg  --rounding=void  -T=391  --dataset=cifar10  --model=resnet  --cuda  --double-debug | grep loss


printf "\n\n\n toy sgd example for bc sgd, sgd, lp sgd behavior comparison, expect the numbers to be very close\n"
# test the behavior of bc sgd, we want the following three to have the same training loss history
python exp_script/run_models.py --n-epochs=3 --batch-size=1 --reg=0.000 --alpha=0.003 --seed=1 --n-classes=10  --solver='sgd' --cuda --debug-test | grep loss
python exp_script/run_models.py --n-epochs=3 --batch-size=1 --reg=0.000 --alpha=0.003 --seed=1 --n-classes=10  --solver='lp-sgd' --cuda --debug-test | grep loss
python exp_script/run_models.py --n-epochs=3 --batch-size=1 --reg=0.000 --alpha=0.003 --seed=1 --n-classes=10  --solver='bc-sgd' --cuda --debug-test | grep loss

printf "\n\n\n toy sgd example for bc svrg, svrg and lp svrg behavior comparison, expect the numbers to be very close\n"
# test the behavior of bc svrg, we want the following three to have the same training loss history
python exp_script/run_models.py --n-epochs=3 --batch-size=1 --reg=0.001 --alpha=0.003 --seed=1 --n-classes=10  --solver='svrg' --cuda --debug-test | grep loss
python exp_script/run_models.py --n-epochs=3 --batch-size=1 --reg=0.001 --alpha=0.003 --seed=1 --n-classes=10  --solver='lp-svrg' --cuda --debug-test | grep loss
python exp_script/run_models.py --n-epochs=3 --batch-size=1 --reg=0.001 --alpha=0.003 --seed=1 --n-classes=10  --solver='bc-svrg' --cuda --debug-test | grep loss

printf "\n\n\n fine tune test, the sgd based optimizer should produce similar number under double mode\n"
cd /dfs/scratch0/zjian/floating_halp/halp/halp/exp_script && python /dfs/scratch0/zjian/floating_halp/halp/halp/exp_script/run_models.py --double-debug --resnet-fine-tune --n-epochs=3 --batch-size=128 --reg=0.0005 --alpha=0.1 --momentum=0.9 --seed=1  --n-classes=10  --solver=bc-sgd  --rounding=void  -T=391  --dataset=cifar10  --model=resnet  --cuda  --resnet-load-ckpt --resnet-load-ckpt-epoch-id=300 --resnet-save-ckpt-path=/dfs/scratch0/zjian/floating_halp/exp_res/resnet_weight_saving_nov_30_backup/opt_lp-sgd_momentum_0.9_lr_0.1_l2_reg_0.0005_seed_1 | grep "Test\|epoch: 1 iter: 0"
echo "bc sgd done"
cd /dfs/scratch0/zjian/floating_halp/halp/halp/exp_script && python /dfs/scratch0/zjian/floating_halp/halp/halp/exp_script/run_models.py --double-debug --resnet-fine-tune --n-epochs=3 --batch-size=128 --reg=0.0005 --alpha=0.1 --momentum=0.9 --seed=1  --n-classes=10  --solver=lp-sgd  --rounding=void  -T=391  --dataset=cifar10  --model=resnet  --cuda  --resnet-load-ckpt --resnet-load-ckpt-epoch-id=300 --resnet-save-ckpt-path=/dfs/scratch0/zjian/floating_halp/exp_res/resnet_weight_saving_nov_30_backup/opt_lp-sgd_momentum_0.9_lr_0.1_l2_reg_0.0005_seed_1 | grep "Test\|epoch: 1 iter: 0"
echo "lp sgd done"
cd /dfs/scratch0/zjian/floating_halp/halp/halp/exp_script && python /dfs/scratch0/zjian/floating_halp/halp/halp/exp_script/run_models.py --double-debug --resnet-fine-tune --n-epochs=3 --batch-size=128 --reg=0.0005 --alpha=0.1 --momentum=0.9 --seed=1  --n-classes=10  --solver=sgd  --rounding=void  -T=391  --dataset=cifar10  --model=resnet  --cuda  --resnet-load-ckpt --resnet-load-ckpt-epoch-id=300 --resnet-save-ckpt-path=/dfs/scratch0/zjian/floating_halp/exp_res/resnet_weight_saving_nov_30_backup/opt_lp-sgd_momentum_0.9_lr_0.1_l2_reg_0.0005_seed_1 | grep "Test\|epoch: 1 iter: 0"
echo "sgd done"


printf "\n\n\n fine tune test, the svrg based optimizer should produce similar number under double mode"
cd /dfs/scratch0/zjian/floating_halp/halp/halp/exp_script && python /dfs/scratch0/zjian/floating_halp/halp/halp/exp_script/run_models.py --double-debug --resnet-fine-tune --n-epochs=3 --batch-size=128 --reg=0.0005 --alpha=0.1 --momentum=0.9 --seed=1  --n-classes=10  --solver=bc-svrg  --rounding=void  -T=391  --dataset=cifar10  --model=resnet  --cuda  --resnet-load-ckpt --resnet-load-ckpt-epoch-id=300 --resnet-save-ckpt-path=/dfs/scratch0/zjian/floating_halp/exp_res/resnet_weight_saving_nov_30_backup/opt_lp-sgd_momentum_0.9_lr_0.1_l2_reg_0.0005_seed_1 | grep "Test\|epoch: 1 iter: 0"
echo "bc svrg done"
 | grep "Test\|epoch: 1 iter: 0"cd /dfs/scratch0/zjian/floating_halp/halp/halp/exp_script && python /dfs/scratch0/zjian/floating_halp/halp/halp/exp_script/run_models.py --double-debug --resnet-fine-tune --n-epochs=3 --batch-size=128 --reg=0.0005 --alpha=0.1 --momentum=0.9 --seed=1  --n-classes=10  --solver=lp-svrg  --rounding=void  -T=391  --dataset=cifar10  --model=resnet  --cuda  --resnet-load-ckpt --resnet-load-ckpt-epoch-id=300 --resnet-save-ckpt-path=/dfs/scratch0/zjian/floating_halp/exp_res/resnet_weight_saving_nov_30_backup/opt_lp-sgd_momentum_0.9_lr_0.1_l2_reg_0.0005_seed_1
echo "lp svrg done"
cd /dfs/scratch0/zjian/floating_halp/halp/halp/exp_script && python /dfs/scratch0/zjian/floating_halp/halp/halp/exp_script/run_models.py --double-debug --resnet-fine-tune --n-epochs=3 --batch-size=128 --reg=0.0005 --alpha=0.1 --momentum=0.9 --seed=1  --n-classes=10  --solver=svrg  --rounding=void  -T=391  --dataset=cifar10  --model=resnet  --cuda  --resnet-load-ckpt --resnet-load-ckpt-epoch-id=300 --resnet-save-ckpt-path=/dfs/scratch0/zjian/floating_halp/exp_res/resnet_weight_saving_nov_30_backup/opt_lp-sgd_momentum_0.9_lr_0.1_l2_reg_0.0005_seed_1 | grep "Test\|epoch: 1 iter: 0"
echo "svrg done"


printf "\n\n\n fine tune real precision test, the sgd based optimizer should roughly produce similar number under double mode\n"
cd /dfs/scratch0/zjian/floating_halp/halp/halp/exp_script && python /dfs/scratch0/zjian/floating_halp/halp/halp/exp_script/run_models.py --resnet-fine-tune --n-epochs=2 --batch-size=128 --reg=0.0005 --alpha=0.1 --momentum=0.9 --seed=1  --n-classes=10  --solver=bc-sgd  --rounding=near  -T=391  --dataset=cifar10  --model=resnet  --cuda  --resnet-load-ckpt --resnet-load-ckpt-epoch-id=300 --resnet-save-ckpt-path=/dfs/scratch0/zjian/floating_halp/exp_res/resnet_weight_saving_nov_30_backup/opt_lp-sgd_momentum_0.9_lr_0.1_l2_reg_0.0005_seed_1 | grep "Test\|epoch: 1 iter: 0\|epoch: 0 iter: 1 "
echo "bc sgd done"
cd /dfs/scratch0/zjian/floating_halp/halp/halp/exp_script && python /dfs/scratch0/zjian/floating_halp/halp/halp/exp_script/run_models.py --resnet-fine-tune --n-epochs=2 --batch-size=128 --reg=0.0005 --alpha=0.1 --momentum=0.9 --seed=1  --n-classes=10  --solver=lp-sgd  --rounding=near  -T=391  --dataset=cifar10  --model=resnet  --cuda  --resnet-load-ckpt --resnet-load-ckpt-epoch-id=300 --resnet-save-ckpt-path=/dfs/scratch0/zjian/floating_halp/exp_res/resnet_weight_saving_nov_30_backup/opt_lp-sgd_momentum_0.9_lr_0.1_l2_reg_0.0005_seed_1 | grep "Test\|epoch: 1 iter: 0\|epoch: 0 iter: 1 "
echo "lp sgd done"
cd /dfs/scratch0/zjian/floating_halp/halp/halp/exp_script && python /dfs/scratch0/zjian/floating_halp/halp/halp/exp_script/run_models.py --resnet-fine-tune --n-epochs=2 --batch-size=128 --reg=0.0005 --alpha=0.1 --momentum=0.9 --seed=1  --n-classes=10  --solver=sgd  --rounding=void  -T=391  --dataset=cifar10  --model=resnet  --cuda  --resnet-load-ckpt --resnet-load-ckpt-epoch-id=300 --resnet-save-ckpt-path=/dfs/scratch0/zjian/floating_halp/exp_res/resnet_weight_saving_nov_30_backup/opt_lp-sgd_momentum_0.9_lr_0.1_l2_reg_0.0005_seed_1 | grep "Test\|epoch: 1 iter: 0\|epoch: 0 iter: 1 "
echo "sgd done"

printf "\n\n\n fine tune real precision test, the svrg based optimizer should roughly produce similar number under double mode\n"
cd /dfs/scratch0/zjian/floating_halp/halp/halp/exp_script && python /dfs/scratch0/zjian/floating_halp/halp/halp/exp_script/run_models.py --resnet-fine-tune --n-epochs=2 --batch-size=128 --reg=0.0005 --alpha=0.1 --momentum=0.9 --seed=1  --n-classes=10  --solver=bc-svrg  --rounding=near  -T=391  --dataset=cifar10  --model=resnet  --cuda  --resnet-load-ckpt --resnet-load-ckpt-epoch-id=300 --resnet-save-ckpt-path=/dfs/scratch0/zjian/floating_halp/exp_res/resnet_weight_saving_nov_30_backup/opt_lp-sgd_momentum_0.9_lr_0.1_l2_reg_0.0005_seed_1 | grep "Test\|epoch: 1 iter: 0\|epoch: 0 iter: 1 "
echo "bc svrg done"
cd /dfs/scratch0/zjian/floating_halp/halp/halp/exp_script && python /dfs/scratch0/zjian/floating_halp/halp/halp/exp_script/run_models.py --resnet-fine-tune --n-epochs=2 --batch-size=128 --reg=0.0005 --alpha=0.1 --momentum=0.9 --seed=1  --n-classes=10  --solver=lp-svrg  --rounding=near  -T=391  --dataset=cifar10  --model=resnet  --cuda  --resnet-load-ckpt --resnet-load-ckpt-epoch-id=300 --resnet-save-ckpt-path=/dfs/scratch0/zjian/floating_halp/exp_res/resnet_weight_saving_nov_30_backup/opt_lp-sgd_momentum_0.9_lr_0.1_l2_reg_0.0005_seed_1 | grep "Test\|epoch: 1 iter: 0\|epoch: 0 iter: 1 "
echo "lp svrg done"
cd /dfs/scratch0/zjian/floating_halp/halp/halp/exp_script && python /dfs/scratch0/zjian/floating_halp/halp/halp/exp_script/run_models.py --resnet-fine-tune --n-epochs=2 --batch-size=128 --reg=0.0005 --alpha=0.1 --momentum=0.9 --seed=1  --n-classes=10  --solver=svrg  --rounding=void  -T=391  --dataset=cifar10  --model=resnet  --cuda  --resnet-load-ckpt --resnet-load-ckpt-epoch-id=300 --resnet-save-ckpt-path=/dfs/scratch0/zjian/floating_halp/exp_res/resnet_weight_saving_nov_30_backup/opt_lp-sgd_momentum_0.9_lr_0.1_l2_reg_0.0005_seed_1 | grep "Test\|epoch: 1 iter: 0\|epoch: 0 iter: 1 "
echo "svrg done"
