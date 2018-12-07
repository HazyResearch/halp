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

# the following 2 run should generate almost the same test loss and accuracy
python exp_script/run_models.py --n-epochs=2 --batch-size=128 --reg=0.0005 --alpha=0.1 --momentum=0.9 --seed=1  --n-classes=10  --solver=bc-svrg  --rounding=void  -T=391  --dataset=cifar10  --model=resnet  --cuda  --double-debug | grep loss
python exp_script/run_models.py --n-epochs=2 --batch-size=128 --reg=0.0005 --alpha=0.1 --momentum=0.9 --seed=1  --n-classes=10  --solver=svrg  --rounding=void  -T=391  --dataset=cifar10  --model=resnet  --cuda  --double-debug | grep loss


echo "toy sgd example for bc sgd, sgd, lp sgd behavior comparison, expect the numbers to be very close"
# test the behavior of bc sgd, we want the following three to have the same training loss history
python exp_script/run_models.py --n-epochs=3 --batch-size=1 --reg=0.000 --alpha=0.003 --seed=1 --n-classes=10  --solver='sgd' --cuda --debug-test | grep loss
python exp_script/run_models.py --n-epochs=3 --batch-size=1 --reg=0.000 --alpha=0.003 --seed=1 --n-classes=10  --solver='lp-sgd' --cuda --debug-test | grep loss
python exp_script/run_models.py --n-epochs=3 --batch-size=1 --reg=0.000 --alpha=0.003 --seed=1 --n-classes=10  --solver='bc-sgd' --cuda --debug-test | grep loss

echo "toy sgd example for bc svrg, svrg and lp svrg behavior comparison, expect the numbers to be very close"
# test the behavior of bc svrg, we want the following three to have the same training loss history
python exp_script/run_models.py --n-epochs=3 --batch-size=1 --reg=0.001 --alpha=0.003 --seed=1 --n-classes=10  --solver='svrg' --cuda --debug-test | grep loss
python exp_script/run_models.py --n-epochs=3 --batch-size=1 --reg=0.001 --alpha=0.003 --seed=1 --n-classes=10  --solver='lp-svrg' --cuda --debug-test | grep loss
python exp_script/run_models.py --n-epochs=3 --batch-size=1 --reg=0.001 --alpha=0.003 --seed=1 --n-classes=10  --solver='bc-svrg' --cuda --debug-test | grep loss
