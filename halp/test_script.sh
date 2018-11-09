python -m unittest optim/bit_center_optim_test.py
python -m unittest layers/linear_layer_test.py
python -m unittest layers/cross_entropy_test.py
python -m unittest layers/conv_layer_test.py
python -m unittest layers/relu_layer_test.py
python -m unittest layers/max_pool_layer_test.py
python -m unittest utils/utils.py

echo "toy sgd example for bc sgd, sgd, lp sgd behavior comparison, expect the numbers to be very close"
# test the behavior of bc sgd, we want the following three to have the same training loss history
python exp_script/mnist_log_reg.py --n-epochs=3 --batch-size=1 --reg=0.000 --alpha=0.003 --seed=1 --n-classes=10  --solver='sgd' --cuda --debug-test
python exp_script/mnist_log_reg.py --n-epochs=3 --batch-size=1 --reg=0.000 --alpha=0.003 --seed=1 --n-classes=10  --solver='lp-sgd' --cuda --debug-test
python exp_script/mnist_log_reg.py --n-epochs=3 --batch-size=1 --reg=0.000 --alpha=0.003 --seed=1 --n-classes=10  --solver='bc-sgd' --cuda --debug-test

echo "toy sgd example for bc svrg, svrg and lp svrg behavior comparison, expect the numbers to be very close"
# test the behavior of bc svrg, we want the following three to have the same training loss history
python exp_script/mnist_log_reg.py --n-epochs=3 --batch-size=1 --reg=0.001 --alpha=0.003 --seed=1 --n-classes=10  --solver='svrg' --cuda --debug-test
python exp_script/mnist_log_reg.py --n-epochs=3 --batch-size=1 --reg=0.001 --alpha=0.003 --seed=1 --n-classes=10  --solver='lp-svrg' --cuda --debug-test
python exp_script/mnist_log_reg.py --n-epochs=3 --batch-size=1 --reg=0.001 --alpha=0.003 --seed=1 --n-classes=10  --solver='bc-svrg' --cuda --debug-test
