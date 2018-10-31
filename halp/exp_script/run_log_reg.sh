# python mnist_log_reg.py --n-epochs=100 --batch-size=1 --reg=0.0 --alpha=0.001 --seed=1 --n-classes=10  --solver='sgd' --cuda
# python mnist_log_reg.py --n-epochs=100 --batch-size=100 --reg=0.03 --alpha=0.005 --seed=1 --n-classes=10  --solver='svrg' --cuda -T=600
# python mnist_log_reg.py --n-epochs=100 --batch-size=1 --reg=0.009 --alpha=0.003 --seed=1 --n-classes=10  --solver='lp-sgd' --cuda
# python mnist_log_reg.py --n-epochs=100 --batch-size=100 --reg=0.03 --alpha=0.005 --seed=1 --n-classes=10  --solver='lp-svrg' --cuda -T=600
# python mnist_log_reg.py --n-epochs=100 --batch-size=1 --reg=0.002 --alpha=0.002 --seed=1 --n-classes=10  --solver='bc-sgd' --cuda -T=60000
# python mnist_log_reg.py --n-epochs=100 --batch-size=1 --reg=0.009 --alpha=0.003 --seed=1 --n-classes=10  --solver='bc-sgd' --cuda -T=60000
# python mnist_log_reg.py --n-epochs=100 --batch-size=100 --reg=0.03 --alpha=0.005 --seed=1 --n-classes=10  --solver='bc-svrg' --cuda -T=600

# python mnist_log_reg.py --n-epochs=100 --batch-size=1 --reg=0.000 --alpha=0.003 --seed=1 --n-classes=10  --solver='lp-sgd' --cuda

# python mnist_log_reg.py --n-epochs=100 --batch-size=1 --reg=0.000 --alpha=0.003 --seed=1 --n-classes=10  --solver='sgd' --cuda
# python mnist_log_reg.py --n-epochs=100 --batch-size=1 --reg=0.000 --alpha=0.003 --seed=1 --n-classes=10  --solver='bc-sgd' --cuda -T=60000

# python mnist_log_reg.py --n-epochs=3 --batch-size=1 --reg=0.000 --alpha=0.003 --seed=1 --n-classes=10  --solver='lp-sgd' --cuda --debug-test
# python mnist_log_reg.py --n-epochs=3 --batch-size=1 --reg=0.000 --alpha=0.003 --seed=1 --n-classes=10  --solver='bc-sgd' --cuda --debug-test

# python mnist_log_reg.py --n-epochs=3 --batch-size=1 --reg=0.000 --alpha=0.003 --seed=1 --n-classes=10  --solver='lp-svrg' --cuda --debug-test
# python mnist_log_reg.py --n-epochs=3 --batch-size=1 --reg=0.000 --alpha=0.003 --seed=1 --n-classes=10  --solver='bc-sgd' --cuda --debug-test

