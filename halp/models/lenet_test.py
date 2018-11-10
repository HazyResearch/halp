import torch
import numpy as np
from torch.autograd import Variable
from halp.utils.test_utils import assert_model_grad_equal
from halp.utils.utils import single_to_half_det, single_to_half_stoc, void_cast_func
from halp.utils.utils import copy_model_weights, set_seed
from unittest import TestCase
from halp.models.logistic_regression import LogisticRegression
from halp.utils.test_utils import HalpTest
from halp.models.lenet import LeNet_PyTorch, LeNet


# Test whether it gives the same fw bw output given the same input
# In this test, we compare to the original LeNet implementation
# We test with a epoch with 2 minibatches, we compare the output
# between fp32 model and fp32 bc model
class LeNetTest(HalpTest, TestCase):
    """
    Test whether it gives the same fw bw output given the same input
    In this test, we compare to the original LeNet implementation
    We test with a epoch with 2 minibatches, we compare the output
    between fp32 model and fp32 bc model
    """

    def test_fw_bw_output(self):
        batch_size = 5
        n_minibatch = 6
        n_class = 10
        n_train_sample = batch_size * n_minibatch
        set_seed(0)
        native_model = LeNet_PyTorch().cuda().double()
        fp_model = LeNet(
            cast_func=void_cast_func,
            n_train_sample=n_train_sample,
            dtype="fp").cuda().double()
        copy_model_weights(native_model, fp_model)
        lp_model = LeNet(
            cast_func=void_cast_func,
            n_train_sample=n_train_sample,
            dtype="lp").cuda().double()
        copy_model_weights(native_model, lp_model)
        bc_model = LeNet(
            cast_func=void_cast_func,
            n_train_sample=n_train_sample,
            dtype="bc").double()
        copy_model_weights(native_model, bc_model)
        # compare native LeNet model and fp mode LeNet in our implementation
        # fp model LeNet and bc fp model LeNet in our implementation
        x_list = []
        y_list = []
        for i in range(n_minibatch):
            x_list.append(
                torch.nn.Parameter(
                    torch.randn(batch_size, 3, 32, 32,
                                dtype=torch.float).cuda(),
                    requires_grad=True).cuda().double())
            y_list.append(torch.LongTensor(batch_size).random_(n_class).cuda())
        # check fp forward
        criterion = torch.nn.CrossEntropyLoss()
        bc_model.set_mode(do_offset=True)
        for i in range(n_minibatch):
            # print("input dtype ", x_list[i].dtype)
            output_native = native_model(x_list[i])
            loss_native = criterion(output_native, y_list[i]).detach()
            loss_fp = fp_model(x_list[i], y_list[i]).detach()
            loss_lp = lp_model(x_list[i], y_list[i]).detach()
            loss_bc = bc_model(x_list[i], y_list[i])
            loss_bc.backward()
            bc_model.check_layer_status(do_offset=True)
            # print("fp loss ", loss_native.item(), loss_fp.item(), loss_lp.item(), loss_bc.item())
            np.testing.assert_allclose(
                np.array(loss_native.item()), np.array(loss_fp.item()))
            np.testing.assert_allclose(
                np.array(loss_fp.item()), np.array(loss_lp.item()))
            np.testing.assert_allclose(
                np.array(loss_lp.item()), np.array(loss_bc.item()))

        bc_model.set_mode(do_offset=False)
        for i in range(n_minibatch):
            output_native = native_model(x_list[i])
            loss_native = criterion(output_native, y_list[i])
            loss_native.backward()
            loss_fp = fp_model(x_list[i], y_list[i])
            loss_fp.backward()
            loss_lp = lp_model(x_list[i], y_list[i])
            loss_lp.backward()
            bc_model.check_layer_status(do_offset=False)
            loss_bc = bc_model(torch.zeros_like(x_list[i]), y_list[i])
            bc_model.check_layer_status(do_offset=False)
            loss_bc.backward()
            np.testing.assert_allclose(
                np.array(loss_native.item()), np.array(loss_fp.item()))
            np.testing.assert_allclose(
                np.array(loss_fp.item()), np.array(loss_lp.item()))
            np.testing.assert_allclose(
                np.array(loss_lp.item()), np.array(loss_bc.item()))
            if i == n_minibatch - 1:
                # we only test the gradient for the last minibatch because the gradient offset
                # changes across minibatch
                assert_model_grad_equal(native_model, fp_model)
                assert_model_grad_equal(fp_model, lp_model)
                assert_model_grad_equal(lp_model, bc_model, model2_is_bc=True)


if __name__ == "__main__":
    print(torch.__version__)
    unittest.main()
