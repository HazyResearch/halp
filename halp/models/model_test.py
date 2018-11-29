import torch
import numpy as np
from torch.autograd import Variable
from halp.utils.test_utils import assert_model_grad_equal
from halp.utils.utils import single_to_half_det, single_to_half_stoc, void_cast_func
from halp.utils.utils import copy_model_weights, set_seed
from halp.utils.test_utils import HalpTest


class BitCenterModelTest(HalpTest):
    """
    Test whether it gives the same fw bw output given the same input
    In this test, we compare to the original LeNet implementation
    We test with a epoch with 2 minibatches, we compare the output
    between fp32 model and fp32 bc model
    """

    def get_config(self):
        pass

    def get_models(self, n_minibatch, batch_size, n_class):
        pass

    def get_inputs(self, n_minibatch, batch_size, n_class):
        pass

    def test_fw_bw_output(self):
        set_seed(0)
        config = self.get_config()
        native_model, fp_model, lp_model, bc_model = self.get_models(**config)
        x_list, y_list = self.get_inputs(**config)
        # check fp forward
        criterion = torch.nn.CrossEntropyLoss() # for native model
        bc_model.set_mode(do_offset=True)
        n_minibatch = config["n_minibatch"]
        for i in range(n_minibatch):
            # print("input dtype ", x_list[i].dtype)
            output_native = native_model(x_list[i])
            loss_native = criterion(output_native, y_list[i]).detach()
            loss_fp = fp_model(x_list[i], y_list[i]).detach()
            loss_lp = lp_model(x_list[i], y_list[i]).detach()
            loss_bc = bc_model(x_list[i], y_list[i])
            loss_bc.backward()
            self.check_layer_status(bc_model, do_offset=True)
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
            self.check_layer_status(bc_model, do_offset=False)
            loss_bc = bc_model(torch.zeros_like(x_list[i]), y_list[i])
            self.check_layer_status(bc_model, do_offset=False)
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
