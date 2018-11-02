import torch
import numpy as np
import ctypes


def single_to_half_det(tensor):
    return tensor.half()

def single_to_half_stoc(tensor):
    value = tensor.clone().cpu().numpy()
    value = np.ascontiguousarray(value)
    value_shape = value.shape
    assert tensor.dtype == torch.float

    print("float value ", value)

    value_ptr = value.ctypes.data_as(ctypes.POINTER(ctypes.c_int32))
    value_int = np.ctypeslib.as_array(value_ptr, shape=value_shape)

    print("ckpt 1 ", np.binary_repr(value_int[0, 0], width=32))

    # # generating exponent by shift to the right. There
    # # are 23 bit of mentissa. The sign bit is kept
    # # print(value, int("0xFF800000", 16))
    sign = np.bitwise_and(value_int, int("0x80000000", 16))
    exponent = np.bitwise_and(value_int, int("0x7F800000", 16))
    mantissa = np.bitwise_and(value_int, int("0x007FFFFF", 16))
    mantissa_rand = np.random.random_integers(
        low=0, high=int("0x7FFFFFFF", 16), size=value.shape)
    # only keeping the last 13 bit mantissa as random bit seq
    mantissa_rand = np.bitwise_and(mantissa_rand, int("0x00001FFF", 16))
    mantissa += mantissa_rand
    # check if we need to shift the mantissa 1 bit right
    # to accommodate the overflow of mantissa
    mantissa_shift_bit = np.right_shift(mantissa, 23)
    mantissa = np.right_shift(mantissa, mantissa_shift_bit)
    exponent_inc_bit = mantissa_shift_bit
    exponent += np.left_shift(exponent_inc_bit, 23)
    # if exponent is larger than the max range for fp32
    # we saturate at the largest value for fp32. 
    # This is very unlikely to happen
    overflow_bit = np.right_shift(exponent, 31)
    exponent[overflow_bit != 0] = int("0x7E800000", 16)
    mantissa[overflow_bit != 0] = int("0x007eeeee", 16)
    value_int[:] = 0
    value_int |= sign
    value_int |= exponent
    value_int |= mantissa

    # print("ckpt 2 ", np.binary_repr(mantissa_rand[0, 0], width=32))

    # print("ckpt 3 ", np.binary_repr(value_int[0, 0], width=32))

    # add the random addition, and then round towards 0 to achieve the stochastic rounding
    assert value_int.flags["C_CONTIGUOUS"]
    value_stoc = np.bitwise_and(value_int, int("0xFFFFE000", 16))
    value_ptr = value_stoc.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    # rand_add = rand_add.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    value = np.ctypeslib.as_array(value_ptr, shape=value_shape)
    value = value.astype(np.float16)
    # deal with the saturation on fp16
    value[value == np.inf] = np.finfo(dtype=np.float16).max
    value[value == -np.inf] = np.finfo(dtype=np.float16).min
    output = torch.HalfTensor(value)
    if tensor.is_cuda:
        output = output.cuda()
    return output


def void_cast_func(tensor):
    return tensor


def get_recur_attr(obj, attr_str_list):
    if len(attr_str_list) == 0:
        return obj
    else:
        sub_obj = getattr(obj, attr_str_list[0])
        return get_recur_attr(sub_obj, attr_str_list[1:])


def void_func():
    pass


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def test_single_to_half_stoc():
    # np.random.seed(1)
    t_np = np.random.randn(*[10, 10]).astype(np.float32)
    t_np_tmp = t_np.copy()
    t_np_ptr = t_np_tmp.ctypes.data_as(ctypes.POINTER(ctypes.c_int32))
    t_np_cp = np.ctypeslib.as_array(t_np_ptr, shape=t_np.shape)
    t = torch.FloatTensor(t_np)
    print("input ", t)
    output = single_to_half_stoc(t)
    print("output ", t)
    assert output.dtype == torch.float16
    output = output.type(torch.float)
    output_tmp = output.cpu().numpy()
    output = output_tmp.ctypes.data_as(ctypes.POINTER(ctypes.c_int32))
    output = np.ctypeslib.as_array(output, shape=t_np.shape)
    input_str = np.binary_repr(t_np_cp[0, 0], width=32)
    output_str = np.binary_repr(output[0, 0], width=32)
    print(input_str)
    print(output_str)




def test_single_to_half_det():
    pass


if __name__ == "__main__":
    test_single_to_half_stoc()