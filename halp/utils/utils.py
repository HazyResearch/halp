import torch

def single_to_half_det(tensor):
    return tensor.half()

def single_to_half_stoc(tensor):
    pass

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