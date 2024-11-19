# pylint: disable=C,W
import numpy as np
import types

kwargs = {}
kwargs['cat'] = '1'
kwargs['dog'] = '2.5'
kwargs['bird'] = 'False' 


def func(keys, types, **kwargs):
	rval = {}

	for i in range(len(kwargs)):
		key = keys[i]
		type_ = types[i]

		if key in kwargs:
			if type_ == bool:
				rval[key] = kwargs[key] == 'True'
			else:
				rval[key] = type_(kwargs[key])

	return rval

keys = ('cat', 'dog', 'bird')
types = (int, float, bool)

print(func(keys, types, **kwargs))
