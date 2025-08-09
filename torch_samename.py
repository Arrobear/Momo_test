torch_samename_data = {
    "torch.dequantize_1":r'''dequantize(tensor) -> Tensor

Returns an fp32 Tensor by dequantizing a quantized Tensor

Args:
    tensor (Tensor): A quantized Tensor''',
    "torch.dequantize_2":r'''dequantize(tensors) -> sequence of Tensors
   :noindex:

Given a list of quantized Tensors, dequantize them and return a list of fp32 Tensors

Args:
     tensors (sequence of Tensors): A list of quantized Tensors''',
    "torch.where_1":r'''where(condition, input, other, *, out=None) -> Tensor

Return a tensor of elements selected from either :attr:`input` or :attr:`other`, depending on :attr:`condition`.

The operation is defined as:

.. math::
    \text{out}_i = \begin{cases}
        \text{input}_i & \text{if } \text{condition}_i \\
        \text{other}_i & \text{otherwise} \\
    \end{cases}

.. note::
    The tensors :attr:`condition`, :attr:`input`, :attr:`other` must be :ref:`broadcastable <broadcasting-semantics>`.

Arguments:
    condition (BoolTensor): When True (nonzero), yield input, otherwise yield other
    input (Tensor or Scalar): value (if :attr:`input` is a scalar) or values selected at indices
                          where :attr:`condition` is ``True``
    other (Tensor or Scalar): value (if :attr:`other` is a scalar) or values selected at indices
                          where :attr:`condition` is ``False``

Keyword args:
    out (Tensor, optional): the output tensor.

Returns:
    Tensor: A tensor of shape equal to the broadcasted shape of :attr:`condition`, :attr:`input`, :attr:`other`

Example::

    >>> x = torch.randn(3, 2)
    >>> y = torch.ones(3, 2)
    >>> x
    tensor([[-0.4620,  0.3139],
            [ 0.3898, -0.7197],
            [ 0.0478, -0.1657]])
    >>> torch.where(x > 0, 1.0, 0.0)
    tensor([[0., 1.],
            [1., 0.],
            [1., 0.]])
    >>> torch.where(x > 0, x, y)
    tensor([[ 1.0000,  0.3139],
            [ 0.3898,  1.0000],
            [ 0.0478,  1.0000]])
    >>> x = torch.randn(2, 2, dtype=torch.double)
    >>> x
    tensor([[ 1.0779,  0.0383],
            [-0.8785, -1.1089]], dtype=torch.float64)
    >>> torch.where(x > 0, x, 0.)
    tensor([[1.0779, 0.0383],
            [0.0000, 0.0000]], dtype=torch.float64)''',
    "torch.where_2":r'''where(condition) -> tuple of LongTensor
   :noindex:

``torch.where(condition)`` is identical to
``torch.nonzero(condition, as_tuple=True)``.

.. note::
    See also :func:`torch.nonzero`.''',
    "torch.normal_4":r'''normal(mean, std, *, generator=None, out=None) -> Tensor

Returns a tensor of random numbers drawn from separate normal distributions
whose mean and standard deviation are given.

The :attr:`mean` is a tensor with the mean of
each output element's normal distribution

The :attr:`std` is a tensor with the standard deviation of
each output element's normal distribution

The shapes of :attr:`mean` and :attr:`std` don't need to match, but the
total number of elements in each tensor need to be the same.

.. note:: When the shapes do not match, the shape of :attr:`mean`
          is used as the shape for the returned output tensor

.. note:: When :attr:`std` is a CUDA tensor, this function synchronizes
          its device with the CPU.

Args:
    mean (Tensor): the tensor of per-element means
    std (Tensor): the tensor of per-element standard deviations

Keyword args:
    generator (:class:`torch.Generator`, optional): a pseudorandom number generator for sampling
    out (Tensor, optional): the output tensor.

Example::

    >>> torch.normal(mean=torch.arange(1., 11.), std=torch.arange(1, 0, -0.1))
    tensor([  1.0425,   3.5672,   2.7969,   4.2925,   4.7229,   6.2134,
              8.0505,   8.1408,   9.0563,  10.0566])''',
    "torch.normal_3":r'''normal(mean=0.0, std, *, out=None) -> Tensor
   :noindex:

Similar to the function above, but the means are shared among all drawn
elements.

Args:
    mean (float, optional): the mean for all distributions
    std (Tensor): the tensor of per-element standard deviations

Keyword args:
    out (Tensor, optional): the output tensor.

Example::

    >>> torch.normal(mean=0.5, std=torch.arange(1., 6.))
    tensor([-1.2793, -1.0732, -2.0687,  5.1177, -1.2303])''',
    "torch.normal_1":r'''normal(mean, std=1.0, *, out=None) -> Tensor
   :noindex:

Similar to the function above, but the standard deviations are shared among
all drawn elements.

Args:
    mean (Tensor): the tensor of per-element means
    std (float, optional): the standard deviation for all distributions

Keyword args:
    out (Tensor, optional): the output tensor

Example::

    >>> torch.normal(mean=torch.arange(1., 6.))
    tensor([ 1.1552,  2.6148,  2.6535,  5.8318,  4.2361])''',
    "torch.normal_2":r'''normal(mean, std, size, *, out=None) -> Tensor
   :noindex:

Similar to the function above, but the means and standard deviations are shared
among all drawn elements. The resulting tensor has size given by :attr:`size`.

Args:
    mean (float): the mean for all distributions
    std (float): the standard deviation for all distributions
    size (int...): a sequence of integers defining the shape of the output tensor.

Keyword args:
    out (Tensor, optional): the output tensor.

Example::

    >>> torch.normal(2, 3, size=(1, 4))
    tensor([[-1.3987, -1.9544,  3.6048,  0.7909]])''',
    "torch.max_3":r'''max(input) -> Tensor

Returns the maximum value of all elements in the ``input`` tensor.

Args:
    input (Tensor): the input tensor.

Example::

    >>> a = torch.randn(1, 3)
    >>> a
    tensor([[ 0.6763,  0.7445, -2.2369]])
    >>> torch.max(a)
    tensor(0.7445)''',
    "torch.max_1":r'''max(input, dim, keepdim=False, *, out=None) -> (Tensor, LongTensor)
   :noindex:

Returns a namedtuple ``(values, indices)`` where ``values`` is the maximum
value of each row of the :attr:`input` tensor in the given dimension
:attr:`dim`. And ``indices`` is the index location of each maximum value found
(argmax).

If ``keepdim`` is ``True``, the output tensors are of the same size
as ``input`` except in the dimension ``dim`` where they are of size 1.
Otherwise, ``dim`` is squeezed (see :func:`torch.squeeze`), resulting
in the output tensors having 1 fewer dimension than ``input``.

.. note:: If there are multiple maximal values in a reduced row then
          the indices of the first maximal value are returned.

Args:
    input (Tensor): the input tensor.
    
    dim (int or tuple of ints, optional): the dimension or dimensions to reduce.
        If ``None``, all dimensions are reduced.

    
    keepdim (bool, optional): whether the output tensor has :attr:`dim` retained or not. Default: ``False``.


Keyword args:
    out (tuple, optional): the result tuple of two output tensors (max, max_indices)

Example::

    >>> a = torch.randn(4, 4)
    >>> a
    tensor([[-1.2360, -0.2942, -0.1222,  0.8475],
            [ 1.1949, -1.1127, -2.2379, -0.6702],
            [ 1.5717, -0.9207,  0.1297, -1.8768],
            [-0.6172,  1.0036, -0.6060, -0.2432]])
    >>> torch.max(a, 1)
    torch.return_types.max(values=tensor([0.8475, 1.1949, 1.5717, 1.0036]), indices=tensor([3, 0, 0, 1]))
    >>> a = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    >>> a.max(dim=1, keepdim=True)
    torch.return_types.max(
    values=tensor([[2.], [4.]]),
    indices=tensor([[1], [1]]))
    >>> a.max(dim=1, keepdim=False)
    torch.return_types.max(
    values=tensor([2., 4.]),
    indices=tensor([1, 1]))

.. function:: max(input, other, *, out=None) -> Tensor
   :noindex:

See :func:`torch.maximum`.''',
    "torch.max_2":r'''Return a tensor of elements selected from either :attr:`input` or :attr:`other`, depending on :attr:`condition`.      

The operation is defined as:

.. math::
    \text{out}_i = \begin{cases}
        \text{input}_i & \text{if } \text{condition}_i \\
        \text{other}_i & \text{otherwise} \\
    \end{cases}

.. note::
    The tensors :attr:`condition`, :attr:`input`, :attr:`other` must be :ref:`broadcastable <broadcasting-semantics>`.

Arguments:
    condition (BoolTensor): When True (nonzero), yield input, otherwise yield other
    input (Tensor or Scalar): value (if :attr:`input` is a scalar) or values selected at indices
                          where :attr:`condition` is ``True``
    other (Tensor or Scalar): value (if :attr:`other` is a scalar) or values selected at indices
                          where :attr:`condition` is ``False``

Keyword args:
    out (Tensor, optional): the output tensor.

Returns:
    Tensor: A tensor of shape equal to the broadcasted shape of :attr:`condition`, :attr:`input`, :attr:`other`

Example::

    >>> x = torch.randn(3, 2)
    >>> y = torch.ones(3, 2)
    >>> x
    tensor([[-0.4620,  0.3139],
            [ 0.3898, -0.7197],
            [ 0.0478, -0.1657]])
    >>> torch.where(x > 0, 1.0, 0.0)
    tensor([[0., 1.],
            [1., 0.],
            [1., 0.]])
    >>> torch.where(x > 0, x, y)
    tensor([[ 1.0000,  0.3139],
            [ 0.3898,  1.0000],
            [ 0.0478,  1.0000]])
    >>> x = torch.randn(2, 2, dtype=torch.double)
    >>> x
    tensor([[ 1.0779,  0.0383],
            [-0.8785, -1.1089]], dtype=torch.float64)
    >>> torch.where(x > 0, x, 0.)
    tensor([[1.0779, 0.0383],
            [0.0000, 0.0000]], dtype=torch.float64)

maximum(input, other, *, out=None) -> Tensor

Computes the element-wise maximum of :attr:`input` and :attr:`other`.

.. note::
    If one of the elements being compared is a NaN, then that element is returned.
    :func:`maximum` is not supported for tensors with complex dtypes.

Args:
    input (Tensor): the input tensor.
    other (Tensor): the second input tensor

Keyword args:
    out (Tensor, optional): the output tensor.

Example::

    >>> a = torch.tensor((1, 2, -1))
    >>> b = torch.tensor((3, 0, 4))
    >>> torch.maximum(a, b)
    tensor([3, 2, 4])''',
    "torch.pow_1":r'''pow(input, exponent, *, out=None) -> Tensor

Takes the power of each element in :attr:`input` with :attr:`exponent` and
returns a tensor with the result.

:attr:`exponent` can be either a single ``float`` number or a `Tensor`
with the same number of elements as :attr:`input`.

When :attr:`exponent` is a scalar value, the operation applied is:

.. math::
    \text{out}_i = x_i ^ \text{exponent}

When :attr:`exponent` is a tensor, the operation applied is:

.. math::
    \text{out}_i = x_i ^ {\text{exponent}_i}

When :attr:`exponent` is a tensor, the shapes of :attr:`input`
and :attr:`exponent` must be :ref:`broadcastable <broadcasting-semantics>`.

Args:
    input (Tensor): the input tensor.
    exponent (float or tensor): the exponent value

Keyword args:
    out (Tensor, optional): the output tensor.

Example::

    >>> a = torch.randn(4)
    >>> a
    tensor([ 0.4331,  1.2475,  0.6834, -0.2791])
    >>> torch.pow(a, 2)
    tensor([ 0.1875,  1.5561,  0.4670,  0.0779])
    >>> exp = torch.arange(1., 5.)

    >>> a = torch.arange(1., 5.)
    >>> a
    tensor([ 1.,  2.,  3.,  4.])
    >>> exp
    tensor([ 1.,  2.,  3.,  4.])
    >>> torch.pow(a, exp)
    tensor([   1.,    4.,   27.,  256.])''',
    "torch.pow_2":r'''pow(self, exponent, *, out=None) -> Tensor
   :noindex:

:attr:`self` is a scalar ``float`` value, and :attr:`exponent` is a tensor.
The returned tensor :attr:`out` is of the same shape as :attr:`exponent`

The operation applied is:

.. math::
    \text{out}_i = \text{self} ^ {\text{exponent}_i}

Args:
    self (float): the scalar base value for the power operation
    exponent (Tensor): the exponent tensor

Keyword args:
    out (Tensor, optional): the output tensor.

Example::

    >>> exp = torch.arange(1., 5.)
    >>> base = 2
    >>> torch.pow(base, exp)
    tensor([  2.,   4.,   8.,  16.])''',
    "torch.argmax_1":r'''argmax(input) -> LongTensor

Returns the indices of the maximum value of all elements in the :attr:`input` tensor.

This is the second value returned by :meth:`torch.max`. See its
documentation for the exact semantics of this method.

.. note:: If there are multiple maximal values then the indices of the first maximal value are returned.

Args:
    input (Tensor): the input tensor.

Example::

    >>> a = torch.randn(4, 4)
    >>> a
    tensor([[ 1.3398,  0.2663, -0.2686,  0.2450],
            [-0.7401, -0.8805, -0.3402, -1.1936],
            [ 0.4907, -1.3948, -1.0691, -0.3132],
            [-1.6092,  0.5419, -0.2993,  0.3195]])
    >>> torch.argmax(a)
    tensor(0)''',
    "torch.argmax_2":r'''argmax(input, dim, keepdim=False) -> LongTensor
   :noindex:

Returns the indices of the maximum values of a tensor across a dimension.

This is the second value returned by :meth:`torch.max`. See its
documentation for the exact semantics of this method.

Args:
    input (Tensor): the input tensor.
    dim (int): the dimension to reduce. If ``None``, the argmax of the flattened input is returned.
    keepdim (bool): whether the output tensor has :attr:`dim` retained or not.

Example::

    >>> a = torch.randn(4, 4)
    >>> a
    tensor([[ 1.3398,  0.2663, -0.2686,  0.2450],
            [-0.7401, -0.8805, -0.3402, -1.1936],
            [ 0.4907, -1.3948, -1.0691, -0.3132],
            [-1.6092,  0.5419, -0.2993,  0.3195]])
    >>> torch.argmax(a, dim=1)
    tensor([ 0,  2,  0,  1])''',
    "torch.all_1":r'''all(input: Tensor) -> Tensor

Tests if all elements in :attr:`input` evaluate to `True`.

.. note:: This function matches the behaviour of NumPy in returning
          output of dtype `bool` for all supported dtypes except `uint8`.
          For `uint8` the dtype of output is `uint8` itself.

Example::

    >>> a = torch.rand(1, 2).bool()
    >>> a
    tensor([[False, True]], dtype=torch.bool)
    >>> torch.all(a)
    tensor(False, dtype=torch.bool)
    >>> a = torch.arange(0, 3)
    >>> a
    tensor([0, 1, 2])
    >>> torch.all(a)
    tensor(False)''',
    "torch.all_2":r'''all(input, dim, keepdim=False, *, out=None) -> Tensor
   :noindex:

For each row of :attr:`input` in the given dimension :attr:`dim`,
returns `True` if all elements in the row evaluate to `True` and `False` otherwise.


If :attr:`keepdim` is ``True``, the output tensor is of the same size
as :attr:`input` except in the dimension(s) :attr:`dim` where it is of size 1.
Otherwise, :attr:`dim` is squeezed (see :func:`torch.squeeze`), resulting in the
output tensor having 1 (or ``len(dim)``) fewer dimension(s).


Args:
    input (Tensor): the input tensor.
    dim (int or tuple of ints): the dimension or dimensions to reduce.
    keepdim (bool): whether the output tensor has :attr:`dim` retained or not.

Keyword args:
    out (Tensor, optional): the output tensor.

Example::

    >>> a = torch.rand(4, 2).bool()
    >>> a
    tensor([[True, True],
            [True, False],
            [True, True],
            [True, True]], dtype=torch.bool)
    >>> torch.all(a, dim=1)
    tensor([ True, False,  True,  True], dtype=torch.bool)
    >>> torch.all(a, dim=0)
    tensor([ True, False], dtype=torch.bool)''',
    "torch.any_1":r'''any(input: Tensor, *, out: Optional[Tensor]) -> Tensor

Tests if any element in :attr:`input` evaluates to `True`.

.. note:: This function matches the behaviour of NumPy in returning
          output of dtype `bool` for all supported dtypes except `uint8`.
          For `uint8` the dtype of output is `uint8` itself.

Example::

    >>> a = torch.rand(1, 2).bool()
    >>> a
    tensor([[False, True]], dtype=torch.bool)
    >>> torch.any(a)
    tensor(True, dtype=torch.bool)
    >>> a = torch.arange(0, 3)
    >>> a
    tensor([0, 1, 2])
    >>> torch.any(a)
    tensor(True)''',
    "torch.any_2":r'''any(input, dim, keepdim=False, *, out=None) -> Tensor
   :noindex:

For each row of :attr:`input` in the given dimension :attr:`dim`,
returns `True` if any element in the row evaluate to `True` and `False` otherwise.


If :attr:`keepdim` is ``True``, the output tensor is of the same size
as :attr:`input` except in the dimension(s) :attr:`dim` where it is of size 1.
Otherwise, :attr:`dim` is squeezed (see :func:`torch.squeeze`), resulting in the
output tensor having 1 (or ``len(dim)``) fewer dimension(s).


Args:
    input (Tensor): the input tensor.
    dim (int or tuple of ints): the dimension or dimensions to reduce.
    keepdim (bool): whether the output tensor has :attr:`dim` retained or not.

Keyword args:
    out (Tensor, optional): the output tensor.

Example::

    >>> a = torch.randn(4, 2) < 0
    >>> a
    tensor([[ True,  True],
            [False,  True],
            [ True,  True],
            [False, False]])
    >>> torch.any(a, 1)
    tensor([ True,  True,  True, False])
    >>> torch.any(a, 0)
    tensor([True, True])''',
    "torch.min_1":r'''min(input) -> Tensor

Returns the minimum value of all elements in the :attr:`input` tensor.

Args:
    input (Tensor): the input tensor.

Example::

    >>> a = torch.randn(1, 3)
    >>> a
    tensor([[ 0.6750,  1.0857,  1.7197]])
    >>> torch.min(a)
    tensor(0.6750)
''',
    "torch.min_2":r'''min(input, dim, keepdim=False, *, out=None) -> (Tensor, LongTensor)
   :noindex:

Returns a namedtuple ``(values, indices)`` where ``values`` is the minimum
value of each row of the :attr:`input` tensor in the given dimension
:attr:`dim`. And ``indices`` is the index location of each minimum value found
(argmin).

If :attr:`keepdim` is ``True``, the output tensors are of the same size as
:attr:`input` except in the dimension :attr:`dim` where they are of size 1.
Otherwise, :attr:`dim` is squeezed (see :func:`torch.squeeze`), resulting in
the output tensors having 1 fewer dimension than :attr:`input`.

.. note:: If there are multiple minimal values in a reduced row then
          the indices of the first minimal value are returned.

Args:
    input (Tensor): the input tensor.
    dim (int): the dimension to reduce.
    keepdim (bool): whether the output tensor has :attr:`dim` retained or not.

Keyword args:
    out (tuple, optional): the tuple of two output tensors (min, min_indices)

Example::

    >>> a = torch.randn(4, 4)
    >>> a
    tensor([[-0.6248,  1.1334, -1.1899, -0.2803],
            [-1.4644, -0.2635, -0.3651,  0.6134],
            [ 0.2457,  0.0384,  1.0128,  0.7015],
            [-0.1153,  2.9849,  2.1458,  0.5788]])
    >>> torch.min(a, 1)
    torch.return_types.min(values=tensor([-1.1899, -1.4644,  0.0384, -0.1153]), indices=tensor([2, 0, 1, 0]))

.. function:: min(input, other, *, out=None) -> Tensor
   :noindex:

See :func:`torch.minimum`.''',
    "torch.mean_1":r'''mean(input, *, dtype=None) -> Tensor

.. note::
    If the `input` tensor is empty, ``torch.mean()`` returns ``nan``.
    This behavior is consistent with NumPy and follows the definition
    that the mean over an empty set is undefined.


Returns the mean value of all elements in the :attr:`input` tensor. Input must be floating point or complex.

Args:
    input (Tensor):
      the input tensor, either of floating point or complex dtype

Keyword args:
    dtype (:class:`torch.dtype`, optional): the desired data type of returned tensor.
        If specified, the input tensor is casted to :attr:`dtype` before the operation
        is performed. This is useful for preventing data type overflows. Default: None.

Example::

    >>> a = torch.randn(1, 3)
    >>> a
    tensor([[ 0.2294, -0.5481,  1.3288]])
    >>> torch.mean(a)
    tensor(0.3367)''',
    "torch.mean_2":r'''mean(input, dim, keepdim=False, *, dtype=None, out=None) -> Tensor
   :noindex:

Returns the mean value of each row of the :attr:`input` tensor in the given
dimension :attr:`dim`. If :attr:`dim` is a list of dimensions,
reduce over all of them.


If :attr:`keepdim` is ``True``, the output tensor is of the same size
as :attr:`input` except in the dimension(s) :attr:`dim` where it is of size 1.
Otherwise, :attr:`dim` is squeezed (see :func:`torch.squeeze`), resulting in the
output tensor having 1 (or ``len(dim)``) fewer dimension(s).


Args:
    input (Tensor): the input tensor.
    dim (int or tuple of ints): the dimension or dimensions to reduce.
    keepdim (bool): whether the output tensor has :attr:`dim` retained or not.

Keyword args:
    dtype (:class:`torch.dtype`, optional): the desired data type of returned tensor.
        If specified, the input tensor is casted to :attr:`dtype` before the operation
        is performed. This is useful for preventing data type overflows. Default: None.
    out (Tensor, optional): the output tensor.

.. seealso::

    :func:`torch.nanmean` computes the mean value of `non-NaN` elements.

Example::

    >>> a = torch.randn(4, 4)
    >>> a
    tensor([[-0.3841,  0.6320,  0.4254, -0.7384],
            [-0.9644,  1.0131, -0.6549, -1.4279],
            [-0.2951, -1.3350, -0.7694,  0.5600],
            [ 1.0842, -0.9580,  0.3623,  0.2343]])
    >>> torch.mean(a, 1)
    tensor([-0.0163, -0.5085, -0.4599,  0.1807])
    >>> torch.mean(a, 1, True)
    tensor([[-0.0163],
            [-0.5085],
            [-0.4599],
            [ 0.1807]])''',
    "torch.median_1":r'''median(input) -> Tensor

Returns the median of the values in :attr:`input`.

.. note::
    The median is not unique for :attr:`input` tensors with an even number
    of elements. In this case the lower of the two medians is returned. To
    compute the mean of both medians, use :func:`torch.quantile` with ``q=0.5`` instead.

.. warning::
    This function produces deterministic (sub)gradients unlike ``median(dim=0)``

Args:
    input (Tensor): the input tensor.

Example::

    >>> a = torch.randn(1, 3)
    >>> a
    tensor([[ 1.5219, -1.5212,  0.2202]])
    >>> torch.median(a)
    tensor(0.2202)''',
    "torch.median_2":r'''median(input, dim=-1, keepdim=False, *, out=None) -> (Tensor, LongTensor)
   :noindex:

Returns a namedtuple ``(values, indices)`` where ``values`` contains the median of each row of :attr:`input`
in the dimension :attr:`dim`, and ``indices`` contains the index of the median values found in the dimension :attr:`dim`.

By default, :attr:`dim` is the last dimension of the :attr:`input` tensor.

If :attr:`keepdim` is ``True``, the output tensors are of the same size
as :attr:`input` except in the dimension :attr:`dim` where they are of size 1.
Otherwise, :attr:`dim` is squeezed (see :func:`torch.squeeze`), resulting in
the outputs tensor having 1 fewer dimension than :attr:`input`.

.. note::
    The median is not unique for :attr:`input` tensors with an even number
    of elements in the dimension :attr:`dim`. In this case the lower of the
    two medians is returned. To compute the mean of both medians in
    :attr:`input`, use :func:`torch.quantile` with ``q=0.5`` instead.

.. warning::
    ``indices`` does not necessarily contain the first occurrence of each
    median value found, unless it is unique.
    The exact implementation details are device-specific.
    Do not expect the same result when run on CPU and GPU in general.
    For the same reason do not expect the gradients to be deterministic.

Args:
    input (Tensor): the input tensor.
    dim (int): the dimension to reduce.
    keepdim (bool): whether the output tensor has :attr:`dim` retained or not.

Keyword args:
    out ((Tensor, Tensor), optional): The first tensor will be populated with the median values and the second
                                      tensor, which must have dtype long, with their indices in the dimension
                                      :attr:`dim` of :attr:`input`.

Example::

    >>> a = torch.randn(4, 5)
    >>> a
    tensor([[ 0.2505, -0.3982, -0.9948,  0.3518, -1.3131],
            [ 0.3180, -0.6993,  1.0436,  0.0438,  0.2270],
            [-0.2751,  0.7303,  0.2192,  0.3321,  0.2488],
            [ 1.0778, -1.9510,  0.7048,  0.4742, -0.7125]])
    >>> torch.median(a, 1)
    torch.return_types.median(values=tensor([-0.3982,  0.2270,  0.2488,  0.4742]), indices=tensor([1, 4, 4, 3]))''',
    "torch.nanmedian_1":r'''nanmedian(input) -> Tensor

Returns the median of the values in :attr:`input`, ignoring ``NaN`` values.

This function is identical to :func:`torch.median` when there are no ``NaN`` values in :attr:`input`.
When :attr:`input` has one or more ``NaN`` values, :func:`torch.median` will always return ``NaN``,
while this function will return the median of the non-``NaN`` elements in :attr:`input`.
If all the elements in :attr:`input` are ``NaN`` it will also return ``NaN``.

Args:
    input (Tensor): the input tensor.

Example::

    >>> a = torch.tensor([1, float('nan'), 3, 2])
    >>> a.median()
    tensor(nan)
    >>> a.nanmedian()
    tensor(2.)''',
    "torch.nanmedian_2":r'''nanmedian(input, dim=-1, keepdim=False, *, out=None) -> (Tensor, LongTensor)
   :noindex:

Returns a namedtuple ``(values, indices)`` where ``values`` contains the median of each row of :attr:`input`
in the dimension :attr:`dim`, ignoring ``NaN`` values, and ``indices`` contains the index of the median values
found in the dimension :attr:`dim`.

This function is identical to :func:`torch.median` when there are no ``NaN`` values in a reduced row. When a reduced row has
one or more ``NaN`` values, :func:`torch.median` will always reduce it to ``NaN``, while this function will reduce it to the
median of the non-``NaN`` elements. If all the elements in a reduced row are ``NaN`` then it will be reduced to ``NaN``, too.

Args:
    input (Tensor): the input tensor.
    dim (int): the dimension to reduce.
    keepdim (bool): whether the output tensor has :attr:`dim` retained or not.

Keyword args:
    out ((Tensor, Tensor), optional): The first tensor will be populated with the median values and the second
                                      tensor, which must have dtype long, with their indices in the dimension
                                      :attr:`dim` of :attr:`input`.

Example::

    >>> a = torch.tensor([[2, 3, 1], [float('nan'), 1, float('nan')]])
    >>> a
    tensor([[2., 3., 1.],
            [nan, 1., nan]])
    >>> a.median(0)
    torch.return_types.median(values=tensor([nan, 1., nan]), indices=tensor([1, 1, 1]))
    >>> a.nanmedian(0)
    torch.return_types.nanmedian(values=tensor([2., 1., 1.]), indices=tensor([0, 1, 0]))''',
    "torch.nansum_1":r'''nansum(input, *, dtype=None) -> Tensor

Returns the sum of all elements, treating Not a Numbers (NaNs) as zero.

Args:
    input (Tensor): the input tensor.

Keyword args:
    dtype (:class:`torch.dtype`, optional): the desired data type of returned tensor.
        If specified, the input tensor is casted to :attr:`dtype` before the operation
        is performed. This is useful for preventing data type overflows. Default: None.

Example::

    >>> a = torch.tensor([1., 2., float('nan'), 4.])
    >>> torch.nansum(a)
    tensor(7.)''',
    "torch.nansum_2":r'''nansum(input, dim, keepdim=False, *, dtype=None) -> Tensor
   :noindex:

Returns the sum of each row of the :attr:`input` tensor in the given
dimension :attr:`dim`, treating Not a Numbers (NaNs) as zero.
If :attr:`dim` is a list of dimensions, reduce over all of them.


If :attr:`keepdim` is ``True``, the output tensor is of the same size
as :attr:`input` except in the dimension(s) :attr:`dim` where it is of size 1.
Otherwise, :attr:`dim` is squeezed (see :func:`torch.squeeze`), resulting in the
output tensor having 1 (or ``len(dim)``) fewer dimension(s).


Args:
    input (Tensor): the input tensor.
    
    dim (int or tuple of ints, optional): the dimension or dimensions to reduce.
        If ``None``, all dimensions are reduced.

    keepdim (bool): whether the output tensor has :attr:`dim` retained or not.

Keyword args:
    dtype (:class:`torch.dtype`, optional): the desired data type of returned tensor.
        If specified, the input tensor is casted to :attr:`dtype` before the operation
        is performed. This is useful for preventing data type overflows. Default: None.

Example::

    >>> torch.nansum(torch.tensor([1., float("nan")]))
    tensor(1.)
    >>> a = torch.tensor([[1, 2], [3., float("nan")]])
    >>> torch.nansum(a)
    tensor(6.)
    >>> torch.nansum(a, dim=0)
    tensor([4., 2.])
    >>> torch.nansum(a, dim=1)
    tensor([3., 3.])''',
    "torch.prod_1":r'''prod(input: Tensor, *, dtype: Optional[_dtype]) -> Tensor

Returns the product of all elements in the :attr:`input` tensor.

Args:
    input (Tensor): the input tensor.

Keyword args:
    dtype (:class:`torch.dtype`, optional): the desired data type of returned tensor.
        If specified, the input tensor is casted to :attr:`dtype` before the operation
        is performed. This is useful for preventing data type overflows. Default: None.

Example::

    >>> a = torch.randn(1, 3)
    >>> a
    tensor([[-0.8020,  0.5428, -1.5854]])
    >>> torch.prod(a)
    tensor(0.6902)
''',
    "torch.prod_2":r'''prod(input, dim, keepdim=False, *, dtype=None) -> Tensor
   :noindex:

Returns the product of each row of the :attr:`input` tensor in the given
dimension :attr:`dim`.

If :attr:`keepdim` is ``True``, the output tensor is of the same size
as :attr:`input` except in the dimension :attr:`dim` where it is of size 1.
Otherwise, :attr:`dim` is squeezed (see :func:`torch.squeeze`), resulting in
the output tensor having 1 fewer dimension than :attr:`input`.

Args:
    input (Tensor): the input tensor.
    dim (int): the dimension to reduce.
    keepdim (bool): whether the output tensor has :attr:`dim` retained or not.

Keyword args:
    dtype (:class:`torch.dtype`, optional): the desired data type of returned tensor.
        If specified, the input tensor is casted to :attr:`dtype` before the operation
        is performed. This is useful for preventing data type overflows. Default: None.

Example::

    >>> a = torch.randn(4, 2)
    >>> a
    tensor([[ 0.5261, -0.3837],
            [ 1.1857, -0.2498],
            [-1.1646,  0.0705],
            [ 1.1131, -1.0629]])
    >>> torch.prod(a, 1)
    tensor([-0.2018, -0.2962, -0.0821, -1.1831])''',
    "torch.sum_1":r'''sum(input, *, dtype=None) -> Tensor

Returns the sum of all elements in the :attr:`input` tensor.

Args:
    input (Tensor): the input tensor.

Keyword args:
    dtype (:class:`torch.dtype`, optional): the desired data type of returned tensor.
        If specified, the input tensor is casted to :attr:`dtype` before the operation
        is performed. This is useful for preventing data type overflows. Default: None.

.. note:: Use the `dtype` argument if you need the result in a specific tensor type.
          Otherwise, the result type may be automatically promoted (e.g., from `torch.int32` to `torch.int64`).

Example::

    >>> a = torch.randn(1, 3)
    >>> a
    tensor([[ 0.1133, -0.9567,  0.2958]])
    >>> torch.sum(a)
    tensor(-0.5475)''',
    "torch.sum_2":r'''sum(input, dim, keepdim=False, *, dtype=None) -> Tensor
   :noindex:

Returns the sum of each row of the :attr:`input` tensor in the given
dimension :attr:`dim`. If :attr:`dim` is a list of dimensions,
reduce over all of them.


If :attr:`keepdim` is ``True``, the output tensor is of the same size
as :attr:`input` except in the dimension(s) :attr:`dim` where it is of size 1.
Otherwise, :attr:`dim` is squeezed (see :func:`torch.squeeze`), resulting in the
output tensor having 1 (or ``len(dim)``) fewer dimension(s).


Args:
    input (Tensor): the input tensor.
    
    dim (int or tuple of ints, optional): the dimension or dimensions to reduce.
        If ``None``, all dimensions are reduced.

    keepdim (bool): whether the output tensor has :attr:`dim` retained or not.

Keyword args:
    dtype (:class:`torch.dtype`, optional): the desired data type of returned tensor.
        If specified, the input tensor is casted to :attr:`dtype` before the operation
        is performed. This is useful for preventing data type overflows. Default: None.

Example::

    >>> a = torch.randn(4, 4)
    >>> a
    tensor([[ 0.0569, -0.2475,  0.0737, -0.3429],
            [-0.2993,  0.9138,  0.9337, -1.6864],
            [ 0.1132,  0.7892, -0.1003,  0.5688],
            [ 0.3637, -0.9906, -0.4752, -1.5197]])
    >>> torch.sum(a, 1)
    tensor([-0.4598, -0.1381,  1.3708, -2.6217])
    >>> b = torch.arange(4 * 5 * 6).view(4, 5, 6)
    >>> torch.sum(b, (2, 1))
    tensor([  435.,  1335.,  2235.,  3135.])''',
}