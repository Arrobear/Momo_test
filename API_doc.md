# Torch

## torch.nn.functional.conv1d

conv1d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1) -> Tensor

Applies a 1D convolution over an input signal composed of several input
planes.

This operator supports :ref:`TensorFloat32<tf32_on_ampere>`.

See :class:`~torch.nn.Conv1d` for details and output shape.

Note:
    In some circumstances when given tensors on a CUDA device and using CuDNN, this operator may select a nondeterministic algorithm to increase performance. If this is undesirable, you can try to make the operation deterministic (potentially at a performance cost) by setting ``torch.backends.cudnn.deterministic = True``. See :doc:`/notes/randomness` for more information.

Note:
    This operator supports complex data types i.e. ``complex32, complex64, complex128``.

Args:
    input: input tensor of shape :math:`(\text{minibatch} , \text{in\_channels} , iW)`
    weight: filters of shape :math:`(\text{out\_channels} , \frac{\text{in\_channels}}{\text{groups}} , kW)`
    bias: optional bias of shape :math:`(\text{out\_channels})`. Default: ``None``
    stride: the stride of the convolving kernel. Can be a single number or
      a one-element tuple `(sW,)`. Default: 1
    padding: implicit paddings on both sides of the input. Can be a string {'valid', 'same'},
      single number or a one-element tuple `(padW,)`. Default: 0
      ``padding='valid'`` is the same as no padding. ``padding='same'`` pads
      the input so the output has the same shape as the input. However, this mode
      doesn't support any stride values other than 1.

  .. warning::
      For ``padding='same'``, if the ``weight`` is even-length and
      ``dilation`` is odd in any dimension, a full :func:`pad` operation
      may be needed internally. Lowering performance.
dilation: the spacing between kernel elements. Can be a single number or
  a one-element tuple `(dW,)`. Default: 1
groups: split input into groups, :math:`\text{in\_channels}` should be divisible by
  the number of groups. Default: 1

Examples::

>>> inputs = torch.randn(33, 16, 30)
>>> filters = torch.randn(20, 16, 5)
>>> F.conv1d(inputs, filters)



['input', 'weight', 'bias', 'stride', 'padding', 'dilation', 'groups']

_____________________________________________________________________    __________________________________________________________________________________


## torch.nn.functional.conv_transpose1d

conv_transpose1d(input, weight, bias=None, stride=1, padding=0, output_padding=0, groups=1, dilation=1) -> Tensor

Applies a 1D transposed convolution operator over an input signal
composed of several input planes, sometimes also called "deconvolution".

This operator supports :ref:`TensorFloat32<tf32_on_ampere>`.

See :class:`~torch.nn.ConvTranspose1d` for details and output shape.

Note:
    In some circumstances when given tensors on a CUDA device and using CuDNN, this operator may select a nondeterministic algorithm to increase performance. If this is undesirable, you can try to make the operation deterministic (potentially at a performance cost) by setting ``torch.backends.cudnn.deterministic = True``. See :doc:`/notes/randomness` for more information.

Args:
    input: input tensor of shape :math:`(\text{minibatch} , \text{in\_channels} , iW)`
    weight: filters of shape :math:`(\text{in\_channels} , \frac{\text{out\_channels}}{\text{groups}} , kW)`
    bias: optional bias of shape :math:`(\text{out\_channels})`. Default: None
    stride: the stride of the convolving kernel. Can be a single number or a
      tuple ``(sW,)``. Default: 1
    padding: ``dilation * (kernel_size - 1) - padding`` zero-padding will be added to both
      sides of each dimension in the input. Can be a single number or a tuple
      ``(padW,)``. Default: 0
    output_padding: additional size added to one side of each dimension in the
      output shape. Can be a single number or a tuple ``(out_padW)``. Default: 0
    groups: split input into groups, :math:`\text{in\_channels}` should be divisible by the
      number of groups. Default: 1
    dilation: the spacing between kernel elements. Can be a single number or
      a tuple ``(dW,)``. Default: 1

Examples::

>>> inputs = torch.randn(20, 16, 50)
>>> weights = torch.randn(16, 33, 5)
>>> F.conv_transpose1d(inputs, weights)



_____________________________________________________________________    __________________________________________________________________________________
## torch.nn.functional.adaptive_avg_pool1d



adaptive_avg_pool1d(input, output_size) -> Tensor

Applies a 1D adaptive average pooling over an input signal composed of
several input planes.

See :class:`~torch.nn.AdaptiveAvgPool1d` for details and output shape.

Args:
    output_size: the target output size (single integer)







# TF

## tf.image.extract_patches

Extract `patches` from `images`.

  This op collects patches from the input image, as if applying a
  convolution. All extracted patches are stacked in the depth (last) dimension  
  of the output.

  Specifically, the op extracts patches of shape `sizes` which are `strides`    
  apart in the input image. The output is subsampled using the `rates` argument,
  in the same manner as "atrous" or "dilated" convolutions.

  The result is a 4D tensor which is indexed by batch, row, and column.
  `output[i, x, y]` contains a flattened patch of size `sizes[1], sizes[2]`     
  which is taken from the input starting at
  `images[i, x*strides[1], y*strides[2]]`.

  Each output patch can be reshaped to `sizes[1], sizes[2], depth`, where       
  `depth` is `images.shape[3]`.

  The output elements are taken from the input at intervals given by the `rate`
  argument, as in dilated convolutions.

  The `padding` argument has no effect on the size of each patch, it determines
  how many patches are extracted. If `VALID`, only patches which are fully
  contained in the input image are included. If `SAME`, all patches whose
  starting point is inside the input are included, and areas outside the input
  default to zero.

  Example:

  ```
    n = 10
    # images is a 1 x 10 x 10 x 1 array that contains the numbers 1 through 100
    images = [[[[x * n + y + 1] for y in range(n)] for x in range(n)]]

    # We generate two outputs as follows:
    # 1. 3x3 patches with stride length 5
    # 2. Same as above, but the rate is increased to 2
    tf.image.extract_patches(images=images,
                             sizes=[1, 3, 3, 1],
                             strides=[1, 5, 5, 1],
                             rates=[1, 1, 1, 1],
                             padding='VALID')

    # Yields:
    [[[[ 1  2  3 11 12 13 21 22 23]
       [ 6  7  8 16 17 18 26 27 28]]
      [[51 52 53 61 62 63 71 72 73]
       [56 57 58 66 67 68 76 77 78]]]]
  ```

  If we mark the pixels in the input image which are taken for the output with
  `*`, we see the pattern:

  ```
     *  *  *  4  5  *  *  *  9 10
     *  *  * 14 15  *  *  * 19 20
     *  *  * 24 25  *  *  * 29 30
    31 32 33 34 35 36 37 38 39 40
    41 42 43 44 45 46 47 48 49 50
     *  *  * 54 55  *  *  * 59 60
     *  *  * 64 65  *  *  * 69 70
     *  *  * 74 75  *  *  * 79 80
    81 82 83 84 85 86 87 88 89 90
    91 92 93 94 95 96 97 98 99 100
  ```

  ```
    tf.image.extract_patches(images=images,
                             sizes=[1, 3, 3, 1],
                             strides=[1, 5, 5, 1],
                             rates=[1, 2, 2, 1],
                             padding='VALID')

    # Yields:
    [[[[  1   3   5  21  23  25  41  43  45]
       [  6   8  10  26  28  30  46  48  50]]

      [[ 51  53  55  71  73  75  91  93  95]
       [ 56  58  60  76  78  80  96  98 100]]]]
  ```

  We can again draw the effect, this time using the symbols `*`, `x`, `+` and
  `o` to distinguish the patches:

  ```
     *  2  *  4  *  x  7  x  9  x
    11 12 13 14 15 16 17 18 19 20
     * 22  * 24  *  x 27  x 29  x
    31 32 33 34 35 36 37 38 39 40
     * 42  * 44  *  x 47  x 49  x
     + 52  + 54  +  o 57  o 59  o
    61 62 63 64 65 66 67 68 69 70
     + 72  + 74  +  o 77  o 79  o
    81 82 83 84 85 86 87 88 89 90
     + 92  + 94  +  o 97  o 99  o
  ```

  Args:
    images: A 4-D Tensor with shape `[batch, in_rows, in_cols, depth]`.
    sizes: The size of the extracted patches. Must be
      `[1, size_rows, size_cols, 1]`.
    strides: A 1-D Tensor of length 4. How far the centers of two consecutive
      patches are in the images. Must be: `[1, stride_rows, stride_cols, 1]`.
    rates: A 1-D Tensor of length 4. Must be: `[1, rate_rows, rate_cols, 1]`.
      This is the input stride, specifying how far two consecutive patch samples
      are in the input. Equivalent to extracting patches with `patch_sizes_eff =
      patch_sizes + (patch_sizes - 1) * (rates - 1)`, followed by subsampling
      them spatially by a factor of `rates`. This is equivalent to `rate` in
      dilated (a.k.a. Atrous) convolutions.
    padding: The type of padding algorithm to use.
    name: A name for the operation (optional).

  Returns:
    A 4-D Tensor of the same type as the input.




## tf.keras.layers.SeparableConv2D

2D separable convolution layer.

    This layer performs a depthwise convolution that acts separately on
    channels, followed by a pointwise convolution that mixes channels.
    If `use_bias` is True and a bias initializer is provided,
    it adds a bias vector to the output. It then optionally applies an
    activation function to produce the final output.
    
    Args:
        filters: int, the dimensionality of the output space (i.e. the number
            of filters in the pointwise convolution).
        kernel_size: int or tuple/list of 2 integers, specifying the size of the
            depthwise convolution window.
        strides: int or tuple/list of 2 integers, specifying the stride length
            of the depthwise convolution. If only one int is specified, the same
            stride size will be used for all dimensions. `strides > 1` is
            incompatible with `dilation_rate > 1`.
        padding: string, either `"valid"` or `"same"` (case-insensitive).
            `"valid"` means no padding. `"same"` results in padding evenly to
            the left/right or up/down of the input. When `padding="same"` and
            `strides=1`, the output has the same size as the input.
        data_format: string, either `"channels_last"` or `"channels_first"`.
            The ordering of the dimensions in the inputs. `"channels_last"`
            corresponds to inputs with shape `(batch, height, width, channels)`
            while `"channels_first"` corresponds to inputs with shape
            `(batch, channels, height, width)`. It defaults to the
            `image_data_format` value found in your Keras config file
            at `~/.keras/keras.json`.
            If you never set it, then it will be `"channels_last"`.
        dilation_rate: int or tuple/list of 2 integers, specifying the dilation
            rate to use for dilated convolution. If only one int is specified,
            the same dilation rate will be used for all dimensions.
        depth_multiplier: The number of depthwise convolution output channels
            for each input channel. The total number of depthwise convolution
            output channels will be equal to `input_channel * depth_multiplier`.
        activation: Activation function. If `None`, no activation is applied.
        use_bias: bool, if `True`, bias will be added to the output.
        depthwise_initializer: An initializer for the depthwise convolution
            kernel. If None, then the default initializer (`"glorot_uniform"`)
            will be used.
        pointwise_initializer: An initializer for the pointwise convolution
            kernel. If None, then the default initializer (`"glorot_uniform"`)
            will be used.
        bias_initializer: An initializer for the bias vector. If None, the
            default initializer ('"zeros"') will be used.
        depthwise_regularizer: Optional regularizer for the depthwise
            convolution kernel.
        pointwise_regularizer: Optional regularizer for the pointwise
            convolution kernel.
        bias_regularizer: Optional regularizer for the bias vector.
        activity_regularizer: Optional regularizer function for the output.
        depthwise_constraint: Optional projection function to be applied to the
            depthwise kernel after being updated by an `Optimizer` (e.g. used
            for norm constraints or value constraints for layer weights). The
            function must take as input the unprojected variable and must return
            the projected variable (which must have the same shape).
        pointwise_constraint: Optional projection function to be applied to the
            pointwise kernel after being updated by an `Optimizer`.
        bias_constraint: Optional projection function to be applied to the
            bias after being updated by an `Optimizer`.
    
    Input shape:
    
    - If `data_format="channels_last"`:
        A 4D tensor with shape: `(batch_size, height, width, channels)`
    - If `data_format="channels_first"`:
        A 4D tensor with shape: `(batch_size, channels, height, width)`
    
    Output shape:
    
    - If `data_format="channels_last"`:
        A 4D tensor with shape: `(batch_size, new_height, new_width, filters)`
    - If `data_format="channels_first"`:
        A 4D tensor with shape: `(batch_size, filters, new_height, new_width)`
    
    Returns:
        A 4D tensor representing
        `activation(separable_conv2d(inputs, kernel) + bias)`.
    
    Example:
    
    >>> x = np.random.rand(4, 10, 10, 12)
    >>> y = keras.layers.SeparableConv2D(3, 4, 3, 2, activation='relu')(x)
    >>> print(y.shape)
    (4, 4, 4, 4)



_____________________________________________________________________    __________________________________________________________________________________
## tf.broadcast_to



Broadcast an array for a compatible shape.

  Broadcasting is the process of making arrays to have compatible shapes
  for arithmetic operations. Two shapes are compatible if for each
  dimension pair they are either equal or one of them is one.

  For example:

  >>> x = tf.constant([[1, 2, 3]])   # Shape (1, 3,)
  >>> y = tf.broadcast_to(x, [2, 3])
  >>> print(y)
  >>> tf.Tensor(
  >>> [[1 2 3]
  >>>  [1 2 3]], shape=(2, 3), dtype=int32)

  In the above example, the input Tensor with the shape of `[1, 3]`
  is broadcasted to output Tensor with shape of `[2, 3]`.

  When broadcasting, if a tensor has fewer axes than necessary its shape is
  padded on the left with ones. So this gives the same result as the previous
  example:

  >>> x = tf.constant([1, 2, 3])   # Shape (3,)
  >>> y = tf.broadcast_to(x, [2, 3])


  When doing broadcasted operations such as multiplying a tensor
  by a scalar, broadcasting (usually) confers some time or space
  benefit, as the broadcasted tensor is never materialized.

  However, `broadcast_to` does not carry with it any such benefits.
  The newly-created tensor takes the full memory of the broadcasted
  shape. (In a graph context, `broadcast_to` might be fused to
  subsequent operation and then be optimized away, however.)

  Args:
    input: A `Tensor`. A Tensor to broadcast.
    shape: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      An 1-D `int` Tensor. The shape of the desired output.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `input`.



