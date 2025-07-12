#  

# 1.0.0:最初版本

## prompt：

You are an expert in deep learning.

Background:
1.The structure of the API document is: API name, API introduction, multiple notes (precautions), parameters and parameter introduction, examples.
2.There are usually various dependencies and restrictions between API parameters, such as some parameters being optional, some parameters being mandatory, and some parameters having to appear in pairs. Attention should be paid to the above limitations when executing tasks.

Your task as follows:
1.Generate reasonable parameter combination conditions based on the API documentation and parameter name array. Output strictly limited to the API's explicitly defined parameters.
2.Extract parameter value types from the API documentation and add them to the parameter combination array.

Output format: 
  {
  "Parameter type": {
      "Parameter 1": "Type",
      "Parameter 2": "Type",
     ...
    }
  “Required Parameters”: [Parameter 1, Parameter 2, ...],
  “Forced Coexistence”: [[Parameter 1, Parameter 2], [Parameter 3, Parameter 4], ...],
  “Mutual Exclusion”: [[Parameter 1, Parameter 3],[Parameter 2, Parameter 4], ...]
}
Output description:
1.Parameter type: represents the parameter types of the API; 
2.Required Parameters: represents the necessary parameters that must exist; 
3.Forced Coexistence: means that parameter 1 and parameter 2 must appear simultaneously; 
4.Mutual Exclusion: means that parameter 1 and parameter 3 cannot occur simultaneously.

Output Examples: 
{
  "Parameter type": {
    "input": "Tensor (shape: (minibatch, in_channels, iW))",
    "weight": "Tensor (shape: (out_channels, in_channels // groups, kW))",
    "bias": "Optional[Tensor] (shape: (out_channels))",
    "stride": "Union[int, Tuple[int]] (default: 1)",
    "padding": "Union[int, Tuple[int], str] (default: 0; options: 'valid', 'same')",
    "dilation": "Union[int, Tuple[int]] (default: 1)",
    "groups": "int (default: 1)"
  },
    "Required Parameters": ["input", "weight"],
    "Forced Coexistence": [],
    "Mutual Exclusion": []
}

Parameter name array : 
["input", "weight", "bias", "stride", "padding", "dilation", "groups"]

API ducumentation:
api_name: torch.nn.functional.conv1d
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

### 输出结果_tes_1：

```
{
  "Parameter type": {
    "input": "Tensor (shape: (minibatch, in_channels, iW))",
    "weight": "Tensor (shape: (out_channels, in_channels // groups, kW))",
    "bias": "Optional[Tensor] (shape: (out_channels))",
    "stride": "Union[int, Tuple[int]] (default: 1)",
    "padding": "Union[int, Tuple[int], str] (default: 0; options: 'valid', 'same')",
    "dilation": "Union[int, Tuple[int]] (default: 1)",
    "groups": "int (default: 1)"
  },
  "Required Parameters": ["input", "weight"],
  "Forced Coexistence": [],
  "Mutual Exclusion": []
}

```



# 1.1.0:杨老师修改第一版

## prompt：

1.Role:

You are an expert in [Python Library Name], with deep knowledge of its API design, functionality, and practical usage across a wide range of scenarios.

---

2. Background and Context:

(1) API Documentation.

We provide below the official documentation for the API to be analyzed: [torch.nn.functional.conv1d]. 
This documentation specifies the API’s function signature, behavior, supported data types, argument definitions, default values, constraints, and usage examples, enabling precise understanding of its operational semantics.
The specific API documentation content is as below:
[
    api_name: torch.nn.functional.conv1d
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
        ``dilation`` is odd in any dimension, a full :func:`pad` operation
        may be needed internally. Lowering performance.
    dilation: the spacing between kernel elements. Can be a single number or
    a one-element tuple `(dW,)`. Default: 1
    groups: split input into groups, :math:`\text{in\_channels}` should be divisible by
    the number of groups. Default: 1

Examples::

​    >>> inputs = torch.randn(33, 16, 30)
​    >>> filters = torch.randn(20, 16, 5)
​    >>> F.conv1d(inputs, filters)

]

(2) Parameter Dependency Types.

In deep learning libraries such as [Python Library Name], many APIs accept a set of configurable parameters, enabling multiple usage patterns based on different parameter combinations. 
These parameters may exhibit various types of interdependencies that govern valid and efficient usage. Specifically, we consider the following three relationship types:

- "Mandatory Parameters": Parameters that must be explicitly specified for the API to function correctly.  
- "Mutually Exclusive Parameter Pairs": Pairs of parameters that cannot be used together in a single call due to logical or operational conflicts.  
- "Associated Parameter Groups": Sets of parameters that must be provided together to ensure valid configuration or meaningful behavior.

---

3. Your Tasks:

Based on the documentation and definitions provided:

1. Determine the Type of Each Parameter.
   For each explicitly defined parameter in [torch.nn.functional.conv1d], determine its type, such as: tensor, int, str, optional tensor, etc.

2. Identify Parameter Dependency Structures as accurately and completely as possible:
   - List all "Mandatory Parameters"  
   - Identify any "Mutually Exclusive Parameter Pairs"  
   - Identify any "Associated Parameter Groups"

---

4. Output Format:

   ```
   {
     "Parameter type": {
       "input": "...",
       "weight": "...",
       "bias": "...",
       "stride": "...",
       "padding": "...",
       "dilation": "...",
       "groups": "..."
     },
     "Mandatory Parameters": ["...", "..."],
     "Mutually Exclusive Parameter Pairs": [["...", "..."], ...],
     "Associated Parameter Groups": [["...", "...", "..."], ["...", "..."], ...]
   }
   ```


### 输出结果_tes_1：

```
{
  "Parameter type": {
    "input": "Tensor of shape (minibatch, in_channels, iW)",
    "weight": "Tensor of shape (out_channels, in_channels / groups, kW)",
    "bias": "Optional Tensor of shape (out_channels)",
    "stride": "int or tuple of one int (sW,)",
    "padding": "int, tuple of one int (padW,), or str ('valid' or 'same')",
    "dilation": "int or tuple of one int (dW,)",
    "groups": "int"
  },
  "Mandatory Parameters": ["input", "weight"],
  "Mutually Exclusive Parameter Pairs": [["padding (str)", "padding (int or tuple)"]],
  "Associated Parameter Groups": [["input", "weight", "bias"], ["weight", "groups"]]
}

```

# 1.1.1添加example：

## prompt：

1. Role:

You are an expert in [Python Library Name], with deep knowledge of its API design, functionality, and practical usage across a wide range of scenarios.

---

2. Background and Context:

(1) API Documentation.

We provide below the official documentation for the API to be analyzed: [torch.nn.functional.conv1d]. 
This documentation specifies the API’s function signature, behavior, supported data types, argument definitions, default values, constraints, and usage examples, enabling precise understanding of its operational semantics.
THe specific API documentation content is as below:
[
    api_name: torch.nn.functional.conv1d
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
        ``dilation`` is odd in any dimension, a full :func:`pad` operation
        may be needed internally. Lowering performance.
    dilation: the spacing between kernel elements. Can be a single number or
    a one-element tuple `(dW,)`. Default: 1
    groups: split input into groups, :math:`\text{in\_channels}` should be divisible by
    the number of groups. Default: 1

Examples::

​    >>> inputs = torch.randn(33, 16, 30)
​    >>> filters = torch.randn(20, 16, 5)
​    >>> F.conv1d(inputs, filters)

]

(2) Parameter Dependency Types.

In deep learning libraries such as [Python Library Name], many APIs accept a set of configurable parameters, enabling multiple usage patterns based on different parameter combinations. 
These parameters may exhibit various types of interdependencies that govern valid and efficient usage. Specifically, we consider the following three relationship types:

- "Mandatory Parameters": Parameters that must be explicitly specified for the API to function correctly.  
- "Mutually Exclusive Parameter Pairs": Pairs of parameters that cannot be used together in a single call due to logical or operational conflicts.  
- "Associated Parameter Groups": Sets of parameters that must be provided together to ensure valid configuration or meaningful behavior.

---

3. Your Tasks:

Based on the documentation and definitions provided:

1. Determine the Type of Each Parameter.
   For each explicitly defined parameter in [torch.nn.functional.conv1d], determine its type, such as: tensor, int, str, optional tensor, etc.

2. Identify Parameter Dependency Structures as accurately and completely as possible:
   - List all "Mandatory Parameters"  
   - Identify any "Mutually Exclusive Parameter Pairs"  
   - Identify any "Associated Parameter Groups"

---

4. Output Format:

```json
{
  "Parameter type": {
    "input": "...",
    "weight": "...",
    "bias": "...",
    "stride": "...",
    "padding": "...",
    "dilation": "...",
    "groups": "..."
  },
  "Mandatory Parameters": ["...", "..."],
  "Mutually Exclusive Parameter Pairs": [["...", "..."], ...],
  "Associated Parameter Groups": [["...", "...", "..."], ["...", "..."], ...]
}

---

5. Examples:
Output Examples: 
{
  "Parameter type": {
    "input": "Tensor (shape: (minibatch, in_channels, iW))",
    "weight": "Tensor (shape: (out_channels, in_channels // groups, kW))",
    "bias": "Optional[Tensor] (shape: (out_channels))",
    "stride": "Union[int, Tuple[int]] (default: 1)",
    "padding": "Union[int, Tuple[int], str] (default: 0; options: 'valid', 'same')",
    "dilation": "Union[int, Tuple[int]] (default: 1)",
    "groups": "int (default: 1)"
  },
    "Mandatory Parameters": ["input", "weight"],
    "Mutually Exclusive Parameter Pairs": [],
    "Associated Parameter Groups"": []
}


```

### 输出结果_tes_1：

```
{
  "Parameter type": {
    "input": "Tensor (shape: (minibatch, in_channels, iW))",
    "weight": "Tensor (shape: (out_channels, in_channels // groups, kW))",
    "bias": "Optional[Tensor] (shape: (out_channels))",
    "stride": "Union[int, Tuple[int]] (default: 1)",
    "padding": "Union[int, Tuple[int], str] (default: 0; options: 'valid', 'same')",
    "dilation": "Union[int, Tuple[int]] (default: 1)",
    "groups": "int (default: 1)"
  },
  "Mandatory Parameters": ["input", "weight"],
  "Mutually Exclusive Parameter Pairs": [],
  "Associated Parameter Groups": [["input", "weight", "bias"]]
}

```

# 1.0.1：更改输出格式不添加example

## prompt：

1. Role:

You are an expert in [Python Library Name], with deep knowledge of its API design, functionality, and practical usage across a wide range of scenarios.

---

2. Background and Context:

(1) API Documentation.

We provide below the official documentation for the API to be analyzed: [torch.nn.functional.conv1d]. 
This documentation specifies the API’s function signature, behavior, supported data types, argument definitions, default values, constraints, and usage examples, enabling precise understanding of its operational semantics.
THe specific API documentation content is as below:
[
    api_name: torch.nn.functional.conv1d
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
        ``dilation`` is odd in any dimension, a full :func:`pad` operation
        may be needed internally. Lowering performance.
    dilation: the spacing between kernel elements. Can be a single number or
    a one-element tuple `(dW,)`. Default: 1
    groups: split input into groups, :math:`\text{in\_channels}` should be divisible by
    the number of groups. Default: 1

Examples::

​    >>> inputs = torch.randn(33, 16, 30)
​    >>> filters = torch.randn(20, 16, 5)
​    >>> F.conv1d(inputs, filters)

]

(2) Parameter Dependency Types.

In deep learning libraries such as [Python Library Name], many APIs accept a set of configurable parameters, enabling multiple usage patterns based on different parameter combinations. 
These parameters may exhibit various types of interdependencies that govern valid and efficient usage. Specifically, we consider the following three relationship types:

- "Mandatory Parameters": Parameters that must be explicitly specified for the API to function correctly.  
- "Mutually Exclusive Parameter Pairs": Pairs of parameters that cannot be used together in a single call due to logical or operational conflicts.  
- "Mandatory coexistence parameters": Sets of parameters that must be provided together to ensure valid configuration or meaningful behavior.

---

3. Your Tasks:

Based on the documentation and definitions provided:

1. Determine the Type of Each Parameter.
   For each explicitly defined parameter in [torch.nn.functional.conv1d], determine its type, such as: tensor, int, str, optional tensor, etc.
2. Identify Parameter Dependency Structures as accurately and completely as possible:
   - List all "Mandatory Parameters"  
   - Identify any "Mutually Exclusive Parameter Pairs"  
   - Identify any "Mandatory coexistence parameters"

------

4.Output Format:

{
  "Parameter type": {
    "input": "...",
    "weight": "...",
    "bias": "...",
    "stride": "...",
    "padding": "...",
    "dilation": "...",
    "groups": "..."
  },
  "Mandatory Parameters": ["...", "..."],
  "Mutually Exclusive Parameter Pairs": [["...", "..."], ...],
  "Mandatory coexistence parameters": [["...", "...", "..."], ["...", "..."], ...]
}



### 输出结果_tes_1：

```
{
  "Parameter type": {
    "input": "Tensor of shape (minibatch, in_channels, iW)",
    "weight": "Tensor of shape (out_channels, in_channels / groups, kW)",
    "bias": "Optional Tensor of shape (out_channels)",
    "stride": "int or one-element tuple (sW,)",
    "padding": "int, one-element tuple (padW,), or str {'valid', 'same'}",
    "dilation": "int or one-element tuple (dW,)",
    "groups": "int"
  },
  "Mandatory Parameters": ["input", "weight"],
  "Mutually Exclusive Parameter Pairs": [],
  "Mandatory coexistence parameters": [
    ["input", "weight"],
    ["weight", "groups"]
  ]
}
```



# 1.0.2：更改输出格式添加example：

## prompt：

1. Role:

You are an expert in [Python Library Name], with deep knowledge of its API design, functionality, and practical usage across a wide range of scenarios.

---

2. Background and Context:

(1) API Documentation.

We provide below the official documentation for the API to be analyzed: [torch.nn.functional.conv1d]. 
This documentation specifies the API’s function signature, behavior, supported data types, argument definitions, default values, constraints, and usage examples, enabling precise understanding of its operational semantics.
THe specific API documentation content is as below:
[
    api_name: torch.nn.functional.conv1d
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
        ``dilation`` is odd in any dimension, a full :func:`pad` operation
        may be needed internally. Lowering performance.
    dilation: the spacing between kernel elements. Can be a single number or
    a one-element tuple `(dW,)`. Default: 1
    groups: split input into groups, :math:`\text{in\_channels}` should be divisible by
    the number of groups. Default: 1

Examples::

​    >>> inputs = torch.randn(33, 16, 30)
​    >>> filters = torch.randn(20, 16, 5)
​    >>> F.conv1d(inputs, filters)

]

(2) Parameter Dependency Types.

In deep learning libraries such as [Python Library Name], many APIs accept a set of configurable parameters, enabling multiple usage patterns based on different parameter combinations. 
These parameters may exhibit various types of interdependencies that govern valid and efficient usage. Specifically, we consider the following three relationship types:

- "Mandatory Parameters": Parameters that must be explicitly specified for the API to function correctly.  
- "Mutually Exclusive Parameter Pairs": Pairs of parameters that cannot be used together in a single call due to logical or operational conflicts.  
- "Mandatory coexistence parameters": Sets of parameters that must be provided together to ensure valid configuration or meaningful behavior.

---

3. Your Tasks:

Based on the documentation and definitions provided:

1. Determine the Type of Each Parameter.
   For each explicitly defined parameter in [torch.nn.functional.conv1d], determine its type, such as: tensor, int, str, optional tensor, etc.
2. Identify Parameter Dependency Structures as accurately and completely as possible:
   - List all "Mandatory Parameters"  
   - Identify any "Mutually Exclusive Parameter Pairs"  
   - Identify any "Mandatory coexistence parameters"

------

4.Output Format:

{
  "Parameter type": {
    "input": "...",
    "weight": "...",
    "bias": "...",
    "stride": "...",
    "padding": "...",
    "dilation": "...",
    "groups": "..."
  },
  "Mandatory Parameters": ["...", "..."],
  "Mutually Exclusive Parameter Pairs": [["...", "..."], ...],
  "Mandatory coexistence parameters": [["...", "...", "..."], ["...", "..."], ...]
}

---

5. Examples:
   Output Examples: 
   {
     "Parameter type": {
    "input": "Tensor (shape: (minibatch, in_channels, iW))",
    "weight": "Tensor (shape: (out_channels, in_channels // groups, kW))",
    "bias": "Optional[Tensor] (shape: (out_channels))",
    "stride": "Union[int, Tuple[int]] (default: 1)",
    "padding": "Union[int, Tuple[int], str] (default: 0; options: 'valid', 'same')",
    "dilation": "Union[int, Tuple[int]] (default: 1)",
    "groups": "int (default: 1)"
     },
    "Mandatory Parameters": ["input", "weight"],
    "Mutually Exclusive Parameter Pairs": [],
    "Mandatory coexistence parameters": []
   }

### 输出结果_tes_1：torch.nn.functional.conv1d

```
{
  "Parameter type": {
    "input": "Tensor (shape: (minibatch, in_channels, iW))",
    "weight": "Tensor (shape: (out_channels, in_channels // groups, kW))",
    "bias": "Optional[Tensor] (shape: (out_channels))",
    "stride": "Union[int, Tuple[int]] (default: 1)",
    "padding": "Union[int, Tuple[int], str] (default: 0; options: 'valid', 'same')",
    "dilation": "Union[int, Tuple[int]] (default: 1)",
    "groups": "int (default: 1)"
  },
  "Mandatory Parameters": ["input", "weight"],
  "Mutually Exclusive Parameter Pairs": [],
  "Mandatory coexistence parameters": []
}
```

### 输出结果_tes_2：torch.nn.functional.conv_transpose1d

```
{
  "Parameter type": {
    "input": "Tensor (shape: (minibatch, in_channels, iW))",
    "weight": "Tensor (shape: (in_channels, out_channels // groups, kW))",
    "bias": "Optional[Tensor] (shape: (out_channels))",
    "stride": "Union[int, Tuple[int]] (default: 1)",
    "padding": "Union[int, Tuple[int]] (default: 0)",
    "output_padding": "Union[int, Tuple[int]] (default: 0)",
    "groups": "int (default: 1)",
    "dilation": "Union[int, Tuple[int]] (default: 1)"
  },
  "Mandatory Parameters": ["input", "weight"],
  "Mutually Exclusive Parameter Pairs": [],
  "Mandatory coexistence parameters": []
}

```

### 输出结果_tes_3：torch.nn.functional.adaptive_avg_pool1d

```
{
  "Parameter type": {
    "input": "Tensor (shape: (minibatch, in_channels, input_length))",
    "output_size": "int"
  },
  "Mandatory Parameters": ["input", "output_size"],
  "Mutually Exclusive Parameter Pairs": [],
  "Mandatory coexistence parameters": []
}

```

### 输出结果_tes_4：tf.image.extract_patches

```
{
  "Parameter type": {
    "images": "Tensor (4-D Tensor of shape [batch, in_rows, in_cols, depth])",
    "sizes": "List[int] (length 4; must be [1, size_rows, size_cols, 1])",
    "strides": "List[int] (length 4; must be [1, stride_rows, stride_cols, 1])",
    "rates": "List[int] (length 4; must be [1, rate_rows, rate_cols, 1])",
    "padding": "str (\"VALID\" or \"SAME\")",
    "name": "Optional[str] (operation name; default: None)"
  },
  "Mandatory Parameters": [
    "images",
    "sizes",
    "strides",
    "rates",
    "padding"
  ],
  "Mutually Exclusive Parameter Pairs": [],
  "Mandatory coexistence parameters": []
}

```

### 输出结果_tes_5：tf.keras.layers.SeparableConv2D

```
{
  "Parameter type": {
    "filters": "int",
    "kernel_size": "Union[int, Tuple[int, int]]",
    "strides": "Union[int, Tuple[int, int]] (default: 1)",
    "padding": "str (default: 'valid'; options: 'valid', 'same')",
    "data_format": "Optional[str] (default: 'channels_last'; options: 'channels_last', 'channels_first')",
    "dilation_rate": "Union[int, Tuple[int, int]] (default: 1)",
    "depth_multiplier": "int (default: 1)",
    "activation": "Optional[Callable or str] (default: None)",
    "use_bias": "bool (default: True)",
    "depthwise_initializer": "Optional[Initializer] (default: 'glorot_uniform')",
    "pointwise_initializer": "Optional[Initializer] (default: 'glorot_uniform')",
    "bias_initializer": "Optional[Initializer] (default: 'zeros')",
    "depthwise_regularizer": "Optional[Regularizer] (default: None)",
    "pointwise_regularizer": "Optional[Regularizer] (default: None)",
    "bias_regularizer": "Optional[Regularizer] (default: None)",
    "activity_regularizer": "Optional[Regularizer] (default: None)",
    "depthwise_constraint": "Optional[Constraint function] (default: None)",
    "pointwise_constraint": "Optional[Constraint function] (default: None)",
    "bias_constraint": "Optional[Constraint function] (default: None)"
  },
  "Mandatory Parameters": [
    "filters",
    "kernel_size"
  ],
  "Mutually Exclusive Parameter Pairs": [
    ["strides", "dilation_rate"]  // Only when both are >1
  ],
  "Mandatory coexistence parameters": []
}

```

### 输出结果_tes_6：tf.broadcast_to

```
{
  "Parameter type": {
    "input": "Tensor (any shape, any dtype)",
    "shape": "Tensor (1-D Tensor of int32 or int64)",
    "name": "Optional[str] (default: None)"
  },
  "Mandatory Parameters": ["input", "shape"],
  "Mutually Exclusive Parameter Pairs": [],
  "Mandatory coexistence parameters": []
}

```

# 1.0.3：考虑条件互斥情况

## prompt：

1. Role:

You are an expert in [Python Library Name], with deep knowledge of its API design, functionality, and practical usage across a wide range of scenarios.

---

2. Background and Context:

(1) API Documentation.

We provide below the official documentation for the API to be analyzed: 

[

torch.nn.functional.conv1d

]. 

This documentation specifies the API’s function signature, behavior, supported data types, argument definitions, default values, constraints, and usage examples, enabling precise understanding of its operational semantics.
THe specific API documentation content is as below:

[

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

\>>> inputs = torch.randn(33, 16, 30)

\>>> filters = torch.randn(20, 16, 5)

\>>> F.conv1d(inputs, filters)

]

(2) Parameter Dependency Types.

In deep learning libraries such as [Python Library Name], many APIs accept a set of configurable parameters, enabling multiple usage patterns based on different parameter combinations. 
These parameters may exhibit various types of interdependencies that govern valid and efficient usage. Specifically, we consider the following three relationship types:

- "Mandatory Parameters": Parameters that must be explicitly specified for the API to function correctly.  
- "Mutually Exclusive Parameter Pairs": Pairs of parameters that cannot be used together in a single call due to logical or operational conflicts.  
- "Mandatory coexistence parameters": Sets of parameters that must be provided together to ensure valid configuration or meaningful behavior.
- "Conditional Mutual Exclusion Parameters": When the parameters meet certain conditions, it will prevent the function from running.

---

3. Your Tasks:

Based on the documentation and definitions provided:

1. Determine the Type of Each Parameter.
   For each explicitly defined parameter in [torch.nn.functional.conv1d], determine its type, such as: tensor, int, str, optional tensor, etc.
2. Identify Parameter Dependency Structures as accurately and completely as possible:
   - List all "Mandatory Parameters"  
   - Identify any "Mutually Exclusive Parameter Pairs"  
   - Identify any "Mandatory Coexistence Parameters"
   - Identify any "Conditional Mutual Exclusion Parameters"

------

4.Output Format:

{
  "Parameter type": {
    "input": "...",
    "weight": "...",
    "bias": "...",
    "stride": "...",
    "padding": "...",
    "dilation": "...",
    "groups": "..."
  },
  "Mandatory Parameters": ["...", "..."],
  "Mutually Exclusive Parameter Pairs": [["...", "..."], ...],
  "Mandatory Coexistence Parameters": [["...", "...", "..."], ["...", "..."], ...],

"Conditional Mutual Exclusion Parameters":["para_1", "para_2","(para_1>1)&(para_2>1)"] 

}

---

5. Examples:
   Output Examples: 
   {
     "Parameter type": {
    "input": "Tensor (shape: (minibatch, in_channels, iW))",
    "weight": "Tensor (shape: (out_channels, in_channels // groups, kW))",
    "bias": "Optional[Tensor] (shape: (out_channels))",
    "stride": "Union[int, Tuple[int]] (default: 1)",
    "padding": "Union[int, Tuple[int], str] (default: 0; options: 'valid', 'same')",
    "dilation": "Union[int, Tuple[int]] (default: 1)",
    "groups": "int (default: 1)"
     },
    "Mandatory Parameters": ["input", "weight"],
    "Mutually Exclusive Parameter Pairs": [],
    "Mandatory coexistence parameters": []

   "Conditional mutual exclusion parameters": ["strides", "dilation","(strides>1)&(dilation>1)"] 

   }

### 输出结果_tes_1：torch.nn.functional.conv1d

```
{
  "Parameter type": {
    "input": "Tensor (shape: (minibatch, in_channels, iW))",
    "weight": "Tensor (shape: (out_channels, in_channels // groups, kW))",
    "bias": "Optional[Tensor] (shape: (out_channels))",
    "stride": "Union[int, Tuple[int]] (default: 1)",
    "padding": "Union[int, Tuple[int], str] (default: 0; options: 'valid', 'same')",
    "dilation": "Union[int, Tuple[int]] (default: 1)",
    "groups": "int (default: 1)"
  },
  "Mandatory Parameters": ["input", "weight"],
  "Mutually Exclusive Parameter Pairs": [],
  "Mandatory Coexistence Parameters": [],
  "Conditional Mutual Exclusion Parameters": [
    ["stride", "padding", "(isinstance(padding, str) and padding == 'same') and (stride != 1)"]
  ]
}

```



### 输出结果_tes_5：tf.keras.layers.SeparableConv2D

```
{
  "Parameter type": {
    "filters": "int",
    "kernel_size": "Union[int, Tuple[int, int]]",
    "strides": "Union[int, Tuple[int, int]] (default: (1, 1))",
    "padding": "str (default: 'valid'; options: 'valid', 'same')",
    "data_format": "str (default: keras config; options: 'channels_last', 'channels_first')",
    "dilation_rate": "Union[int, Tuple[int, int]] (default: (1, 1))",
    "depth_multiplier": "int (default: 1)",
    "activation": "Optional[Callable] (e.g., tf.nn.relu)",
    "use_bias": "bool (default: True)",
    "depthwise_initializer": "Optional[Initializer] (default: 'glorot_uniform')",
    "pointwise_initializer": "Optional[Initializer] (default: 'glorot_uniform')",
    "bias_initializer": "Optional[Initializer] (default: 'zeros')",
    "depthwise_regularizer": "Optional[Regularizer]",
    "pointwise_regularizer": "Optional[Regularizer]",
    "bias_regularizer": "Optional[Regularizer]",
    "activity_regularizer": "Optional[Regularizer]",
    "depthwise_constraint": "Optional[Constraint]",
    "pointwise_constraint": "Optional[Constraint]",
    "bias_constraint": "Optional[Constraint]"
  },
  "Mandatory Parameters": ["filters", "kernel_size"],
  "Mutually Exclusive Parameter Pairs": [],
  "Mandatory Coexistence Parameters": [],
  "Conditional Mutual Exclusion Parameters": [
    ["strides", "dilation_rate", "(strides > 1) & (dilation_rate > 1)"]
  ]
}

```

