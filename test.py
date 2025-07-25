from config import *
from function import *
from main import *
from generate_prompt import *

a = '''1. Role:
        You are an expert in [torch], with deep knowledge of its API design, functionality, and practical usage across a wide range of scenarios.

    ---
    2. Background and Context:

    (1) API Documentation.

    We provide below the official documentation for the API to be analyzed: 
    [torch.nn.functional.conv1d]
    This documentation specifies the API’s function signature, behavior, supported data types, argument definitions, default values, constraints, and usage examples, enabling precise understanding of its operational semantics.
    The specific API documentation content is as below:
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
    input: input tensor of shape :math:`(\text{minibatch}, \text{in\_channels}, iW)`
    weight: filters of shape :math:`(\text{out\_channels}, \frac{\text{in\_channels}}{\text{groups}}, kW)`
    bias: optional bias of shape :math:`(\text{out\_channels})`. Default: ``None``
    stride: the stride of the convolving kernel. Can be a single number or
      a one-element tuple `(sW,)`. Default: 1
    padding: implicit paddings on both sides of the input. Can be a string {'valid','same'},
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
]

    (2) Parameter Dependency Types.

    In deep learning libraries such as [torch], many APIs accept a set of configurable parameters, enabling multiple usage patterns based on different parameter combinations. 
    These parameters may exhibit various types of interdependencies that govern valid and efficient usage. Specifically, we consider the following three relationship types:

    - "Mandatory Parameters": Parameters that must be explicitly specified for the API to function correctly.  
    - "Mutual Exclusion Parameter Pairs": Pairs of parameters that cannot be used together in a single call due to logical or operational conflicts.  
    - "Conditional Mutual Exclusion Parameter Pairs" refer to parameter pairs that are not universally mutually exclusive but become incompatible only under specific conditions. In these cases, the mutual exclusion depends on certain value combinations, e.g., the parameters conflict only when both assume particular values or satisfy certain conditions based on their assigned values. Thus, these are a subset of "Mutual Exclusion Parameter Pairs" that exhibit conditional, rather than absolute, exclusivity.
    - "Mandatory coexistence parameters": Sets of parameters that must be provided together to ensure valid configuration or meaningful behavior.
    - "Conditional Mutual Exclusion Parameters": When the parameters meet certain conditions, it will prevent the function from running.

    ---
    3. Your Tasks:

    Based on the documentation and definitions provided:

    1. Determine the Type of Each Parameter.
    For each explicitly defined parameter in [fun_string], determine its type, such as: tensor, int, str, optional tensor, etc.
    2. Identify Parameter Dependency Structures as accurately and completely as possible:
    - List all "Mandatory Parameters"  
    - Identify any "Mutual Exclusion Parameter Pairs"  
    - Identify any "Conditional Mutual Exclusion Parameter Pairs" 
    - Identify any "Mandatory Coexistence Parameters"

    ---
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
    "Mutually Exclusive Parameter Pairs": [["...", "..."],...],
    "Mandatory Coexistence Parameters": [["...", "...", "..."], ["...", "..."],...],
    "Conditional Mutual Exclusion Parameters":["para_1", "para_2","(para_1>1)&(para_2>1)"] 
    }

    ---
    5.Examples:
    Output Examples: 
    {
    "Parameter type": {
    "input": "Tensor (shape: (minibatch, in_channels, iW))",
    "weight": "Tensor (shape: (out_channels, in_channels // groups, kW))",
    "bias": "Optional[Tensor] (shape: (out_channels))",
    "stride": "Union[int, Tuple[int]] (default: 1)",
    "padding": "Union[int, Tuple[int], str] (default: 0; options: 'valid','same')",
    "dilation": "Union[int, Tuple[int]] (default: 1)",
    "groups": "int (default: 1)"
    },
    "Mandatory Parameters": ["input", "weight"],
    "Mutually Exclusive Parameter Pairs": [],
    "Mandatory Coexistence Parameters": [],
    "Conditional Mutual Exclusion Parameters": ["strides", "dilation","(strides>1)&(dilation>1)"] 
    }

    6.Notions:
    Only output the json content of the example in the output format, do not add explanations.assistant

{
    "Parameter type": {
        "input": "Tensor (shape: (minibatch, in_channels, iW))",
        "weight": "Tensor (shape: (out_channels, in_channels // groups  kW))",
        "bias": "Optional[Tensor] (shape: (out_channels))",
        "stride": "Union[int, Tuple[int]] (default: 1)",
        "padding": "Union[int, Tuple[int], str] (default: 0; options: 'valid','same')",
        "dilation": "Union[int  Tuple[int]] (default: 1)",
        "groups": "int (default: 1)"
    },
    "Mandatory Parameters": ["input", "weight"],
    "Mutually Exclusive Parameter Pairs": [],
    "Mandatory Coexistence Parameters": [],
    "Conditional Mutual Exclusion Parameters": ["strides", "dilation","(strides>1)&(dilation>1)"]
}'''

b = handle_output(a, model_path)

print(b)
append_api_condition_to_json(f'test.json', '''a''', b)