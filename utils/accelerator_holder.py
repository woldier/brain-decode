# -*- coding:utf-8 -*-
"""
 @FileName   : accelerator_holder.py
 @Time       : 5/1/24 11:37 PM
 @Author     : Woldier Wong
 @Description: A generic accelerator holder utility class. For distributed training, fp16 support (for better control,
                use manual method).
               一个通用的 accelerator holder 工具类. 用于分布式训练, 支持fp16(为了更好的控制, 使用手动方式).

               To use it you only need:
               想要使用只需要:
               >>> from utils.accelerator_hold import get_accelerator
               >>> get_accelerator().print("hello accelerator in DDP!")

"""
from accelerate import Accelerator, AutocastKwargs, DistributedDataParallelKwargs
from accelerate.utils import set_seed
import warnings, inspect

"""
Support for fp16, however, not all models in training need to be fp16, 
so it is set to manual conversion instead of auto cast. 
Therefore, it is set to manual cast, instead of auto cast.
支持fp16, 然而, 并不是训练中所有的model 都需要进行fp16, 
因此 设置成了手动转换, 而不是auto cast.

Then where it is needed:
随后在需要使用的地方:
    >>>from accelerate import Accelerator, AutocastKwargs
    >>>accelerator = Accelerator(mixed_precision='fp16', kwargs_handlers=[AutocastKwargs(enabled=False)])
    >>> with accelerator.autocast(autocast_handler=AutocastKwargs(enabled=True)):
    >>>     net(input)
    >>>    
"""

"""
The number of steps that should pass before gradients are accumulated. A number > 1 should be combined with
`Accelerator.accumulate`. If not passed, will default to the value in the environment variable
`ACCELERATE_GRADIENT_ACCUMULATION_STEPS`. Can also be configured through a `GradientAccumulationPlugin`.

Should be used in place of `torch.nn.utils.clip_grad_norm_`.

Example:
    >>> accelerator = Accelerator(gradient_accumulation_steps=2)
    >>> dataloader, model, optimizer, scheduler = accelerator.prepare(dataloader, model, optimizer, scheduler)
    >>> for input, target in dataloader:
    ...     optimizer.zero_grad()
    ...     output = model(input)
    ...     loss = loss_func(output, target)
    ...     accelerator.backward(loss)
    ...     if accelerator.sync_gradients:
    ...         accelerator.clip_grad_norm_(model.parameters(), max_grad_norm)
    ...     optimizer.step()

> Note
For a discussion of the max_grad_norm parameter setting, there is the blog.
对于 max_grad_norm 参数设置的探讨, 有blog 可以参考
https://blog.csdn.net/zhaohongfei_358/article/details/122820992
"""
accelerator: Accelerator = Accelerator(
    mixed_precision='fp16',
    gradient_accumulation_steps=2,
    kwargs_handlers=[AutocastKwargs(enabled=False),
                     DistributedDataParallelKwargs(find_unused_parameters=True)]
)

# fix the seed for reproducibility
# Helper function for reproducible behavior to set the seed in `random`, `numpy`, `torch`
set_seed(42)

__all__ = ["get_accelerator", "assert_", "warning_","print"]


def get_accelerator() -> Accelerator:
    """
    获取accelerator
    :return:
    """
    return accelerator


def assert_(condition: bool, message: str):
    """
    Print assertion error msg based on accelerator
    基于accelerator 并在主线程中打印断言错误 msg

    """
    if not condition:
        accelerator.print(message)
        raise AssertionError(message)


def warning_(msg: str):
    """
    warning info
    :param msg:
    :return:
    """
    # 获取调用 warning_ 函数的代码位置信息
    caller_frame = inspect.stack()[1]
    caller_filename = caller_frame.filename
    caller_lineno = caller_frame.lineno

    if accelerator.is_local_main_process:
        warnings.warn_explicit(msg, category=UserWarning, filename=caller_filename, lineno=caller_lineno)


def print(*args, **kwargs):
    accelerator.print(*args, **kwargs)
