import os


def ddp_print(*args, print_rank: int = 0, **kwargs):
    """
    DDP 环境下的专属打印函数。
    只有当前进程的 local_rank 等于指定的 print_rank（或非 DDP 环境下的 -1）时，才会执行 print。
    防止多卡训练时终端被日志冲烂。
    """
    # 动态获取当前进程的 rank，非 DDP 环境默认是 -1
    local_rank = int(os.environ.get("LOCAL_RANK", -1))

    if local_rank in [-1, print_rank]:
        print(*args, **kwargs)