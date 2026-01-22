from .get_nasbench101 import get_top_arch as get_nasbench101_top_arch
from .get_nasbench201 import get_top_arch as get_nasbench201_top_arch
from .get_nasbench301 import get_top_arch as get_nasbench301_top_arch
from .get_hwnasbench import get_top_arch as get_hwnasbench_top_arch

def get_top_arch(args):
    if args.nasbench == "nasbench101":
        return get_nasbench101_top_arch(args)
    elif args.nasbench == "nasbench201":
        return get_nasbench201_top_arch(args)
    elif args.nasbench == "nasbench301":
        return get_nasbench301_top_arch(args)
    elif args.nasbench == "hwnasbench":
        return get_hwnasbench_top_arch(args)
    else:
        raise NotImplementedError(f"Top architecture retrieval not implemented for {args.nasbench}")