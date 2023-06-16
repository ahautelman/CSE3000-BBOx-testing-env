from bbox._src.prefabs import (
    chemopt_gmm, convex, convex_nonstationary,
    rbf_gp, matern12_gp, matern32_gp
)

try:
    from bbox.prefabs import env
except ModuleNotFoundError:
    # Case already handled by bbox.__init__
    pass
