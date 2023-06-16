from bbox.version import __version__, __version_info__
from bbox._src.core import (
    Function,
    FunctionWrapper,
    get_param,
    as_callable,
    as_transformed,
)

try:
    # Setup extras for jumanji based environment backend.
    from bbox import env
    from bbox._src.env.function_env import as_bandit, as_nonstationary_bandit

except ModuleNotFoundError:
    from warnings import warn
    warn("The `env` API could not be loaded.", UserWarning)
    del warn

from bbox import functions
from bbox import procgen
from bbox import types
from bbox import utils
from bbox import wrappers
from bbox import prefabs
