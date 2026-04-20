# TODO: we might eventually want to just export
# the user facing main functions and keep the utils internal

from .data import load_chetty_data, standardize

from .spurtransform import (
    haversine_distance,
    get_distance_matrix,
    nn_matrix,
    iso_matrix,
    demean_matrix,
    get_sigma_lbm,
    lbmgls_matrix,
    cluster_matrix,
    transform,
    spurtransform,
    get_transformation_stats,
)

from .spurhalflife import (
    HalfLifeResult,
    spatial_persistence,
    spurhalflife,
)
from .pipeline import SpurResult, spur

from .spurtest import (
    SpurTestResult,
    get_distmat_normalized,
    get_R,
    get_sigma_dm,
    lvech,
    getcbar,
    _cholesky_upper,
    getpow_qf,
    get_ha_parm_I1,
    get_ha_parm_I0,
    spurtest_i1,
    spurtest_i0,
    get_sigma_residual,
    get_ha_parm_I1_residual,
    spurtest_i1resid,
    spurtest_i0resid,
    spurtest,
)

__all__ = [
    "load_chetty_data",
    "standardize",
    "haversine_distance",
    "get_distance_matrix",
    "nn_matrix",
    "iso_matrix",
    "demean_matrix",
    "get_sigma_lbm",
    "lbmgls_matrix",
    "cluster_matrix",
    "transform",
    "spurtransform",
    "get_transformation_stats",
    "HalfLifeResult",
    "spatial_persistence",
    "spurhalflife",
    "SpurResult",
    "spur",
    "SpurTestResult",
    "get_distmat_normalized",
    "get_R",
    "get_sigma_dm",
    "lvech",
    "getcbar",
    "_cholesky_upper",
    "getpow_qf",
    "get_ha_parm_I1",
    "get_ha_parm_I0",
    "spurtest_i1",
    "spurtest_i0",
    "get_sigma_residual",
    "get_ha_parm_I1_residual",
    "spurtest_i1resid",
    "spurtest_i0resid",
    "spurtest",
]
