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
    spur_persistence,
    spurhalflife,
)
from .pipeline import SpurResult, spur

from .spurtest import (
    SpurTestResult,
    normalized_distmat,
    get_r,
    get_sigma_dm,
    lvech,
    get_cbar,
    cholesky_upper,
    get_pow_qf,
    get_ha_param_i1,
    get_ha_param_i0,
    spurtest_i1,
    spurtest_i0,
    get_sigma_residual,
    get_ha_param_i1_residual,
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
    "spur_persistence",
    "spurhalflife",
    "SpurResult",
    "spur",
    "SpurTestResult",
    "normalized_distmat",
    "get_r",
    "get_sigma_dm",
    "lvech",
    "get_cbar",
    "cholesky_upper",
    "get_pow_qf",
    "get_ha_param_i1",
    "get_ha_param_i0",
    "spurtest_i1",
    "spurtest_i0",
    "get_sigma_residual",
    "get_ha_param_i1_residual",
    "spurtest_i1resid",
    "spurtest_i0resid",
    "spurtest",
]
