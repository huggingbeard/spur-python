
clear all
set more off
adopath + "D:\UZHechist Dropbox\Joachim Voth\SPUR-Stata\SPUR_code"

import delimited using "D:\UZHechist Dropbox\Joachim Voth\claudecode\spur-python\spurtest_data.csv", clear
rename lat s_1
rename lon s_2

set seed 42

* Test spurhalflife on y_i1
spurhalflife y_i1, q(10) nrep(50000) level(95) latlong
di "y_i1: ci_l=" %12.4f r(ci_l) ", ci_u=" %12.4f r(ci_u) ", max_dist=" %12.4f r(max_dist)

* Test on y_i0 (should have short half-life)
spurhalflife y_i0, q(10) nrep(50000) level(95) latlong
di "y_i0: ci_l=" %12.4f r(ci_l) ", ci_u=" %12.4f r(ci_u) ", max_dist=" %12.4f r(max_dist)

* Normalized distance
spurhalflife y_i1, q(10) nrep(50000) level(95) latlong normdist
di "y_i1 (norm): ci_l=" %9.4f r(ci_l) ", ci_u=" %9.4f r(ci_u)
