
* Validate residual tests
clear all
set more off
adopath + "D:\UZHechist Dropbox\Joachim Voth\SPUR-Stata\SPUR_code"

import delimited using "D:\UZHechist Dropbox\Joachim Voth\claudecode\spur-python\spurtest_data.csv", clear
rename lat s_1
rename lon s_2

set seed 42

* I1resid: regress y_i1 on x
spurtest i1resid y_i1 x, q(10) nrep(200000) latlong
di "i1resid y_i1 x: LR=" %9.4f r(teststat) ", p=" %9.4f r(p) ", ha=" %9.4f r(ha_param)

* I0resid: regress y_i1 on x
spurtest i0resid y_i1 x, q(10) nrep(200000) latlong
di "i0resid y_i1 x: LR=" %9.4f r(teststat) ", p=" %9.4f r(p) ", ha=" %9.4f r(ha_param)

* I1resid: regress y_i0 on x
spurtest i1resid y_i0 x, q(10) nrep(200000) latlong
di "i1resid y_i0 x: LR=" %9.4f r(teststat) ", p=" %9.4f r(p) ", ha=" %9.4f r(ha_param)

* I0resid: regress y_i0 on x
spurtest i0resid y_i0 x, q(10) nrep(200000) latlong
di "i0resid y_i0 x: LR=" %9.4f r(teststat) ", p=" %9.4f r(p) ", ha=" %9.4f r(ha_param)
