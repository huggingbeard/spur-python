
clear all
set more off
adopath + "D:\UZHechist Dropbox\Joachim Voth\SPUR-Stata\SPUR_code"

import delimited using "D:\UZHechist Dropbox\Joachim Voth\claudecode\spur-python\spurtest_data.csv", clear
rename lat s_1
rename lon s_2

set seed 42

* I(0) test - check ha_param
spurtest i0 y_i1, q(10) nrep(50000) latlong
di "i0 y_i1: LR=" %9.4f r(teststat) ", p=" %9.4f r(p) ", ha=" %9.4f r(ha_param)

* Run again with different seed
set seed 99999
spurtest i0 y_i1, q(10) nrep(50000) latlong
di "i0 y_i1 (seed2): LR=" %9.4f r(teststat) ", p=" %9.4f r(p) ", ha=" %9.4f r(ha_param)

* Try larger nrep
set seed 42
spurtest i0 y_i1, q(10) nrep(200000) latlong
di "i0 y_i1 nrep=200k: LR=" %9.4f r(teststat) ", p=" %9.4f r(p) ", ha=" %9.4f r(ha_param)
