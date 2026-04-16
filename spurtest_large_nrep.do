
* Test with large nrep for convergence
clear all
set more off
adopath + "D:\UZHechist Dropbox\Joachim Voth\SPUR-Stata\SPUR_code"

import delimited using "D:\UZHechist Dropbox\Joachim Voth\claudecode\spur-python\spurtest_data.csv", clear
rename lat s_1
rename lon s_2

set seed 12345

* Run with 50000 reps
spurtest i1 y_i1, q(10) nrep(50000) latlong
di "nrep=50000: LR=" %9.4f r(teststat) ", p=" %9.4f r(p) ", ha=" %9.4f r(ha_param)

* Run again with different seed
set seed 67890
spurtest i1 y_i1, q(10) nrep(50000) latlong
di "nrep=50000 (seed2): LR=" %9.4f r(teststat) ", p=" %9.4f r(p) ", ha=" %9.4f r(ha_param)

* Run with 100000 reps
set seed 12345
spurtest i1 y_i1, q(10) nrep(100000) latlong
di "nrep=100000: LR=" %9.4f r(teststat) ", p=" %9.4f r(p) ", ha=" %9.4f r(ha_param)
