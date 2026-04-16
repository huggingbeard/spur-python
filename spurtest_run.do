
* SPUR Test Validation

clear all
set more off
adopath + "D:\UZHechist Dropbox\Joachim Voth\SPUR-Stata\SPUR_code"

* Load data
import delimited using "D:\UZHechist Dropbox\Joachim Voth\claudecode\spur-python\spurtest_data.csv", clear

* Rename coords
rename lat s_1
rename lon s_2

* Set seed for MC (not exactly matching Python, but stable across Stata runs)
set seed 42

* I(1) test on y_i1
spurtest i1 y_i1, q(10) nrep(200000) latlong
scalar lfur_i1_yi1 = r(teststat)
scalar pvalue_i1_yi1 = r(p)
scalar haparm_i1_yi1 = r(ha_param)

* I(0) test on y_i1
spurtest i0 y_i1, q(10) nrep(200000) latlong
scalar lfst_i0_yi1 = r(teststat)
scalar pvalue_i0_yi1 = r(p)

* I(1) test on y_i0
spurtest i1 y_i0, q(10) nrep(200000) latlong
scalar lfur_i1_yi0 = r(teststat)
scalar pvalue_i1_yi0 = r(p)

* I(0) test on y_i0
spurtest i0 y_i0, q(10) nrep(200000) latlong
scalar lfst_i0_yi0 = r(teststat)
scalar pvalue_i0_yi0 = r(p)

* Display all results
di ""
di "===== STATA RESULTS ====="
di "i1 test on y_i1: LR=" %9.4f lfur_i1_yi1 ", p=" %9.4f pvalue_i1_yi1 ", ha=" %9.4f haparm_i1_yi1
di "i0 test on y_i1: LR=" %9.4f lfst_i0_yi1 ", p=" %9.4f pvalue_i0_yi1
di "i1 test on y_i0: LR=" %9.4f lfur_i1_yi0 ", p=" %9.4f pvalue_i1_yi0
di "i0 test on y_i0: LR=" %9.4f lfst_i0_yi0 ", p=" %9.4f pvalue_i0_yi0
di "========================="
