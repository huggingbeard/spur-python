
* Run Stata spurtest N times with different seeds at nrep=1000
clear all
set more off
adopath + "D:\UZHechist Dropbox\Joachim Voth\SPUR-Stata\SPUR_code"

import delimited using "D:\UZHechist Dropbox\Joachim Voth\claudecode\spur-python\spurtest_data.csv", clear
rename lat s_1
rename lon s_2

* Storage for results
tempfile results
postfile handler run LR pvalue ha_param using `results', replace

* Run 20 times with different seeds
forval i = 1/20 {
    local seed = `i' * 1000 + 7
    set seed `seed'
    spurtest i1 y_i1, q(10) nrep(1000) latlong
    post handler (`i') (r(teststat)) (r(p)) (r(ha_param))
}

postclose handler

use `results', clear
su
export delimited using "D:\UZHechist Dropbox\Joachim Voth\claudecode\spur-python\stata_mc_results.csv", replace
di "Stata MC variability exported"
