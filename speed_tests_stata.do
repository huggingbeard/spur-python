* Speed tests for SPUR Stata
* Generates synthetic data of varying N and times each function
* Outputs to stata_speed_results.csv for comparison with Python

clear all
set more off
adopath + "D:\UZHechist Dropbox\Joachim Voth\SPUR-Stata\SPUR_code"

* Storage for results
tempfile results
postfile handler str20 func n nrep double time_sec using `results', replace

* Function to time spurtransform
local ns 100 300 1000 3000
foreach n of local ns {
    clear
    set obs `n'
    set seed 42
    gen s_1 = runiform(30, 50)
    gen s_2 = runiform(-100, -80)
    gen y = rnormal() + 0.5 * s_1
    gen x = rnormal()

    * NN
    timer clear 1
    timer on 1
    spurtransform y, prefix(d_) transformation(nn) latlong
    timer off 1
    qui timer list 1
    post handler ("transform_nn") (`n') (0) (r(t1))
    drop d_y

    * ISO
    timer clear 2
    timer on 2
    spurtransform y, prefix(d_) transformation(iso) radius(500000) latlong
    timer off 2
    qui timer list 2
    post handler ("transform_iso") (`n') (0) (r(t2))
    drop d_y

    * LBM-GLS
    timer clear 3
    timer on 3
    spurtransform y, prefix(d_) transformation(lbmgls) latlong
    timer off 3
    qui timer list 3
    post handler ("transform_lbmgls") (`n') (0) (r(t3))
    drop d_y

    di "N=`n': transforms done"
}

* Tests - vary N and nrep
local ns 30 100 300
local nreps 10000 50000
foreach n of local ns {
    clear
    set obs `n'
    set seed 42
    gen s_1 = runiform(30, 50)
    gen s_2 = runiform(-100, -80)
    gen y = rnormal() + 0.5 * s_1

    foreach nrep of local nreps {
        set seed 42

        * i1
        timer clear 4
        timer on 4
        qui spurtest i1 y, q(15) nrep(`nrep') latlong
        timer off 4
        qui timer list 4
        post handler ("spurtest_i1") (`n') (`nrep') (r(t4))

        * i0
        timer clear 5
        timer on 5
        qui spurtest i0 y, q(15) nrep(`nrep') latlong
        timer off 5
        qui timer list 5
        post handler ("spurtest_i0") (`n') (`nrep') (r(t5))

        * halflife
        timer clear 6
        timer on 6
        qui spurhalflife y, q(15) nrep(`nrep') level(95) latlong
        timer off 6
        qui timer list 6
        post handler ("spurhalflife") (`n') (`nrep') (r(t6))

        di "N=`n', nrep=`nrep': tests done"
    }
}

postclose handler
use `results', clear
export delimited using "D:\UZHechist Dropbox\Joachim Voth\claudecode\spur-python\stata_speed_results.csv", replace
di "Stata speed results exported"
