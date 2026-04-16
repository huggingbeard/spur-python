
* SPUR Validation - Run from Stata
* This file runs spurtransform on test data

clear all
set more off

* Add SPUR to adopath
adopath + "D:\UZHechist Dropbox\Joachim Voth\SPUR-Stata\SPUR_code"

* Load test data
import delimited using "D:\UZHechist Dropbox\Joachim Voth\claudecode\spur-python\validation_data.csv", clear

* Rename coords to s_1 and s_2 as required by SPUR
rename lat s_1
rename lon s_2

* NN transform
spurtransform y x, prefix(nn_) transformation(nn) latlong

* ISO transform (200km radius)
spurtransform y x, prefix(iso_) transformation(iso) radius(200000) latlong

* LBM-GLS transform (default/recommended)
spurtransform y x, prefix(lbm_) transformation(lbmgls) latlong

* Cluster transform (latitude tercile)
xtile cluster = s_1, nq(3)
spurtransform y x, prefix(cl_) transformation(cluster) clustvar(cluster) latlong

* Export results
export delimited using "D:\UZHechist Dropbox\Joachim Voth\claudecode\spur-python\validation_stata_output.csv", replace

di "Stata validation complete!"
