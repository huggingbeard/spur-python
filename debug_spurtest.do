
* Debug: check intermediate values in Python vs Stata
clear all
set more off
adopath + "D:\UZHechist Dropbox\Joachim Voth\SPUR-Stata\SPUR_code"

import delimited using "D:\UZHechist Dropbox\Joachim Voth\claudecode\spur-python\spurtest_data.csv", clear
rename lat s_1
rename lon s_2

* Load the mlib functions
mata mata mlib index

* Test getcbar via mata
mata {
    // Replicate get_s_matrix
    s = st_data(., ("s_1","s_2"))
    latlongflag = 1

    // Get distance matrix
    distmat = getdistmat_normalized(s)
    printf("Distance matrix max: %9.6f\n", max(distmat))

    // Test getcbar
    c_095 = getcbar(0.95, distmat)
    c_05 = getcbar(0.5, distmat)
    c_001 = getcbar(0.01, distmat)
    printf("getcbar(0.95) = %9.6f\n", c_095)
    printf("getcbar(0.5)  = %9.6f\n", c_05)
    printf("getcbar(0.01) = %9.6f\n", c_001)

    // Test sigdm_bm
    sigdm_bm = get_sigma_lbm_dm(distmat)
    printf("sigdm_bm trace: %9.6f\n", trace(sigdm_bm))
    printf("sigdm_bm[1,1]:  %9.6f\n", sigdm_bm[1,1])
    printf("sigdm_bm[1,2]:  %9.6f\n", sigdm_bm[1,2])

    // Test R
    R = get_R(sigdm_bm, 10)
    om_ho = R' * sigdm_bm * R
    printf("om_ho trace:  %9.6f\n", trace(om_ho))
}
