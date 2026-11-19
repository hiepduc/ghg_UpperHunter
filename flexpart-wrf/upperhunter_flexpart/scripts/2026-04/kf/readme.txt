(1) To produce KF and RTS unconstrained hourly and daily emission scaling factors

bounded_batch_MAP_validation.py             constrained_smoother_active_set.py            run_kf_upperhunter_hourly_obs_daily_state.py
compare_bounded_KF_vs_bounded_batch_MAP.py  equality_constrained_smoother_validation.py   run_rts_upperhunter_daily.py
compare_KF_4DVar_total_emissions.py         plot_hourly_KF_scaling_factors.py             run_rts_upperhunter_full_footprint_augmented.py
compare_KF_batch_MAP_corrected.py           __pycache__                                   run_rts_upperhunter_full_footprint.py
compare_RTS_corrected_MAP.py                run_kf_upperhunter_full_footprint_bounded.py  validate_RTS_MAP_stable.py
compare_RTS_direct_MAP.py                   run_kf_upperhunter_full_footprint.py
compare_RTS_vs_4DVar_unconstrained.py       run_kf_upperhunter_full_footprint_revised.py

(a) Run run_kf_upperhunter_hourly_obs_daily_state.py
to produce KF hourly and daily scaling factors

Can compare this with 4DVar emission : compare_KF_4DVar_total_emissions.py

Can compare this with FLEXPART-WRF Bayesian unconstrained : compare_KF_batch_MAP_corrected.py

Can plot KF hourly scaling factors: plot_hourly_KF_scaling_factors.py

(b) Run run_rts_upperhunter_full_footprint_augmented.py
to produce RTS-KF smoother hourly and daily emission scaling factors

To check RTS and MAP (FLEXPART-WRF Bayesian unconstrained): validate_RTS_MAP_stable.py
Can compare this RTS with 4DVar : compare_RTS_vs_4DVar_unconstrained.py
Can compare this RTS with MAP (FLEXPART-WRF Bayesian unconstrained)): compare_RTS_corrected_MAP.py (or compare_RTS_direct_MAP.py?)
 
(2) To produce KF and RTS unconstrained hourly and daily emission scaling factors

(a) Run run_kf_upperhunter_full_footprint_bounded.py
to produce hourly and daily emission scaling factors 
Run constrained_smoother_active_set.py (OR bounded_batch_MAP_validation.py)
to produce MAP bounded (FLEXPART-WRF Bayesian constrained)

Script A — bounded_batch_MAP_validation.py

using scipy.optimize.lsq_linear.

Script B — constrained_smoother_active_set.py

using our own active-set QP algorithm.

Can compare bounded KF with bounded MAP: compare_bounded_KF_vs_bounded_batch_MAP.py

To check the MAP bounded, use equality_constrained_smoother_validation.py which is a validation script for the bounded batch MAP solution
(b) Run rts_projected_bounded_upperhunter.py

Important: This is a recursive approximate constrained RTS smoother. It applies the RTS backward recursion to the saved 26-dimensional augmented states and projects the smoothed estimates onto the [0,5] bounds. Because the bounds make the posterior non-Gaussian, it is not an exact constrained RTS solution. After running it, compare its scaling factors with your validated global constrained batch MAP before deciding whether it is suitable as the final method for your analysis.

To check RTS results: 

(3) Your results now support the following distinctions:

Method
What it estimates
Status


Unbounded Kalman filter
Filtered estimates using observations available up to each time
Validated previously

Unbounded RTS smoother
Estimates using the full observation period
Previously matched unbounded batch MAP

Bounded Kalman filter
Sequential estimates constrained to [0, 5]
Implemented

rts_bounded_upperhunter_v2.py
Local constrained backward approximation
Does not reproduce global MAP

rts_global_constrained_smoother_upperhunter.py
Global box-constrained posterior MAP
Matches bounded batch MAP

Bounded batch MAP
Global constrained Bayesian optimum
Independent numerical reference

Bottom line: Keep rts_global_constrained_smoother_upperhunter.py as your validated global reference solver.

(4) To plot  bounded KF and RTS time series of emission scaling factors at the 13 mines

Run
plot_bounded_KF_vs_RTS_timeseries_hourly.py

(5) To plot the comparison of unbounded and bounded RTS-KF daily emission scaling factors

Run
plot_bounded_vs_unbounded_RTS.py

 
