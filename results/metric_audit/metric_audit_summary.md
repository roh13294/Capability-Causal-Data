# Metric Audit

## confident_wrong
- shifted examples: 300, failures: 277, correct: 23, failure rate: 0.923
- AUROC confidence: 0.085, entropy: 0.085, margin: 0.085, ShiftRisk: 0.977, 1-CR: 0.977
- high-confidence failures: 139, high-confidence ShiftRisk AUROC: nan
- flags: high-confidence subset AUROC undefined; ShiftRisk adds value over confidence

## negative_controls
- shifted examples: 120, failures: 61, correct: 59, failure rate: 0.508
- AUROC confidence: 0.139, entropy: 0.139, margin: 0.139, ShiftRisk: 0.919, 1-CR: 0.919
- high-confidence failures: 0, high-confidence ShiftRisk AUROC: nan
- flags: high-confidence subset AUROC undefined; ShiftRisk adds value over confidence

## negative_controls
- shifted examples: 120, failures: 40, correct: 80, failure rate: 0.333
- AUROC confidence: 0.811, entropy: 0.811, margin: 0.811, ShiftRisk: 0.999, 1-CR: 0.999
- high-confidence failures: 0, high-confidence ShiftRisk AUROC: nan
- flags: high-confidence subset AUROC undefined; ShiftRisk adds value over confidence

## negative_controls
- shifted examples: 120, failures: 58, correct: 62, failure rate: 0.483
- AUROC confidence: 0.548, entropy: 0.548, margin: 0.548, ShiftRisk: 0.414, 1-CR: 0.414
- high-confidence failures: 0, high-confidence ShiftRisk AUROC: nan
- flags: high-confidence subset AUROC undefined; confidence already solves failure prediction better than ShiftRisk

## negative_controls
- shifted examples: 120, failures: 52, correct: 68, failure rate: 0.433
- AUROC confidence: 0.935, entropy: 0.935, margin: 0.935, ShiftRisk: 0.775, 1-CR: 0.775
- high-confidence failures: 0, high-confidence ShiftRisk AUROC: nan
- flags: high-confidence subset AUROC undefined; confidence already solves failure prediction better than ShiftRisk

## synthetic
- shifted examples: 128, failures: 41, correct: 87, failure rate: 0.320
- AUROC confidence: 0.559, entropy: 0.559, margin: 0.559, ShiftRisk: 0.964, 1-CR: 0.964
- high-confidence failures: 0, high-confidence ShiftRisk AUROC: nan
- flags: high-confidence subset AUROC undefined; ShiftRisk adds value over confidence

## synthetic/lambda_sweep
- shifted examples: 256, failures: 4, correct: 252, failure rate: 0.016
- AUROC confidence: 0.992, entropy: 0.992, margin: 0.992, ShiftRisk: 0.853, 1-CR: 0.853
- high-confidence failures: 0, high-confidence ShiftRisk AUROC: nan
- flags: confidence: low failure count; entropy: low failure count; margin: low failure count; ShiftRisk: low failure count; causal reliability: low failure count; high-confidence subset AUROC undefined; confidence already solves failure prediction better than ShiftRisk

## synthetic/lambda_sweep
- shifted examples: 256, failures: 1, correct: 255, failure rate: 0.004
- AUROC confidence: 1.000, entropy: 1.000, margin: 1.000, ShiftRisk: 0.471, 1-CR: 0.471
- high-confidence failures: 0, high-confidence ShiftRisk AUROC: nan
- flags: confidence: low failure count; entropy: low failure count; margin: low failure count; ShiftRisk: low failure count; causal reliability: low failure count; high-confidence subset AUROC undefined; confidence already solves failure prediction better than ShiftRisk

## synthetic/lambda_sweep
- shifted examples: 256, failures: 1, correct: 255, failure rate: 0.004
- AUROC confidence: 0.996, entropy: 0.996, margin: 0.996, ShiftRisk: 0.047, 1-CR: 0.047
- high-confidence failures: 0, high-confidence ShiftRisk AUROC: nan
- flags: confidence: low failure count; entropy: low failure count; margin: low failure count; ShiftRisk: low failure count; causal reliability: low failure count; high-confidence subset AUROC undefined; confidence already solves failure prediction better than ShiftRisk

## synthetic/lambda_sweep
- shifted examples: 256, failures: 0, correct: 256, failure rate: 0.000
- AUROC confidence: nan, entropy: nan, margin: nan, ShiftRisk: nan, 1-CR: nan
- high-confidence failures: 0, high-confidence ShiftRisk AUROC: nan
- flags: confidence: AUROC undefined: only one class present; entropy: AUROC undefined: only one class present; margin: AUROC undefined: only one class present; ShiftRisk: AUROC undefined: only one class present; causal reliability: AUROC undefined: only one class present; high-confidence subset AUROC undefined

## synthetic/lambda_sweep
- shifted examples: 256, failures: 10, correct: 246, failure rate: 0.039
- AUROC confidence: 0.928, entropy: 0.928, margin: 0.928, ShiftRisk: 0.763, 1-CR: 0.763
- high-confidence failures: 1, high-confidence ShiftRisk AUROC: 0.927
- flags: confidence already solves failure prediction better than ShiftRisk

## synthetic/lambda_sweep
- shifted examples: 256, failures: 3, correct: 253, failure rate: 0.012
- AUROC confidence: 0.970, entropy: 0.970, margin: 0.970, ShiftRisk: 0.690, 1-CR: 0.690
- high-confidence failures: 2, high-confidence ShiftRisk AUROC: 0.748
- flags: confidence: low failure count; entropy: low failure count; margin: low failure count; ShiftRisk: low failure count; causal reliability: low failure count; confidence already solves failure prediction better than ShiftRisk

## synthetic/lambda_sweep
- shifted examples: 256, failures: 3, correct: 253, failure rate: 0.012
- AUROC confidence: 0.997, entropy: 0.997, margin: 0.997, ShiftRisk: 0.281, 1-CR: 0.281
- high-confidence failures: 0, high-confidence ShiftRisk AUROC: nan
- flags: confidence: low failure count; entropy: low failure count; margin: low failure count; ShiftRisk: low failure count; causal reliability: low failure count; high-confidence subset AUROC undefined; confidence already solves failure prediction better than ShiftRisk

## synthetic/seed_variance
- shifted examples: 256, failures: 55, correct: 201, failure rate: 0.215
- AUROC confidence: 0.767, entropy: 0.767, margin: 0.767, ShiftRisk: 0.634, 1-CR: 0.634
- high-confidence failures: 10, high-confidence ShiftRisk AUROC: 0.663
- flags: confidence already solves failure prediction better than ShiftRisk

## synthetic/seed_variance
- shifted examples: 256, failures: 0, correct: 256, failure rate: 0.000
- AUROC confidence: nan, entropy: nan, margin: nan, ShiftRisk: nan, 1-CR: nan
- high-confidence failures: 0, high-confidence ShiftRisk AUROC: nan
- flags: confidence: AUROC undefined: only one class present; entropy: AUROC undefined: only one class present; margin: AUROC undefined: only one class present; ShiftRisk: AUROC undefined: only one class present; causal reliability: AUROC undefined: only one class present; high-confidence subset AUROC undefined

## synthetic/seed_variance
- shifted examples: 256, failures: 11, correct: 245, failure rate: 0.043
- AUROC confidence: 0.886, entropy: 0.886, margin: 0.886, ShiftRisk: 0.729, 1-CR: 0.729
- high-confidence failures: 5, high-confidence ShiftRisk AUROC: 0.902
- flags: confidence already solves failure prediction better than ShiftRisk

## synthetic/seed_variance
- shifted examples: 256, failures: 4, correct: 252, failure rate: 0.016
- AUROC confidence: 0.993, entropy: 0.993, margin: 0.993, ShiftRisk: 0.402, 1-CR: 0.402
- high-confidence failures: 1, high-confidence ShiftRisk AUROC: 0.376
- flags: confidence: low failure count; entropy: low failure count; margin: low failure count; ShiftRisk: low failure count; causal reliability: low failure count; confidence already solves failure prediction better than ShiftRisk

## synthetic/seed_variance
- shifted examples: 256, failures: 21, correct: 235, failure rate: 0.082
- AUROC confidence: 0.824, entropy: 0.824, margin: 0.824, ShiftRisk: 0.792, 1-CR: 0.792
- high-confidence failures: 5, high-confidence ShiftRisk AUROC: 0.861
- flags: confidence already solves failure prediction better than ShiftRisk

## synthetic/seed_variance
- shifted examples: 256, failures: 1, correct: 255, failure rate: 0.004
- AUROC confidence: 1.000, entropy: 1.000, margin: 1.000, ShiftRisk: 0.902, 1-CR: 0.902
- high-confidence failures: 0, high-confidence ShiftRisk AUROC: nan
- flags: confidence: low failure count; entropy: low failure count; margin: low failure count; ShiftRisk: low failure count; causal reliability: low failure count; high-confidence subset AUROC undefined; confidence already solves failure prediction better than ShiftRisk

## synthetic/seed_variance
- shifted examples: 256, failures: 65, correct: 191, failure rate: 0.254
- AUROC confidence: 0.717, entropy: 0.717, margin: 0.717, ShiftRisk: 0.690, 1-CR: 0.690
- high-confidence failures: 14, high-confidence ShiftRisk AUROC: 0.727
- flags: confidence already solves failure prediction better than ShiftRisk

## synthetic/seed_variance
- shifted examples: 256, failures: 0, correct: 256, failure rate: 0.000
- AUROC confidence: nan, entropy: nan, margin: nan, ShiftRisk: nan, 1-CR: nan
- high-confidence failures: 0, high-confidence ShiftRisk AUROC: nan
- flags: confidence: AUROC undefined: only one class present; entropy: AUROC undefined: only one class present; margin: AUROC undefined: only one class present; ShiftRisk: AUROC undefined: only one class present; causal reliability: AUROC undefined: only one class present; high-confidence subset AUROC undefined

## synthetic/seed_variance
- shifted examples: 256, failures: 17, correct: 239, failure rate: 0.066
- AUROC confidence: 0.885, entropy: 0.885, margin: 0.885, ShiftRisk: 0.679, 1-CR: 0.679
- high-confidence failures: 4, high-confidence ShiftRisk AUROC: 0.850
- flags: confidence already solves failure prediction better than ShiftRisk

## synthetic/seed_variance
- shifted examples: 256, failures: 0, correct: 256, failure rate: 0.000
- AUROC confidence: nan, entropy: nan, margin: nan, ShiftRisk: nan, 1-CR: nan
- high-confidence failures: 0, high-confidence ShiftRisk AUROC: nan
- flags: confidence: AUROC undefined: only one class present; entropy: AUROC undefined: only one class present; margin: AUROC undefined: only one class present; ShiftRisk: AUROC undefined: only one class present; causal reliability: AUROC undefined: only one class present; high-confidence subset AUROC undefined

## synthetic/seed_variance
- shifted examples: 256, failures: 15, correct: 241, failure rate: 0.059
- AUROC confidence: 0.901, entropy: 0.901, margin: 0.901, ShiftRisk: 0.812, 1-CR: 0.812
- high-confidence failures: 3, high-confidence ShiftRisk AUROC: 0.879
- flags: confidence already solves failure prediction better than ShiftRisk

## synthetic/seed_variance
- shifted examples: 256, failures: 1, correct: 255, failure rate: 0.004
- AUROC confidence: 1.000, entropy: 1.000, margin: 1.000, ShiftRisk: 0.467, 1-CR: 0.467
- high-confidence failures: 0, high-confidence ShiftRisk AUROC: nan
- flags: confidence: low failure count; entropy: low failure count; margin: low failure count; ShiftRisk: low failure count; causal reliability: low failure count; high-confidence subset AUROC undefined; confidence already solves failure prediction better than ShiftRisk

## synthetic/seed_variance
- shifted examples: 256, failures: 18, correct: 238, failure rate: 0.070
- AUROC confidence: 0.895, entropy: 0.895, margin: 0.895, ShiftRisk: 0.660, 1-CR: 0.660
- high-confidence failures: 3, high-confidence ShiftRisk AUROC: 0.841
- flags: confidence already solves failure prediction better than ShiftRisk

## synthetic/seed_variance
- shifted examples: 256, failures: 0, correct: 256, failure rate: 0.000
- AUROC confidence: nan, entropy: nan, margin: nan, ShiftRisk: nan, 1-CR: nan
- high-confidence failures: 0, high-confidence ShiftRisk AUROC: nan
- flags: confidence: AUROC undefined: only one class present; entropy: AUROC undefined: only one class present; margin: AUROC undefined: only one class present; ShiftRisk: AUROC undefined: only one class present; causal reliability: AUROC undefined: only one class present; high-confidence subset AUROC undefined

## synthetic/seed_variance
- shifted examples: 256, failures: 22, correct: 234, failure rate: 0.086
- AUROC confidence: 0.863, entropy: 0.863, margin: 0.863, ShiftRisk: 0.635, 1-CR: 0.635
- high-confidence failures: 3, high-confidence ShiftRisk AUROC: 0.859
- flags: confidence already solves failure prediction better than ShiftRisk

## synthetic/seed_variance
- shifted examples: 256, failures: 1, correct: 255, failure rate: 0.004
- AUROC confidence: 1.000, entropy: 1.000, margin: 1.000, ShiftRisk: 0.216, 1-CR: 0.216
- high-confidence failures: 0, high-confidence ShiftRisk AUROC: nan
- flags: confidence: low failure count; entropy: low failure count; margin: low failure count; ShiftRisk: low failure count; causal reliability: low failure count; high-confidence subset AUROC undefined; confidence already solves failure prediction better than ShiftRisk

## synthetic/seed_variance
- shifted examples: 256, failures: 26, correct: 230, failure rate: 0.102
- AUROC confidence: 0.897, entropy: 0.897, margin: 0.897, ShiftRisk: 0.668, 1-CR: 0.668
- high-confidence failures: 3, high-confidence ShiftRisk AUROC: 0.884
- flags: confidence already solves failure prediction better than ShiftRisk

## synthetic/seed_variance
- shifted examples: 256, failures: 0, correct: 256, failure rate: 0.000
- AUROC confidence: nan, entropy: nan, margin: nan, ShiftRisk: nan, 1-CR: nan
- high-confidence failures: 0, high-confidence ShiftRisk AUROC: nan
- flags: confidence: AUROC undefined: only one class present; entropy: AUROC undefined: only one class present; margin: AUROC undefined: only one class present; ShiftRisk: AUROC undefined: only one class present; causal reliability: AUROC undefined: only one class present; high-confidence subset AUROC undefined

## synthetic/seed_variance
- shifted examples: 256, failures: 16, correct: 240, failure rate: 0.062
- AUROC confidence: 0.904, entropy: 0.904, margin: 0.904, ShiftRisk: 0.769, 1-CR: 0.769
- high-confidence failures: 3, high-confidence ShiftRisk AUROC: 0.870
- flags: confidence already solves failure prediction better than ShiftRisk

## synthetic/seed_variance
- shifted examples: 256, failures: 0, correct: 256, failure rate: 0.000
- AUROC confidence: nan, entropy: nan, margin: nan, ShiftRisk: nan, 1-CR: nan
- high-confidence failures: 0, high-confidence ShiftRisk AUROC: nan
- flags: confidence: AUROC undefined: only one class present; entropy: AUROC undefined: only one class present; margin: AUROC undefined: only one class present; ShiftRisk: AUROC undefined: only one class present; causal reliability: AUROC undefined: only one class present; high-confidence subset AUROC undefined

## synthetic/shortcut_sweep
- shifted examples: 256, failures: 1, correct: 255, failure rate: 0.004
- AUROC confidence: 1.000, entropy: 1.000, margin: 1.000, ShiftRisk: 0.196, 1-CR: 0.196
- high-confidence failures: 0, high-confidence ShiftRisk AUROC: nan
- flags: confidence: low failure count; entropy: low failure count; margin: low failure count; ShiftRisk: low failure count; causal reliability: low failure count; high-confidence subset AUROC undefined; confidence already solves failure prediction better than ShiftRisk

## synthetic/shortcut_sweep
- shifted examples: 256, failures: 1, correct: 255, failure rate: 0.004
- AUROC confidence: 0.992, entropy: 0.992, margin: 0.992, ShiftRisk: 0.957, 1-CR: 0.957
- high-confidence failures: 1, high-confidence ShiftRisk AUROC: 0.957
- flags: confidence: low failure count; entropy: low failure count; margin: low failure count; ShiftRisk: low failure count; causal reliability: low failure count; confidence already solves failure prediction better than ShiftRisk

## synthetic/shortcut_sweep
- shifted examples: 256, failures: 9, correct: 247, failure rate: 0.035
- AUROC confidence: 0.971, entropy: 0.971, margin: 0.971, ShiftRisk: 0.884, 1-CR: 0.884
- high-confidence failures: 0, high-confidence ShiftRisk AUROC: nan
- flags: high-confidence subset AUROC undefined; confidence already solves failure prediction better than ShiftRisk

## synthetic/shortcut_sweep
- shifted examples: 256, failures: 8, correct: 248, failure rate: 0.031
- AUROC confidence: 0.975, entropy: 0.975, margin: 0.975, ShiftRisk: 0.790, 1-CR: 0.790
- high-confidence failures: 1, high-confidence ShiftRisk AUROC: 0.918
- flags: confidence already solves failure prediction better than ShiftRisk

## synthetic/shortcut_sweep
- shifted examples: 256, failures: 29, correct: 227, failure rate: 0.113
- AUROC confidence: 0.789, entropy: 0.789, margin: 0.789, ShiftRisk: 0.650, 1-CR: 0.650
- high-confidence failures: 7, high-confidence ShiftRisk AUROC: 0.755
- flags: confidence already solves failure prediction better than ShiftRisk

## synthetic/shortcut_sweep
- shifted examples: 256, failures: 102, correct: 154, failure rate: 0.398
- AUROC confidence: 0.631, entropy: 0.631, margin: 0.631, ShiftRisk: 0.677, 1-CR: 0.677
- high-confidence failures: 47, high-confidence ShiftRisk AUROC: 0.668
- flags: ShiftRisk adds value over confidence

## text
- shifted examples: 256, failures: 175, correct: 81, failure rate: 0.684
- AUROC confidence: 0.449, entropy: 0.449, margin: 0.449, ShiftRisk: 0.725, 1-CR: 0.725
- high-confidence failures: 117, high-confidence ShiftRisk AUROC: 0.728
- flags: ShiftRisk adds value over confidence

## vision
- shifted examples: 192, failures: 169, correct: 23, failure rate: 0.880
- AUROC confidence: 0.533, entropy: 0.533, margin: 0.533, ShiftRisk: 0.466, 1-CR: 0.466
- high-confidence failures: 100, high-confidence ShiftRisk AUROC: 0.787
- flags: confidence already solves failure prediction better than ShiftRisk

## vision/counterfactual_mismatch
- shifted examples: 192, failures: 176, correct: 16, failure rate: 0.917
- AUROC confidence: 0.641, entropy: 0.641, margin: 0.641, ShiftRisk: 0.372, 1-CR: 0.372
- high-confidence failures: 176, high-confidence ShiftRisk AUROC: 0.372
- flags: confidence already solves failure prediction better than ShiftRisk

## vision/counterfactual_mismatch
- shifted examples: 192, failures: 169, correct: 23, failure rate: 0.880
- AUROC confidence: 0.391, entropy: 0.391, margin: 0.391, ShiftRisk: 0.375, 1-CR: 0.375
- high-confidence failures: 167, high-confidence ShiftRisk AUROC: 0.957
- flags: confidence already solves failure prediction better than ShiftRisk

## vision/counterfactual_mismatch
- shifted examples: 192, failures: 96, correct: 96, failure rate: 0.500
- AUROC confidence: 0.891, entropy: 0.891, margin: 0.891, ShiftRisk: 0.000, 1-CR: 0.000
- high-confidence failures: 0, high-confidence ShiftRisk AUROC: nan
- flags: high-confidence subset AUROC undefined; confidence already solves failure prediction better than ShiftRisk

## vision/counterfactual_mismatch
- shifted examples: 192, failures: 176, correct: 16, failure rate: 0.917
- AUROC confidence: 0.641, entropy: 0.641, margin: 0.641, ShiftRisk: 0.316, 1-CR: 0.316
- high-confidence failures: 176, high-confidence ShiftRisk AUROC: 0.316
- flags: confidence already solves failure prediction better than ShiftRisk

## vision/counterfactual_mismatch
- shifted examples: 192, failures: 169, correct: 23, failure rate: 0.880
- AUROC confidence: 0.391, entropy: 0.391, margin: 0.391, ShiftRisk: 0.606, 1-CR: 0.606
- high-confidence failures: 167, high-confidence ShiftRisk AUROC: 0.000
- flags: ShiftRisk adds value over confidence

## vision/counterfactual_mismatch
- shifted examples: 192, failures: 96, correct: 96, failure rate: 0.500
- AUROC confidence: 0.891, entropy: 0.891, margin: 0.891, ShiftRisk: 0.999, 1-CR: 0.999
- high-confidence failures: 0, high-confidence ShiftRisk AUROC: nan
- flags: high-confidence subset AUROC undefined; ShiftRisk adds value over confidence

## vision/counterfactual_mismatch
- shifted examples: 192, failures: 176, correct: 16, failure rate: 0.917
- AUROC confidence: 0.641, entropy: 0.641, margin: 0.641, ShiftRisk: 0.688, 1-CR: 0.688
- high-confidence failures: 176, high-confidence ShiftRisk AUROC: 0.688
- flags: ShiftRisk adds value over confidence

## vision/counterfactual_mismatch
- shifted examples: 192, failures: 169, correct: 23, failure rate: 0.880
- AUROC confidence: 0.391, entropy: 0.391, margin: 0.391, ShiftRisk: 0.569, 1-CR: 0.569
- high-confidence failures: 167, high-confidence ShiftRisk AUROC: 0.000
- flags: ShiftRisk adds value over confidence

## vision/counterfactual_mismatch
- shifted examples: 192, failures: 96, correct: 96, failure rate: 0.500
- AUROC confidence: 0.891, entropy: 0.891, margin: 0.891, ShiftRisk: 0.362, 1-CR: 0.362
- high-confidence failures: 0, high-confidence ShiftRisk AUROC: nan
- flags: high-confidence subset AUROC undefined; confidence already solves failure prediction better than ShiftRisk
