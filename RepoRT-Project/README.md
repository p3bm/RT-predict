# Method-Aware Prediction of Small Molecule LC RT using Structure and Method Conditions

Investigating the use of the RepoRT database to createa a regression model that can predict LC RT for a given chemical structure _and_ given LC method conditions.

--- 

26Aug2025 - create repository, start work.

Primary target - LC RT in minutes. Will likely work entirely with RT values scaled to the gradient time.
+ Go through each gradient.tsv file and extract end time of gradient (not all methods include a re-equilibration step at the end).
+ Then alter gradient interpolation code to only interpolate across the gradient, ignoring any equilibration stage. Any RTs that fall outside of the gradient time should be removed.

New column dwell time values have been calculated as I believe the calculation used to produce the values in the RepoRT processed_data folder is incorrect (see relevant R-local script in /scripts). Calculation used here given below, where the column length and ID are in cm, the flowrate is in mL/min, and epsilon is 0.49 for superficially porous phase columns and 0.66 for totally porous phase columns.

$t_{0, col} = \frac{(\frac{ID}{2})^2 * \pi * Length * epsilon}{flowrate}$


