## Data reading and preprocessing

Depending on the available data format, one may need their own implementations of data-preprocessing.

Please make sure that all time-series, once converted into a `pytorch` tensor, is in the format <img src="https://render.githubusercontent.com/render/math?math=(*, F, T)">, where <img src="https://render.githubusercontent.com/render/math?math=F"> is the number of features, <img src="https://render.githubusercontent.com/render/math?math=T"> is the number of time-steps.
