# Regional air pollutant forecasting

This repository introduces a deep-learning model that makes **REGIONAL** air pollution forecasting for the future 48 hours. It combines the *ground observation* of the past 72 hours and the *WRF-CMAQ* results for the future 48 hours. The proposed constitues a variety of structures: LSTM encoder-decoders, bidirectional LSTMs, time-distributed dense layers, and a newly proposed *broadcasting layer*.

![model](https://user-images.githubusercontent.com/61111285/130564120-b05270c5-b6af-4f74-9913-1b93a45e230b.png)

The main contributions of the project is:

- Breaking the spatial confinement of the ground-observation-based deep-learning air pollution forecast to the ground monitor stations.
- Superceding the widely-used spatial correction (interpolation methods).
- Proposing the novel broadcasting layer that may be applied to deep-learning tasks of similar nature. 

The publication related to the project is under composition and review.
