## Model structure utilities

This folder implements the structures that will be used in `../model.py`. The contents include:

- `gnn.py`: Containing the implementation of the broadcasting layer. The broadcasting layer can be regarded as a "unidirectional bipartite graph message passing".
- `lstm.py`: Bidirectional LSTMs and LSTM encoder-decoders
- `unet.py`: Time-series U-Net, *deprecated*, was an alternative approach for processing the WRF-CMAQ data.
- `utils.py`: The dense layer, including the time-distributed version, which is not originally supported by `pytorch`.
