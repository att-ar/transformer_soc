# Transformers + TensorFlow and Pandas for SOC Estimation

**The Testing branch is the most up to date**

Building a transformer neural network using TensorFlow and Transformers in Python with the goal of prediciting Li-ion (LFP chemistry) State of Charge based on real time voltage, current and delta time data.

The transformers' input will be voltage, current, delta time and previous SOC points in a batch of windowed data of shape:<br>
```(G.batch_size, G.window_size, G.num_features)```

The voltage, current and soc data will be from time: $$t - \text{windowsize} - 1 \rightarrow t - 1$$

The output should be the SOC prediction at time $t$ for each batch, the output shape should be `(G.batch_size, 1)`
