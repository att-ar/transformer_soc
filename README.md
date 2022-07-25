# Transformers + TensorFlow for SOC Estimation

Building a transformer neural network using TensorFlow and Transformers in Python with the goal of prediciting Li-ion (LFP chemistry) State of Charge based on real time voltage, current and delta time data.

The transformers' input will be voltage, current, delta time and previous SOC points in a window of data (of size ```G.window_size```) <br>
The voltage, current and delta time data will be from time: $$t - \text{windowsize} \rightarrow t$$ and the previous soc data will be from time: $$t - \text{windowsize} - 1 \rightarrow t - 1$$

The output should be the SOC prediction at time $t$

**Still a work in progress, I am focusing on the PyTorch network first because PyTorch is more efficient than TensorFlow**
