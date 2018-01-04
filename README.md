# cifar-innovation

[`TensorFlow` and `TensorBoard`] 

**ML Algorithms (implemented by TensorFlow):**  
Feedforward Neural Network and Convolutional Neural Networks  

**Envs:**  
Anaconda and Python 3.5

**Motivation**  
Theoretically, based on [Fourier Transform](https://en.wikipedia.org/wiki/Fourier_transform), we could represent any waveform by an infinite sum of periodic sines waves and cosine waves.  

Nowadays, most neural networks use non-periodic and monotonic activation functions, i.e. relu function, across the whole architecture, but I would like to do some experiments to see if the periodic activation functions could potential help the performance or not.

**Innovation**  
Usually, people use the same monotonic activation function within the same layer.
However, based on the inspiration from Fourier Transform, I created a custom layer, which composed by two same/different kind of periodic/non-periodic activation functions units within the same layer. I put this custom layer in the middle of the FNNs network for classification task and compare it with the baseline result (baseline: all linear activation in the network).  

**Raw Data:**   
CIFAR10 image data set batch 1 (10,000 images)

**Experiment Configurations**  
   * Flowchart: 
![exp_flowchart](./assets/exp_flowchart.png)  

[pre-network configurations]: 
   * Flowthrough
   * Convolutional net (4 layers - conv2d, maxpool, conv2d, maxpool)

[conv2d layer activation configurations]
   * [linear , sin, relu]

[custom layers connections]:
   * constant (1-to-2 with constant weight == 1)
   * 2 weights per units (1-to-2 with weights)
   * dense (128-to-128 with weights)

[custom layer activation configurations]: 
   * 15 total combinations [linear, sin, cos, tan, relu] x [linear, sin, cos, tan, relu]

**Experiment Results**  
   * Validation Accuracy in different experiment configurations: 
![exp_results_1](./assets/exp_results_1.png)  
![exp_results_2](./assets/exp_results_2.png)  

**Summary**  
In this experiment:
   * CNNs + FNNs perform better than purely FNNs as expected 
   * activation function usage performance: relu > linear > sin > cos >> tan  

Here are some lessons I learned/confirmed from this experiment:
   * convolutional network boosts the accuracy and makes network more robust, but the run time increase dramatically as expected.
      * In this experiment, validation accuracy increase from 38% to 56%, but run time is about 13x
      * runtime: 1.55 mins/100 epochs without conv net, 5.18 mins/25 epochs with conv net
   * relu function generally makes the overall network performance better and more robust to the periodic activation inside the network
   * tan activation funcation will make the cost curve unstable and it will also make poorly performance 
   
   * connection configuration (constant weights/sparse weights/dense weights):
      * periodic activation functions are easily overfitting with dense connection in purely FNNs

   * using multiple periodic activation function within the same layer:
      * the result did not perform better than using purely relu/linear function. (i.e. sin + cos v.s. relu + relu or iden + iden)
      * if we use both periodic activation function for custom layer, it will be sensitive to the previous network activation function or connection configuration. (vulnerable in dense configuration which is not preferable) 
   
   * performance-wise not getting better at least in this experiment
  

**Tensorboard Samples**  
   * Conv net graph: 
![conv_net_graph](./assets/conv_net_graph.png)  
   * Cust layer graph: 
![cust_layer_graph](./assets/cust_layer_graph.png)  
   * Accuracy sample: 
![exp_dense_flatten_valid_acc](./assets/exp_dense_flatten_valid_acc.png)  
![exp_dense_conv_valid_acc](./assets/exp_dense_conv_valid_acc.png)  
   * Cost sample: 
![exp_dense_flatten_valid_cost](./assets/exp_dense_flatten_valid_cost.png)  
![exp_dense_conv_valid_cost](./assets/exp_dense_conv_valid_cost.png)  

