<div align='center' ><b><font size='6'>Prediction of Molten Pool Width in Laser Directed Energy Deposition Driven by CNN-BiLSTM Model</font></b></div>

<center>15<sup>th</sup> National Conference on Laser Processing 2022</center>

<center>
<a href="https://www.researchgate.net/profile/Wang-Yanghui">Yanghui Wang</a><sup>1</sup>, 
<a href="https://www.researchgate.net/profile/Kaixiong-Hu-2">Kaixiong Hu</a><sup>1,*</sup>, 
<a href="https://www.researchgate.net/profile/W-Li-8">Weidong Li</a><sup>2</sup> 
</center>
<center> <sup>1</sup>School of Transportation and Logistics Engineering, Wuhan University of Technology, Wuhan, 430063, China<br><sup>2</sup>School of Mechanical Engineering, University of Shanghai for Science and Technology, Shanghai, 200093, China<br>E-mail: kaixiong.hu@whut.com
</center>
<center><a href="https://meegee.gitee.io/15thmeeting/">Webpage</a>| <a href="https://www.sciencedirect.com/science/article/pii/S1526612522002389">Paper</a>|<a href="https://meegee.gitee.io/15thmeeting/">Alternative URL</a></center></center>

## Abstract

Laser Directed Energy Deposition (L-DED) is a promising metal Additive Manufacturing (AM) technology capable of fabricating thin-walled parts to support some high-value applications. Accurate and efficient prediction on the melt pool width is critical to support in-situ control of L-DED for part quality assurance. Nevertheless, owing to the intricate physical mechanisms of the process, it is challenging to designing an effective approach to accomplish the prediction target. To tackle the issue, in this research, a new data model-driven predictive approach, which is enabled by a hybrid machine learning model namely CNN-BiLSTM, is presented. High prediction accuracy and efficiency are achievable through innovative measures in the research, that is, (i) the CNN-BiLSTM model is designed and configured by addressing the characteristics of the [L-DED process](https://www.sciencedirect.com/topics/engineering/deposition-process); (ii) process parameters related to the deposition and heat accumulation phenomena during the L-DED process are extensively considered to strengthen the prediction accuracy. Experiments for thin-walled part fabrication were conducted to validate and benchmark the approach. In average, 4.286% of the mean absolute percentage error (MAPE) was acquired, and the prediction time took by the approach was only 0.04% of that by a finite element analysis (FEA) approach. Compared to the LSTM model, the [BiLSTM](https://www.sciencedirect.com/topics/engineering/long-short-term-memory) model and the CNN-LSTM model, MAPEs of the CNN-BiLSTM model were improved by 27.0%, 17.3% and 12.6%, respectively. It demonstrates that the approach is competent in producing good-quality thin-walled parts using the L-DED process.

## 1.L-DED Process

Additive Manufacturing (AM) has received increasing attention from academic and industrial societies. The great flexibility of AM provides unique opportunities to fabricate freeform parts through stacking melt metal powders, metal wires or plastic wires layer-by-layer utilizing a laser, electric arc or electron beam as a high-energy and focused heat source [[1]](https://www.sciencedirect.com/science/article/pii/S1526612522002389#bb0005). Among various [AM technologies](https://www.sciencedirect.com/topics/engineering/additive-manufacturing-technology), Laser Directed Energy Deposition (L-DED) is immensely promising. It is suitable for producing freeform thin-walled parts with complex geometries or structures in high-value applications ranging from aeronautics, automobile, nuclear, mold and die [[2]](https://www.sciencedirect.com/science/article/pii/S1526612522002389#bb0010), [[3]](https://www.sciencedirect.com/science/article/pii/S1526612522002389#bb0015). A typical [L-DED system](https://www.sciencedirect.com/topics/engineering/deposition-system) consists of a nozzle mounted on a multi-axis arm, which deposits powders onto a substrate flexibly, where the powders are melted through a laser mounted on the multi-axis arm. The process is illustrated in Fig. 1. Nevertheless, inaccurate and non-uniform deposition qualities are essential challenges in L-DED [[4]](https://www.sciencedirect.com/science/article/pii/S1526612522002389#bb0020). A small change in the process could induce significant variations in the transient heating and cooling rate, affecting the desired [bead geometry](https://www.sciencedirect.com/topics/engineering/bead-geometry) and eventually influencing the mechanical property of an AM part. The melt pool width is a primary indicator of the deposition quality and [temperature distribution](https://www.sciencedirect.com/topics/engineering/temperature-distribution) for thin-walled parts [[5]](https://www.sciencedirect.com/science/article/pii/S1526612522002389#bb0025). Accurate and efficient prediction on the melt pool width in real-time is critical for implementing an in-situ control system in order to achieve a high-quality [L-DED process](https://www.sciencedirect.com/topics/engineering/deposition-process) [[6]](https://www.sciencedirect.com/science/article/pii/S1526612522002389#bb0030), [[7]](https://www.sciencedirect.com/science/article/pii/S1526612522002389#bb0035).

![Fig. 1](https://ars.els-cdn.com/content/image/1-s2.0-S1526612522002389-gr1.jpg)

<center>Fig. 1. Illustration of the L-DED process.</center>

## 2.Data-driven predictive approach

### 2.1.Key process parameters in the L-DED process

In the L-DED process, the effect of [heat conduction](https://www.sciencedirect.com/topics/engineering/heat-conduction) makes the residual heat from prior layers not completely dissipated before a following layer is deposited [[8]](https://www.sciencedirect.com/science/article/pii/S1526612522002389#bb0150). Thus, a higher initial temperature for this following layer is induced, which leads to a non-uniform melt pool width. To count the heat accumulation effect between the prior layers and the following layer, it is necessary to define a layer index, which is an integer number that corresponds to a specific deposited layer. For the same layer, studies have shown that the heat accumulation effect is closely related to time [[9]](https://www.sciencedirect.com/science/article/pii/S1526612522002389#bb0155). The following formulas are defined to specify the relevant physical phenomena in the L-DED process.

Assuming the laser beam intensity is uniformly distributed, the energy absorbed *Q* by the powder is represented below [[10]](https://www.sciencedirect.com/science/article/pii/S1526612522002389#bb0160):
$$
Q=(α_p r_p^2 P_l)/(r_l^2 ) ∆t \tag{1}
$$
where $$α_p$$ is the absorption coefficient of the metal powder; $$r_p$$** is the radius of the metal powder; $$P_l$$ is the laser power; $$r_l$$is the radius of the laser spot; ∆*t* is the time during the layer forming.

The heat dissipation *E* of powders during this period of time is described below [[11]](https://www.sciencedirect.com/science/article/pii/S1526612522002389#bb0165):

$$
E=4πr_p^2 ϵδ(T^4-T_a^4 )∆t \tag{2}
$$
where *δ* is the Stefan-Boltzmann constant; *ϵ* is the full-emission coefficient; *T* is the temperature of the melt pool; $$T_a$$ is the temperature of the external environment.

When the heat absorption is greater than the heat dissipation, the heat accumulation effect will occur. The final heat absorbed $$Q_{in}$$ has a positive linear relationship with time, which is represented as below:
$$
Q_{in}=Q-E=[(α_p r_p^2 P_l)/(r_l^2 )-4πr_p^2 ϵδ(T^4-T_a^4 )]∆t \tag{3}
$$
The above formulas describe the physical phenomenon of the L-DED process, i.e., heat accumulation in layers and between layers. It clearly indicates that time plays an essential role on the heat accumulation effect. Thus, in the CNN-BiLSTM-driven predictive model, it needs to take the heat accumulation effect into the time dependent parameter, i.e., a time index, as one of the inputs. Moreover, since the images of each melt pool are recorded in a chronological order (interval: 0.125 s, [deposition time](https://www.sciencedirect.com/topics/engineering/deposition-time) for one layer: 10s, for each layer: about 79 images taken), the images at different positions of the same layer indirectly reflect the changes of heat accumulation over time. Thus, an indirect index, namely image index, is also employed as an input in the predictive model. It is an integer number that corresponds to the chronological order of the images in a layer. That is, with the image index and layer index, the heat accumulation effect in the L-DED process for multilayer thin-walled structures can be considered comprehensively and fed into the predictive model. Meanwhile, some essential process parameters in the above formulas, including the laser power, powder feeding speed and AM scanning speed, have proved to be the critical factors determining the qualities of fabricated parts and are taken as inputs to the predictive model. Therefore, the inputs of the CNN-BiLSTM model are summarized in Table 1.

<center><b>Table 1. Key process parameters to the CNN-BiLSTM-driven predictive model.</b></center>

|     Input process parameters     |            Value            |
| :------------------------------: | :-------------------------: |
|     Laser power ($$P_l,W$$)      | 800, 1000, 1200, 1400, 1600 |
|  AM scanning speed ($$v.mm/s$$)  |        6, 8, 10, 12         |
| Powder feeding speed ($$w,rpm$$) |        0.8, 1.0, 1.2        |
|        Layer index (*L*)         |            1~10             |
|        Image index (*t*)         |            1~79             |

### 2.2. Design of the CNN-BiLSTM model

As aforementioned, owing to the coupled multi-physical fields in the L-DED process, it is arduous to determine the exact values of the coefficients in the (1), (2), (3). To take the advantage of the pros of data-driven approaches such as robustness and less domain-knowledge required, a CNN-BiLSTM enabled predictive approach is therefore designed.

The CNN-BiLSTM model is composed of a CNN model and subsequent two BiLSTM models. After a flatten layer and a fully connected layer, the melt pool width is generated as the output. As described previously, the two-dimensional CNN (2D-CNN) model is mainly used for feature recognition of two-dimensional images. As for the L-DED process, the raw images of the melt pool can be used as the input. However, 2D-CNN model is weak in processing learning relationships between time-series data [[12]](https://www.sciencedirect.com/science/article/pii/S1526612522002389#bb0170). The one-dimensional CNN (1D-CNN) model has been widely adopted for recognition and extraction of time series dependent features without losing the advantages of the 2D-CNN's translation invariance for feature recognition [[13]](https://www.sciencedirect.com/science/article/pii/S1526612522002389#bb0175). Compared with the 2D-CNN model, the input and structure of the 1D-CNN model is much simpler, and results are generated in a faster operation speed. A 1D vector with five critical process parameters (i.e., laser power, AM scanning speed, powder feeding speed, layer index and image index) are fed into the 1D-CNN model. BiLSTM model is an extension of [Recurrent Neural Network](https://www.sciencedirect.com/topics/engineering/recurrent-neural-network) (RNN), which has been proved effective for time-related information extraction. Nevertheless, it will lose important data when the input sequence is too long [[14]](https://www.sciencedirect.com/science/article/pii/S1526612522002389#bb0180). Therefore, it is necessary to combine the CNN method and the BiLSTM method to strengthen the predictive effectiveness. Thus, in this research, the CNN model is designed to extract important features from the input process parameters, followed by the BiLSTM model to extract heat accumulation and deposition time-related features as the influence of the thermal history on the melt pool has a long-term dependency. Meanwhile, when time goes by, the influence of the previous heat accumulation on the melt pool is becoming weaker, which can be also gradually forgotten by the BiLSTM model.

The CNN model, as detailed in Fig. 2, consists of a convolutional layer (including an [activation function](https://www.sciencedirect.com/topics/engineering/activation-function) layer) and pooling layer. $$x_i=\lbrace x_1,x_2,x_3,x_4,x_5\rbrace$$ is the input vector (the laser power ($$x_1$$), powder feeding speed ($$x_2$$), AM scanning speed ($$x_3$$), deposition layer index ($$x_4$$) and image index ($$x_5$$)). The output vector of the convolutional layer $$\pmb{y_{i,j}}$$ can be calculated below.
$$
\pmb{y_{i,j}}=ReLU(\pmb{b_j}+∑_{m=1}^M+\pmb{w_{m,j}}*\pmb{x_{i+m-1}}) \tag{4}
$$
where $\pmb{b_j}$ is the bias coefficient vector of the *j*th feature map; $\pmb{w}$ is the [weight vector](https://www.sciencedirect.com/topics/engineering/weight-vector) of the kernel; *m* is the index value of the filter; ∗ is a convolution operation; *ReLU* is the ReLU activation function.

![Fig. 2](https://ars.els-cdn.com/content/image/1-s2.0-S1526612522002389-gr2.jpg)

<center>Fig. 2: The structure of the CNN model.</center>

Features extracted from the convolution operation are then passed to the pooling layer. The max pooling is the most commonly used [nonlinear function](https://www.sciencedirect.com/topics/engineering/nonlinear-function) for pooling, which can identify the maximum value of the adjacent region. It has the capability of adjusting over-fitting and reducing convolution features, which operation can be formulated below.
$$
p_{i,j}=\underset{(i-1)×S<r<i×S}{max}{y}_{{r},{j}} \tag{5}
$$
where *S* is the size of the pooling core and *r* is the *r*th neuron.

The BiLSTM model consists of two consecutive LSTM models. The structure of a single LSTM model is shown in Fig. 3. Each unit of the model includes the input gate ( $\pmb{i_t}$ ), the output gate ( $\pmb{h_t}$ ), the forget gate ( $\pmb{f_t}$ ) and the memory unit ( $\pmb{c_t}$ ), as formulated as follows.
$$
\pmb{i_t}=\sigma(\pmb{W_{xi}}\pmb{x_t}+\pmb{W_{hi}}\pmb{h_{t-1}}+\pmb{b_i})\tag{6}
$$

$$
\pmb{f_t}=\sigma(\pmb{W_{xf}}\pmb{x_t}+\pmb{W_{hf}}\pmb{h_{t-1}}+\pmb{b_f})\tag{7}
$$

$$
\pmb{C_t}=\pmb{C_t}×\pmb{f_t}+\pmb{i_t}×tanh(\pmb{W_{xc}}\pmb{x_t}+\pmb{W_{hc}}\pmb{h_{t-1}}+\pmb{b_c})\tag{8}
$$

$$
\pmb{o_t}=\sigma(\pmb{W_{xo}}\pmb{x_t}+\pmb{W_{ho}}\pmb{h_{t-1}}+\pmb{b_o})\tag{9}
$$

$$
\pmb{h_t}=\pmb{o_t}∎tanh(\pmb{C_t}) \tag{10}
$$

where $\pmb{i_t}$ ,$\pmb{f_t}$ ,$\pmb{c_t}$, $\pmb{o_t}$, $\pmb{h_t}$ are the input gate, forget gate, memory unit, and output gate, respectively; *σ* is the sigmoid function with the value between 0 and 1; $\pmb{W_{x}}$ and $\pmb{W_{h}}$ are the weight coefficient vectors; $\pmb{b_i,b_f,b_c}$ and $\pmb{b_{o}}$ are the bias coefficient vectors; ∎ is the Hadamard product; *tanh* is the sigmoid function with the value between −1 and 1.

![Fig. 3](https://ars.els-cdn.com/content/image/1-s2.0-S1526612522002389-gr3.jpg)

<center>Fig. 3: The structure of the LSTM model.</center>

The memory unit remembers the state of the model in the previous period of time. If $\pmb{x_t}$ is input to the memory unit as a new data, the output $\pmb{h_{t-1}}$  of the previous time-step and  $\pmb{x_t}$  are simultaneously used as the current input.

However, LSTM can only process the forward sequence, and the impact of the future thermal history cannot be considered. Bi-directional LSTM (BiLSTM) is then developed based on LSTM, containing more useful information related to the past and future changes of the inputs, as demonstrated in Fig. 4. In this model, the forward LSTM is used for the forward processing sequence with the input sequence of $$\pmb{x_t}=\lbrace x_1,x_2,x_3,x_4,...,x_n\rbrace$$. The backward LSTM is used for the backward processing sequence with the input sequence of $$\pmb{x_t}=\lbrace x_n,x_{n-1},x_{n-2},...,x_1\rbrace$$. After training, their outputs *h*(*t*) are integrated as follows:
$$
h(t)=[\vec{h_t},\vec{h_t}]\tag{11}
$$
where $\vec{h_t}$ and $\vec{h_t}$  are the output vectors of the forward and backward LSTM, respectively.

![Fig. 4](https://ars.els-cdn.com/content/image/1-s2.0-S1526612522002389-gr4.jpg)

<center>Fig. 4: The structure of the BiLSTM model.</center>

by trialing the number of filters, and neurons/size of kernel of each layer, optimal hyperparameters of the model were identified. For the training of the model, [Mean Absolute Error](https://www.sciencedirect.com/topics/engineering/mean-absolute-error) (MAE) was used as a cost function and Adam optimiser with weight decaying was used to update the weights. The structure of the proposed CNN-BiLSTM model is shown in [Fig. 5](https://www.sciencedirect.com/science/article/pii/S1526612522002389#f0025), and some hyperparameters are listed in Table 2.

![Fig. 5](https://ars.els-cdn.com/content/image/1-s2.0-S1526612522002389-gr5_lrg.jpg)

<center>Fig. 5: The structure of the CNN-BiLSTM model.</center>

<center><b>Table 2. Some hyperparameters of the CNN-BiLSTM model.</b></center>

| No.  |      Layer type       | Size of kernel, Number of neurons | Parameter |
| :--: | :-------------------: | :-------------------------------: | :-------: |
|  1   |      Convolution      |             (3, 128)              |   2048    |
|  2   |      Max pooling      |             (3, 128)              |     0     |
|  3   |        BiLSTM         |              (3,300)              |  334,800  |
|  4   |        BiLSTM         |              (3,200)              |  320,800  |
|  5   |        Flatten        |            (None, 600)            |     0     |
|  6   | Fully connected layer |            (None, 50)             |  30,050   |
|  7   |        Output         |             (None, 1)             |    51     |

## 3. Experiments

### 3.1. Platform setup and powder materials

A platform was designed to support this research shown in Fig. 6. It is mainly composed of an IPG 6000 W YLS-6000 K [optical fibre](https://www.sciencedirect.com/topics/engineering/optical-fiber) laser, a powder feeder, a [laser cladding](https://www.sciencedirect.com/topics/engineering/laser-cladding) head with a co-axial [powder nozzle](https://www.sciencedirect.com/topics/engineering/powder-nozzle), a water cooler, an image monitoring system, and a KUKA KR60-HA robot. The image monitoring system is based on a CMOS camera (MV-UBS500-T, Mindvision). Experiments show when the CMOS camera is mounted on the side with an inclination of 55° to the laser cladding head, the clearest and most complete melt pool images during the laser deposition process can be captured, as presented in Fig. 7. The CMOS camera is equipped with a lens (F0816-5MP) with the maximum resolution of 2592 × 1944 pixels and frame rate of 8 fps. A Q235 plate specimen with the dimensions of 400 mm × 200 mm × 5 mm was taken as the substrate.  Commercial 316 L was chosen as the powder material in which the granularity of the powder is 50–150 μm. Powder compositions are given in Table 3.

![Fig. 6](https://ars.els-cdn.com/content/image/1-s2.0-S1526612522002389-gr6.jpg)

<center>Fig. 6: The diagram of the L-DED platform.</center>

![Fig. 7](https://ars.els-cdn.com/content/image/1-s2.0-S1526612522002389-gr7.jpg)

<center>Fig. 7: (a) Schematic configuration; (b) The monitoring system.</center>



<center><b>Table 3. Chemical compositions of the 316 L powders (mass fraction: /%).</b></center>

|  C   |  Cr  |  Si  |  Mo  |  Ni  |  Mn  |  Fe  |
| :--: | :--: | :--: | :--: | :--: | :--: | :--: |
| 0.03 | 17.0 | 0.2  | 2.5  | 12.0 | 0.5  | Bal  |

### 3.2. Melt pool image processing

To extract the melt pool-related feature, i.e., the width, an effective image processing method is imperative. Major image processing steps including image denoising and filtering, threshold segmentation using the OTSU algorithm, edge extraction based on the CANNY algorithm to locate feature boundaries in binary images, and width extraction using a minimum quadrilateral method, as presented schematically in Fig. 8. The melt pool width is defined as the length of the rectangle.

![Fig. 8](https://ars.els-cdn.com/content/image/1-s2.0-S1526612522002389-gr8.jpg)

<center>Fig. 8: The schematic diagram of image processing for a melt pool.</center>

## 4. Results and discussions

### 4.1. Characteristics of the melt pool width

The effectiveness of the approach was validated based on the aforementioned experiments. Fig. 9 shows some examples of the melt pool images from the 1st layer to the 10th layer. In the experiments, the laser power was set for 800 W, 1000 W, 1200 W, 1400 W and 1600 W respectively, the AM scanning speed was 6 mm/s, and the powder feeding speed was 0.8 rpm.

![Fig. 11](https://ars.els-cdn.com/content/image/1-s2.0-S1526612522002389-gr11.jpg)

<center>Fig. 9: Images of melt pools from the 1st layer to the 10th layer of thin-walled parts.</center>

The melt pool widths of all the ten layers of the experimental parts were extracted based on the monitoring system described earlier. Fig. 10 shows the results when the laser power was set at 1200 W, the AM scanning speed was 8 mm/s, and the powder feeding speed was 1.0 rpm. Due to the unstable energy absorption and temperature during the L-DED process, there were some fluctuations in the melt pool width during deposition. It can be observed in Fig. 11 clearly, which demonstrates the cross-sections of ten deposition layers of a part. The widths of the earlier layers of the part were smaller, and the latter layers gradually increased. That is, the fluctuation of the width was relatively larger at the first 5 layers with the mean square error of 0.014; the melt pool in the following layers gradually stabilized, showing much less fluctuation in width with the mean square error of 0.006. This could be caused by the unstable powder utilization rate by the variations of the contact surface between the powders and the substrate, and unstable temperature and physical fields at the early stage of the deposition. The measurements are aligned with the physical nature of the L-DED process. It proves the importance of monitoring the melt pool width, which is highly useful to evaluate factors related to width fluctuations in real-time in order to identify appropriate process parameters to minimize the fluctuations.

![Fig. 12](https://ars.els-cdn.com/content/image/1-s2.0-S1526612522002389-gr12.jpg)

<center>Fig. 10: The melt pool widths of the layers of a part.</center>

![Fig. 13](https://ars.els-cdn.com/content/image/1-s2.0-S1526612522002389-gr13.jpg)

<center>Fig. 11: Cross-sectional geometries of the ten layers of the thin-walled part, (a) to (j) corresponds to the layer 1 to layer 10, respectively.</center>

### 4.2. Performance of the CNN-BiLSTM model

Mean Square Error (MSE), Mean Absolute Error (MAE) and Mean Absolute Percentage Error (MAPE) were defined to evaluate the accuracy of the approach. They are below:
$$
MSE=\frac{1}{n}\sum_{i=1}^{n}\left(\widehat{y_i}-y_i\right)^2 \tag{12}
$$

$$
MAE=\frac{1}{n}\sum_{i=1}^{n}\left|\widehat{y_i}-y_i\right|\tag{13}
$$

$$
MAPE=\frac{1}{n}\sum_{i=1}^{n}{\left|\frac{\widehat{y_i}-y_i}{y_i}\right|} \tag{14}
$$

where ${y_i}$ and $\widehat{y_i}$ represent the measured value and predicted value, respectively.

Fig. 12 shows the predictions using the CNN-BiLSTM model and the measurement of the melt pool width. It can be seen that the CNN-BiLSTM model can predict the dynamic changes of the melt pool width for each layer effectively and it provided more realistic results closer to the ground-truths. To further compare the models, different RNN-based ML models, i.e., LSTM, BiLSTM and CNN-LSTM, were also modelled to predict the melt pool width based on the same dataset to benchmark the CNN-BiLSTM model. The same approach for the hyperparameter of the CNN-BiLSTM model was used. The structures of the four models are summarized in Table 4.

![Fig. 14](https://ars.els-cdn.com/content/image/1-s2.0-S1526612522002389-gr14.jpg)

<center>Fig. 12: CNN-BiLSTM-based prediction and the melt pool width measured by CMOS camera under different process parameters: (a) *P* = 800 W, w = 0.8 rpm, v = 6 mm/s; (b) *P* = 1200 W, w = 0.8 rpm, v = 8 mm/s; (c) P = 1200 W, w = 1.2 rpm, v = 8 mm/s; (d) *P* = 1600 W, w = 1.2 rpm, v = 12 mm/s.</center>

<center><b>Table 4. Structures of the four models for comparisons.</b></center>

|   Model    |                          Structure                           | Iteration | Batch |
| :--------: | :----------------------------------------------------------: | :-------: | :---: |
|    LSTM    | LSTM layer (LSTM:100) - LSTM layer (LSTM:50) - Fully connected layer (Dense:1) |    250    |  200  |
|   BiLSTM   | BiLSTM layer (Bidirectional:100) - BiLSTM layer (Bidirectional:50) - Fully connected layer(Dense:1) |    250    |  150  |
|  CNN-LSTM  | Convolutional layer (Conv1D 3*(3*128)) - Maxpooling layer (MaxPooling1D (3*128)) - LSTM layer (LSTM:100) - LSTM layer (LSTM:50) - Flatten layer - Fully connected layer (Dense:50) - Fully connected layer(Dense:1) |    250    |  150  |
| CNN-BiLSTM | Convolutional layer (Conv1D 3*(3*128)) - Maxpooling layer (MaxPooling1D (3*128)) - BiLSTM layer (Bidirectional:150) - BiLSTM layer (Bidirectional:100) - Flatten layer - Fully connected layer (Dense:50) - Fully connected layer(Dense:1) |    200    |  200  |

Table 5 shows the predicted accuracies of these models. The CNN-BiLSTM model achieved at the best values of the average MSE, MAE and MAPE, i.e., 3.183, 1.234 and 4.286%, respectively. The CNN-LSTM model obtained 4.081, 1.450 and 4.903% in terms of MSE, MAE and MAPE, respectively. Compared to the CNN-LSTM model, MSE, MAE and MAPE of the CNN-BiLSTM model were improved by 22.0%, 14.9% and 12.6%, respectively. The BiLSTM model achieved 4.232, 1.518 and 5.183% in terms of MSE, MAE and MAPE. Compared to the BiLSTM model, the accuracies of CNN-BiLSTM in terms of MSE, MAE and MAPE were improved by 24.8%, 18.7% and 17.3%, respectively. The LSTM model did not obtain satisfactory results in general, and MSE, MAE and MAPE were barely 5.181, 1.731 and 5.873%. Compared with the LSTM model, MSE, MAE and MAPE of the CNN-BiLSTM model were improved by 38.6%, 28.7% and 27.0%, respectively. Based on the comparisons, it can easily draw a conclusion that the CNN-BiLSTM model is much better than other models in terms of predicted accuracy.

<center><b>Table 5. The predictive accuracies achieved by the different models.</b></center>

|   Models   | MSE (pixel2) | MAE (pixel) | MAPE (%) |
| :--------: | :----------: | :---------: | :------: |
|    LSTM    |    5.181     |    1.731    |  5.873   |
|   BiLSTM   |    4.232     |    1.518    |  5.183   |
|  CNN-LSTM  |    4.081     |    1.450    |  4.903   |
| CNN-BiLSTM |    3.183     |    1.234    |  4.286   |

The predicted efficiency of the approach was evaluated as well. Table 6 shows the comparison of the training time and predicted time of the above four models based on the test dataset. The CNN-BiLSTM model took a little longer training and predicted time considering it has the most complex architecture. However, achieving a higher predictive accuracy is of great importance in the L-DED process. Therefore, it can be generally concluded that the CNN-BiLSTM model is a more suitable model to predict the melt pool width.

<center><b>Table 6. The predictive efficiencies of the different models.</b></center>

|   Model    | Training time (s) | Predicted time (ms) |
| :--------: | :---------------: | :-----------------: |
|    LSTM    |        264        |        0.28         |
|   BiLSTM   |        506        |        0.38         |
|  CNN-LSTM  |        626        |        0.47         |
| CNN-BiLSTM |        616        |        0.88         |

### 4.3. Performance comparison with FEA

To further prove the applicability of the approach, an FEA approach was modelled for comparison in terms of accuracy and predicting time. Table 7 shows the process and thermal-physical parameters for the FEA approach.

<center><b>Table 7. Settings for the FEA approach.</b></center>

|                Parameters                |               Values                |
| :--------------------------------------: | :---------------------------------: |
|    Melting temperature (*T**m*, *K*)     |                1733                 |
|         Density (*ρ*, *Kg*/*m*3)         |                7980                 |
| Thermal conductivity (*k*,*W*/*m* ∙ *K*) | 13.3 (293.15*K*) -32.2 (1733.15*K*) |
| Specific heat (*c**p*, *J*/(*Kg* ∙ *K*)) | 470 (293.15*K*) – 876 (1733.15*K*)  |
|      Laser absorptivity of iron (β)      |                 0.6                 |
| Radius of the laser spot (*r**l*, *mm*)  |                2.25                 |
|             Type of the unit             |               Solid60               |
|    Length of the unit (*L**u*, *mm*)     |                  1                  |
|     Width of the unit (*W**u*, *mm*)     |                  1                  |
|    Height of the unit (*H**u*, *mm*)     |                0.032                |
|            Number of the unit            |               23,520                |

Fig. 13 shows the melt pool width, including the measured value (ground-truth), the CNN-BiLSTM-based predicted value and the FEA-based predicted value. It is shown that the accuracy of the prediction by CNN-BiLSTM is better than that by FEA.

![Fig. 16](https://ars.els-cdn.com/content/image/1-s2.0-S1526612522002389-gr16.jpg)

<center>Fig. 13: Widths measured by a CMOS camera, predicted by CNN-BiLSTM and FEA.</center>

Table 8 lists the values of MSE, MAE and MAPE of CNN-BiLSTM and FEA under the conditions of *P* = 800W, *w* = 0.8rpm, *v* = 6mm/s. It revealed that the predicted accuracy of FEA was lower than that of CNN-BiLSTM (19.2% v.s. 4.9% in MAPE). Moreover, the predicted time of CNN-BiLSTM was only 0.04% of that of FEA (4620 s v.s. 1.88 s). Based on the facts that the predicted model established with CNN-BiLSTM was more accurate with a much shorter prediction time in comparison with FEA, it can be concluded that FEA is replaceable by CNN-BiLSTM when developing an in-situ control system for the L-DED process.

<center><b>Table 8. Performance comparison of CNN-BiLSTM and FEA.</b></center>

|  Approach  | MSE (mm2 | MAE (mm) | MAPE (%) | Predicting time (s) |
| :--------: | :------: | :------: | :------: | :-----------------: |
|    FEA     |  0.394   |  0.471   |   19.2   |        4,620        |
| CNN-BiLSTM |  0.033   |  0.122   |   4.9    |        1.88         |

## Acknowledgement

This research was sponsored by the National Natural Science Foundation of China (Project No. 51975444) and the  Fundamental Research Funds for the Central Universities  (Project No. WUT225218002)

## References

[[1]](https://www.sciencedirect.com/science/article/pii/S1526612522002389#bbb0005)K.S. Prakash, T. Nancharaih, V.V.S. Rao. Additive manufacturing techniques in manufacturing-an overview. Mater Today Proc, 5 (2018), pp. 3873-3882, [10.1016/j.matpr.2017.11.642](https://doi.org/10.1016/j.matpr.2017.11.642)

[[2]](https://www.sciencedirect.com/science/article/pii/S1526612522002389#bbb0010)L. Yi, C. Gläßner, J.C. Aurich. How to integrate additive manufacturing technologies into manufacturing systems successfully: a perspective from the commercial vehicle industryJ Manuf Syst, 53 (2019), pp. 195-211, [10.1016/j.jmsy.2019.09.007](https://doi.org/10.1016/j.jmsy.2019.09.007)

[[3]](https://www.sciencedirect.com/science/article/pii/S1526612522002389#bbb0015)A. Gisario, M. Kazarian, F. Martina, M. Mehrpouya. Metal additive manufacturing in the commercial aviation industry: a review. J Manuf Syst, 53 (2019), pp. 124-149, [10.1016/j.jmsy.2019.08.005](https://doi.org/10.1016/j.jmsy.2019.08.005)

[[4]](https://www.sciencedirect.com/science/article/pii/S1526612522002389#bbb0020)H. Kim, Y. Lin, T.L.B. Tseng. A review on quality control in additive manufacturing. Rapid Prototyp J, 24 (2018), pp. 645-669, [10.1108/RPJ-03-2017-0048](https://doi.org/10.1108/RPJ-03-2017-0048)

[[5]](https://www.sciencedirect.com/science/article/pii/S1526612522002389#bbb0025)G. Li, K. Odum, C. Yau, M. Soshi, K. Yamazaki. High productivity fluence based control of directed energy deposition (DED) part geometry. J Manuf Process, 65 (2021), pp. 407-417, [10.1016/j.jmapro.2021.03.028](https://doi.org/10.1016/j.jmapro.2021.03.028)

[[6]](https://www.sciencedirect.com/science/article/pii/S1526612522002389#bbb0030) Tang Z. Jue, Liu W. Wei, Wang Y. Wen, K.M. Saleheen, Liu Z. Chao, Peng S. Tong, *et al.*A review on in situ monitoring technology for directed energy deposition of metals. Int J Adv Manuf Technol, 108 (2020), pp. 3437-3463, [10.1007/s00170-020-05569-3](https://doi.org/10.1007/s00170-020-05569-3)

[[7]](https://www.sciencedirect.com/science/article/pii/S1526612522002389#bbb0035)W. He, W. Shi, J. Li, H. Xie. In-situ monitoring and deformation characterization by optical techniques; part I: laser-aided direct metal deposition for additive manufacturing. Opt Lasers Eng, 122 (2019), pp. 74-88, [10.1016/j.optlaseng.2019.05.020](https://doi.org/10.1016/j.optlaseng.2019.05.020)

[[8]](https://www.sciencedirect.com/science/article/pii/S1526612522002389#bbb0150)W.Grace Guo, Q. Tian, S. Guo, Y. Guo. A physics-driven deep learning model for process-porosity causal relationship and porosity prediction with interpretability in laser metal deposition. CIRP Ann, 69 (2020), pp. 205-208, [10.1016/j.cirp.2020.04.049](https://doi.org/10.1016/j.cirp.2020.04.049)

[[9]](https://www.sciencedirect.com/science/article/pii/S1526612522002389#bbb0155)Y. Huang, M.B. Khamesee, E. Toyserkani. A new physics-based model for laser directed energy deposition (powder-fed additive manufacturing): from single-track to multi-track and multi-layer. Opt Laser Technol, 109 (2019), pp. 584-599, [10.1016/j.optlastec.2018.08.015](https://doi.org/10.1016/j.optlastec.2018.08.015)

[[10]](https://www.sciencedirect.com/science/article/pii/S1526612522002389#bbb0160)Z. Zhang, P. Ge, J.Y. Li, Y.F. Wang, X. Gao, X.X. Yao. Laser–particle interaction-based analysis of powder particle effects on temperatures and distortions in directed energy deposition additive manufacturing. J Therm Stress, 44 (2021), pp. 1068-1095, [10.1080/01495739.2021.1954572](https://doi.org/10.1080/01495739.2021.1954572)

[[11]](https://www.sciencedirect.com/science/article/pii/S1526612522002389#bbb0165)J.C. Heigel, P. Michaleris, E.W. Reutzel. Thermo-mechanical model development and validation of directed energy deposition additive manufacturing of Ti-6Al-4V. Addit Manuf. 5 (2015), pp. 9-19, [10.1016/j.addma.2014.10.003](https://doi.org/10.1016/j.addma.2014.10.003)

[[12]](https://www.sciencedirect.com/science/article/pii/S1526612522002389#bbb0170)K. Wang, C. Ma, Y. Qiao, X. Lu, W. Hao, S. Dong, A hybrid deep learning model with 1DCNN-LSTM-attention networks for short-term traffic flow prediction. Phys A Stat Mech Its Appl, 583 (2021), Article 126293, [10.1016/j.physa.2021.126293](https://doi.org/10.1016/j.physa.2021.126293)

[[13]](https://www.sciencedirect.com/science/article/pii/S1526612522002389#bbb0175)T.Y. Kim, S.B. Cho. Predicting residential energy consumption using CNN-LSTM neural networks
Energy, 182 (2019), pp. 72-81, [10.1016/j.energy.2019.05.230](https://doi.org/10.1016/j.energy.2019.05.230)

[[14]](https://www.sciencedirect.com/science/article/pii/S1526612522002389#bbb0180)A. Graves, J. Schmidhuber.Framewise phoneme classification with bidirectional LSTM and other neural network architectures. Neural Netw, 18 (2005), pp. 602-610, [10.1016/j.neunet.2005.06.042](https://doi.org/10.1016/j.neunet.2005.06.042)

## Cite

If you find this work useful in your project, please consider to cite it through:

```BibTex
@article{
	HU202232,
	title = {CNN-BiLSTM enabled prediction on molten pool width for thin-walled part fabrication using Laser 	Directed Energy Deposition},
	journal = {Journal of Manufacturing Processes},
	volume = {78},
	pages = {32-45},
	year = {2022},
	issn = {1526-6125},
	doi = {https://doi.org/10.1016/j.jmapro.2022.04.010},
	url = {https://www.sciencedirect.com/science/article/pii/S1526612522002389},
	author = {Kaixiong Hu and Yanghui Wang and Weidong Li and Lihui Wang},
	keywords = {Laser Directed Energy Deposition (LDED), Molten pool width, Data driven approach, Additive 		manufacturing (AM)}
}
```

