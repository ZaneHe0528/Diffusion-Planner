###### Abstract

We present BLUE, a minimal method for better language use in vision-language-action (VLA) models for autonomous driving (AD). Through extensive analysis, we reveal that language matters on only a small fraction of routes, but on those routes it can greatly improve or degrade performance. Generating language at every frame is therefore inefficient, since most computation is spent on frames that do not benefit from language. We further show that pretrained VLA hidden states potentially already encode whether language will benefit a given frame, even though scene complexity and kinematic features alone struggle to predict this. Based on this finding, BLUE trains a lightweight gate on frozen VLA hidden states to decide per frame whether to activate language generation or predict actions directly, without modifying the backbone or requiring additional human annotation. With just a 0.11M-parameter gate, BLUE sets a new state of the art on both benchmarks, achieving 76.2% success rate on Bench2Drive and 36 driving score on Longest6 v2, while delivering 2.54 $\times$ inference speedup and 8.9% success rate improvement over the backbone. BLUE provides a practical path toward efficient language-augmented AD, showing that VLA models can retain the benefits of language at a fraction of the cost. Our code, data, logs and checkpoints are fully available on [Github](https://github.com/George-Ling3/BLUE).

BLUE: Toward Better Language Use in Efficient  
Vision-Language-Action Models for Autonomous Driving

George Ling, Lijin Yang, Hao Yang, Zhongzhan Huang <sup><math xmlns="http://www.w3.org/1998/Math/MathML" display="inline" data-latex="\dagger"><semantics><mo>†</mo> <annotation>\dagger</annotation></semantics></math></sup> Bosch Research   <sup><math xmlns="http://www.w3.org/1998/Math/MathML" display="inline" data-latex="\dagger"><semantics><mo>†</mo> <annotation>\dagger</annotation></semantics></math></sup> Correspondence to: [hru4sgh@bosch.com](https://arxiv.org/html/2606.08684v1/mailto:hru4sgh@bosch.com) [Code](https://github.com/George-Ling3/BLUE)   [Hugging Face (Data/Logs/Ckpts)](https://huggingface.co/George-Ling/blue_gate)   [Homepage](https://blue-website.github.io/)

## 1 Introduction

Recent vision-language-action (VLA) models for autonomous driving typically generate natural language to reason about the scene before predicting driving actions [^60] [^80]. However, the impact of generated language on closed-loop driving has rarely been systematically quantified. First, we conduct $\sim$ 2000 GPU hours of closed-loop analysis on full Bench2Drive [^31] using SimLingo [^60], a representative VLA driving model, running each route through repeated experiments and categorizing driving outcomes via statistical tests. As Figure 1 shows, the generated language statistically improves driving on only 14.5% of routes, actively degrades it on 23.6%, and has no clear effect on the remaining majority. Yet current many VLA driving models [^60] [^80] [^17] usually generate language at every frame by default, wasting computation on frames that do not benefit and compromising both driving performance and inference efficiency. See more analysis on extra settings in Appendix D.2.

![Refer to caption](https://arxiv.org/html/2606.08684v1/x1.png)

Figure 1: Top: Distribution of Bench2Drive routes by language impact. Bottom: Inference time breakdown comparing the original VLA and BLUE, which reduces unnecessary language generation for 2.54 × \\times speedup. Extended details and results are in Appendix D.2.

![Refer to caption](https://arxiv.org/html/2606.08684v1/x2.png)

Figure 2: Per-scenario distribution of language effects across all 44 Bench2Drive scenario categories. Each bar represents one scenario, showing the proportion of routes where language generation is helpful, neutral, or harmful to driving success. Extended details and results under additional settings are in Appendix D.2.

Since language significantly helps or hurts driving performance on only a minority of routes at substantial inference cost, an intuitive strategy is to detect per frame whether language generation is needed and skip it otherwise, thereby improving both driving performance and inference speed. Fortunately, we find that pretrained VLA hidden states potentially already encode this language-utility signal, even though scene complexity and kinematic features alone struggle to predict it. In Section 3, we leverage this finding and propose BLUE, a minimal method that trains a lightweight gate on frozen hidden states to decide per frame whether to activate language generation or predict actions directly. BLUE requires no backbone modification and no additional human annotation, as training labels are naturally derived from routine driving evaluation. In Section 4, we show that with just a 0.11M-parameter gate, BLUE sets new state of the art on two benchmarks, achieving 76.2% success rate on the multi-scenario Bench2Drive and 36 driving score on the long-horizon Longest6 v2, while delivering 2.54 $\times$ inference speedup and a 8.9% success rate improvement over the VLA backbone. We discuss related work in Appendix B and summarize our contributions as follows:

- We provide a systematic analysis of when language helps and when it hurts driving, showing that on-demand language use can improve both driving performance and inference speed.
- We reveal that pretrained VLA hidden states potentially already encode the language-utility signal. With just a 0.11M-parameter gate on frozen hidden states, BLUE achieves state-of-the-art performance on Bench2Drive and Longest6 v2, while delivering 2.54 $\times$ inference speedup.

## 2 Observations and Motivations

We analyze how generated language affects driving performance across different scenarios and quantify the potential gains from selective language use.

![Refer to caption](https://arxiv.org/html/2606.08684v1/x3.png)

Figure 3: Overview of BLUE. Top: a lightweight gate receives the hidden state from the frozen VLA backbone and decides per frame whether to activate language generation or directly output waypoints. Bottom: gate training pipeline. Labels are derived from route-level success rate comparisons and refined via frame-level reconstruction. The visual encoder and LLM backbone remain frozen, while only a lightweight MLP gate is trainable.

##### Setup.

We study SimLingo [^60], a VLA driving model that generates natural language reasoning before predicting actions. By skipping language generation, the model can also predict actions directly from its internal representations. We evaluate both configurations on all 44 scenario categories of Bench2Drive [^31], running repeated experiments per route. Figure 2 shows the per-scenario results. We provide analysis under additional settings in Appendix D.2.

##### Language Does Not Always Help.

As shown in Figure 2, language generation only matters on a small fraction of routes, but where it matters, the impact on driving performance is substantial, either improving or degrading it by a large margin. On the majority of routes, language has no measurable effect yet still incurs full generation cost. Intuitively, if we can detect when language is needed and skip it otherwise, the model could achieve both better driving performance and faster inference.

##### Room for Improvement.

To quantify the potential gains, we construct a route-level oracle that picks the better-performing configuration for each route. Even with this coarse-grained selection, the oracle already reaches 78.4% success rate, revealing more than 10% room for improvement over the default VLA. Since the oracle only makes a single choice per route while finer per-frame selection could further improve performance, this estimate is conservative. The large gap confirms that a lightweight selection mechanism can unlock substantial performance gains without any model retraining, motivating the gate design in Section 3.

## 3 Method

In this section, we present BLUE, which uses pretrained VLA hidden states to predict per frame whether to activate language generation. Figure 3 shows the overall framework.

##### Hidden States Encode Language Necessity.

To predict when language generation is needed, we look for a signal within the model itself. We find that a simple logistic regression trained on the VLA’s last-layer hidden states can distinguish frames where language helps from those where it does not, without relying on any external features. This shows that the pretrained hidden states potentially already encode a language-utility signal, providing a basis for building a lightweight gate.

##### Data Collection and Labeling.

We run both language mode and direct action mode on training routes thought repeated experiments, collecting the last-layer hidden state $\mathbf{h}\in\mathbb{R}^{d}$ at the final token position for each frame. Details of data splits are provided in Appendix C.1, and additional labeling details are provided in Appendix C.4. This position corresponds to the model’s representation right before language or waypoint generation begins, and is shared by both modes to ensure feature alignment. See more details of data splits and labeling in Appendix C.4.2. Each route $r$ is first assigned a binary label based on the success rate gap:

$$
y_{r}=\mathbbm{1}\!\Big[\frac{1}{|\mathcal{S}|}\sum_{s\in\mathcal{S}}\big(\mathrm{SR}_{\text{lang}}^{(r,s)}-\mathrm{SR}_{\text{direct}}^{(r,s)}\big)>\tau\Big],
$$

where $\mathcal{S}$ is the set of random seeds, $\mathrm{SR}$ denotes success rate, and $\tau{=}10\%$ is a margin threshold. This design defaults to the faster direct action mode unless language shows a clear advantage.

<table><tbody><tr><td rowspan="2">Method</td><td colspan="5">Details</td><td colspan="2">Metrics</td></tr><tr><td>Expert</td><td>Camera</td><td>LiDAR</td><td>Labels</td><td>T-Param.</td><td>SR (%) <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>DS <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td></tr><tr><td>UniAD-Base <sup><a href="#fn:24">24</a></sup></td><td>Think2Drive</td><td>6 <math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math></td><td></td><td>O,M,S</td><td><math><semantics><mo>≥</mo> <annotation>\geq</annotation></semantics></math>  59 M</td><td>16.36</td><td>45.81</td></tr><tr><td>TF++ <sup><a href="#fn:93">93</a></sup></td><td>PDM-Lite</td><td>1 <math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math></td><td></td><td>O,M,S,D</td><td><math><semantics><mo>≥</mo> <annotation>\geq</annotation></semantics></math>  39 M</td><td>67.27</td><td>84.21</td></tr><tr><td>MomAD <sup><a href="#fn:68">68</a></sup></td><td>Think2Drive</td><td>6 <math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math></td><td></td><td>O,M</td><td><math><semantics><mo>≥</mo> <annotation>\geq</annotation></semantics></math>  25 M</td><td>18.11</td><td>47.91</td></tr><tr><td>DriveTrans <sup><a href="#fn:32">32</a></sup></td><td>Think2Drive</td><td>6 <math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math></td><td></td><td>O,M</td><td><math><semantics><mo>≈</mo> <annotation>\approx</annotation></semantics></math>  646 M</td><td>35.01</td><td>63.46</td></tr><tr><td>Hydra-NeXt <sup><a href="#fn:42">42</a></sup></td><td>Think2Drive</td><td>2 <math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math></td><td></td><td>-</td><td><math><semantics><mo>≥</mo> <annotation>\geq</annotation></semantics></math>  25 M</td><td>50.00</td><td>73.86</td></tr><tr><td>DiffusionDrive <sup><a href="#fn:44">44</a></sup></td><td>-</td><td>3 <math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math></td><td></td><td>O,S</td><td><math><semantics><mo>≈</mo> <annotation>\approx</annotation></semantics></math>  60 M</td><td>52.72</td><td>77.68</td></tr><tr><td>ORION <sup><a href="#fn:15">15</a></sup></td><td>Think2Drive</td><td>6 <math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math></td><td></td><td>O,M,L</td><td><math><semantics><mo>≥</mo> <annotation>\geq</annotation></semantics></math>  300 M</td><td>54.62</td><td>77.74</td></tr><tr><td>AutoVLA <sup><a href="#fn:92">92</a></sup></td><td>PDM-Lite</td><td>1 <math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math></td><td></td><td>L</td><td><math><semantics><mo>≥</mo> <annotation>\geq</annotation></semantics></math>  1.5 B</td><td>57.73</td><td>78.84</td></tr><tr><td>SimLingo <sup><a href="#fn:60">60</a></sup></td><td>PDM-Lite</td><td>1 <math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math></td><td></td><td>L</td><td><math><semantics><mo>≥</mo> <annotation>\geq</annotation></semantics></math>  300 M</td><td>67.27</td><td>85.07</td></tr><tr><td>HiP-AD <sup><a href="#fn:71">71</a></sup></td><td>Think2Drive</td><td>6 <math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math></td><td></td><td>O,M,D</td><td><math><semantics><mo>≈</mo> <annotation>\approx</annotation></semantics></math>  97 M</td><td>69.09</td><td>86.77</td></tr><tr><td>ReCogDrive <sup><a href="#fn:39">39</a></sup></td><td>Think2Drive</td><td>6 <math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math></td><td></td><td>L</td><td><math><semantics><mo>≥</mo> <annotation>\geq</annotation></semantics></math>  2 B</td><td>45.45</td><td>71.36</td></tr><tr><td>GeRo <sup><a href="#fn:82">82</a></sup></td><td>Think2Drive</td><td>6 <math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math></td><td></td><td>O,M,L</td><td><math><semantics><mo>≥</mo> <annotation>\geq</annotation></semantics></math>  3 B</td><td>60.10</td><td>81.90</td></tr><tr><td>DeLL <sup><a href="#fn:13">13</a></sup></td><td>Think2Drive</td><td>1 <math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math></td><td></td><td>O,S</td><td><math><semantics><mo>≥</mo> <annotation>\geq</annotation></semantics></math>  38 M</td><td>68.63</td><td>86.86</td></tr><tr><td>R2SE <sup><a href="#fn:47">47</a></sup></td><td>PDM-Lite</td><td>1 <math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math></td><td></td><td>O,M,S,D</td><td><math><semantics><mo>≥</mo> <annotation>\geq</annotation></semantics></math>  39 M</td><td>69.54</td><td>86.28</td></tr><tr><td>AutoMoT <sup><a href="#fn:25">25</a></sup></td><td>PDM-Lite</td><td>1 <math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math></td><td></td><td>-</td><td><math><semantics><mo>≈</mo> <annotation>\approx</annotation></semantics></math>  1.6 B</td><td>70.00</td><td>87.34</td></tr><tr><td>BevAD <sup><a href="#fn:22">22</a></sup></td><td>PDM-Lite</td><td>6 <math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math></td><td></td><td>O</td><td><math><semantics><mo>≥</mo> <annotation>\geq</annotation></semantics></math>  25 M</td><td>72.73</td><td>88.11</td></tr><tr><td>CriticVLA <sup><a href="#fn:80">80</a></sup></td><td>PDM-Lite</td><td>1 <math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math></td><td></td><td>L</td><td><math><semantics><mo>≥</mo> <annotation>\geq</annotation></semantics></math>  300 M</td><td>73.33</td><td>88.02</td></tr><tr><td>TakeVLA <sup><a href="#fn:17">17</a></sup></td><td>PDM-Lite</td><td>1 <math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math></td><td></td><td>L</td><td><math><semantics><mo>≥</mo> <annotation>\geq</annotation></semantics></math>  300 M</td><td>73.73</td><td>89.72</td></tr><tr><td>bbb!15BLUE (Ours)</td><td>PDM-Lite</td><td>1 <math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math></td><td></td><td>L</td><td>0.11 M</td><td>76.18 <math><semantics><mo>±</mo> <annotation>\pm</annotation></semantics></math> 0.64</td><td>90.58 <math><semantics><mo>±</mo> <annotation>\pm</annotation></semantics></math> 0.12</td></tr><tr><td><math><semantics><mi>Δ</mi> <annotation>\Delta</annotation></semantics></math> vs. SimLingo</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>+8.91</td><td>+5.51</td></tr></tbody></table>

Table 1: Results on Bench2Drive. BLUE achieves the highest closed-loop success rate (SR) and driving score (DS), with large margins over its SimLingo backbone. T-Param. reports trainable parameters; we use published values ($\approx$) where available and conservative lower bounds ($\geq$) derived from the minimum size of trained components. BLUE trains only a 0.11M gate while keeping the VLA backbone frozen. Notably, BLUE surpasses methods that employ multi-camera setups, LiDAR, or dense auxiliary labels (O: 3D object detection, M: map, S: semantic segmentation, D: depth, L: language), using only front-view camera with language annotations. See more results in Appendix D.1.

We construct training labels at two granularities. In route-level labeling, every frame on route $r$ shares the same label $y_{r}$. In frame-level labeling, we further refine routes that benefit from language ($y_{r}{=}1$) by marking only the critical regions $\mathcal{C}_{r}$ where language mode and direct action mode differ the most. We identify these regions from spatial patterns of behavior divergence. See Appendix C.4.2 for details. The frame-level label is:

$$
y_{r,t}=\mathbbm{1}\!\big[\Delta\overline{\mathrm{SR}}_{r}>\tau\big]\cdot\mathbbm{1}\!\big[\mathbf{x}_{t}\in\mathcal{C}_{r}\big],
$$

where $\Delta\overline{\mathrm{SR}}_{r}{=}\frac{1}{|\mathcal{S}|}\sum_{s}(\mathrm{SR}_{\text{lang}}^{(r,s)}{-}\mathrm{SR}_{\text{direct}}^{(r,s)})$ is the cross-seed language advantage of route $r$, and $\mathbf{x}_{t}{\in}\mathbb{R}^{2}$ is the spatial coordinate of frame $t$. The first indicator selects language-beneficial routes, while the second restricts positive labels to critical segments within those routes. We mix samples from both labeling granularities during training so that the gate learns both route-level preference and frame-level activation. To address temporal redundancy (e.g., near-identical features when the vehicle is stopped), we group consecutive frames with cosine similarity above $0.99$ into redundant segments and downsample each segment of length $L$ to $\max(2,\lceil\sqrt{L}\rceil)$ representative samples.

##### Gate Design.

The gate is a single-hidden-layer MLP with negligible parameter count, trained with binary cross-entropy loss. This small design is sufficient for mapping pretrained hidden states to a binary gating decision, while reducing overfitting risk and keeping inference overhead negligible.

##### On-demand language use.

The trained gate is integrated into the VLA inference pipeline with negligible overhead. At each frame, the model computes $\mathbf{h}$ through a forward pass shared by both modes. The gate outputs a score $p(\mathbf{h})$: if it exceeds a threshold $\theta$, the model proceeds with language generation; otherwise, it produces actions directly. The threshold $\theta$ governs how selectively the gate triggers language generation at inference time.

<table><tbody><tr><td rowspan="2">Method</td><td colspan="2">Details</td><td colspan="6">Multi-Ability Success Rate (%)</td></tr><tr><td>Camera</td><td>LiDAR</td><td>Merge <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>Overtake <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>EmBrake <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>GiveWay <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>TSign <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>Mean <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td></tr><tr><td>UniAD-Base <sup><a href="#fn:24">24</a></sup></td><td>6 <math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math></td><td></td><td>14.10</td><td>17.78</td><td>21.67</td><td>10.00</td><td>14.21</td><td>15.55</td></tr><tr><td>TF++ <sup><a href="#fn:93">93</a></sup></td><td>1 <math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math></td><td></td><td>58.75</td><td>57.77</td><td>83.33</td><td>40.00</td><td>82.11</td><td>64.39</td></tr><tr><td>DriveTrans <sup><a href="#fn:32">32</a></sup></td><td>6 <math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math></td><td></td><td>17.57</td><td>35.00</td><td>48.36</td><td>40.00</td><td>52.10</td><td>38.60</td></tr><tr><td>Hydra-NeXt <sup><a href="#fn:42">42</a></sup></td><td>2 <math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math></td><td></td><td>40.00</td><td>64.44</td><td>61.67</td><td>50.00</td><td>50.00</td><td>53.22</td></tr><tr><td>DiffusionDrive <sup><a href="#fn:44">44</a></sup></td><td>3 <math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math></td><td></td><td>50.63</td><td>26.67</td><td>68.33</td><td>50.00</td><td>76.32</td><td>54.38</td></tr><tr><td>ORION <sup><a href="#fn:15">15</a></sup></td><td>6 <math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math></td><td></td><td>25.00</td><td>71.11</td><td>78.33</td><td>30.00</td><td>69.15</td><td>54.72</td></tr><tr><td>HiP-AD <sup><a href="#fn:71">71</a></sup></td><td>6 <math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math></td><td></td><td>50.00</td><td>84.44</td><td>83.33</td><td>40.00</td><td>72.10</td><td>65.98</td></tr><tr><td>SimLingo <sup><a href="#fn:60">60</a></sup></td><td>1 <math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math></td><td></td><td>53.78</td><td>67.41</td><td>81.67</td><td>50.00</td><td>77.20</td><td>66.01</td></tr><tr><td>ReCogDrive <sup><a href="#fn:39">39</a></sup></td><td>6 <math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math></td><td></td><td>29.73</td><td>20.00</td><td>69.09</td><td>20.00</td><td>71.34</td><td>42.03</td></tr><tr><td>GeRo <sup><a href="#fn:82">82</a></sup></td><td>6 <math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math></td><td></td><td>40.06</td><td>78.24</td><td>87.32</td><td>50.00</td><td>76.83</td><td>66.49</td></tr><tr><td>R2SE <sup><a href="#fn:47">47</a></sup></td><td>1 <math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math></td><td></td><td>53.33</td><td>61.25</td><td>90.00</td><td>50.00</td><td>84.21</td><td>67.76</td></tr><tr><td>DeLL <sup><a href="#fn:13">13</a></sup></td><td>1 <math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math></td><td></td><td>61.25</td><td>62.22</td><td>80.00</td><td>60.00</td><td>81.05</td><td>68.90</td></tr><tr><td>TakeVLA <sup><a href="#fn:17">17</a></sup></td><td>1 <math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math></td><td></td><td>63.64</td><td>64.44</td><td>91.67</td><td>50.00</td><td>85.48</td><td>71.05</td></tr><tr><td>CriticVLA <sup><a href="#fn:80">80</a></sup></td><td>1 <math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math></td><td></td><td>61.28</td><td>76.30</td><td>88.33</td><td>50.00</td><td>81.06</td><td>71.39</td></tr><tr><td>BevAD <sup><a href="#fn:22">22</a></sup></td><td>6 <math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math></td><td></td><td>71.67</td><td>74.07</td><td>75.56</td><td>76.67</td><td>75.44</td><td>74.68</td></tr><tr><td>bbb!15BLUE (ours)</td><td>1 <math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math></td><td></td><td>61.44 <math><semantics><mo>±</mo> <annotation>\pm</annotation></semantics></math> 1.33</td><td>80.00 <math><semantics><mo>±</mo> <annotation>\pm</annotation></semantics></math> 1.81</td><td>93.27 <math><semantics><mo>±</mo> <annotation>\pm</annotation></semantics></math> 1.33</td><td>50.00 <math><semantics><mo>±</mo> <annotation>\pm</annotation></semantics></math> 0.00</td><td>84.74 <math><semantics><mo>±</mo> <annotation>\pm</annotation></semantics></math> 0.00</td><td>73.89 <math><semantics><mo>±</mo> <annotation>\pm</annotation></semantics></math> 0.14</td></tr><tr><td><math><semantics><mi>Δ</mi> <annotation>\Delta</annotation></semantics></math> vs. SimLingo</td><td>-</td><td>-</td><td>+7.66</td><td>+12.59</td><td>+11.60</td><td>+0.00</td><td>+7.54</td><td>+7.88</td></tr></tbody></table>

Table 2: Multi-ability results on Bench2Drive. Mean denotes the average success rate over the five driving skills. While using only a single front-view camera and no LiDAR, BLUE achieves the second-best mean result and remains close to the best method, which uses six cameras. See additional results in Appendix D.

## 4 Experiments

In this section, we evaluate BLUE on two established closed-loop driving benchmarks and compare against a wide range of published methods.

##### Setup.

We apply BLUE to SimLingo [^60] with a gate threshold of $\theta{=}0.66$. The gate uses a hidden dimension of 128 and is trained on approximately 400 routes sampled from SimLingo’s training set. We evaluate on two benchmarks: Bench2Drive [^31], a multi-scenario benchmark covering 220 routes across 44 scenario categories in CARLA [^12], and Longest6 v2 [^3], a long-horizon benchmark that evaluates sustained driving quality through driving score, route completion, and infraction score. We compare against 26 published methods spanning end-to-end and VLA approaches. All BLUE results are averaged over 3 random seeds. More details on data splits, benchmarks, baselines, implementation, and additional results are provided in Appendix C and D.

##### Closed-Loop Results on Bench2Drive.

Table 1 presents the main comparison on Bench2Drive. BLUE achieves the highest success rate and driving score among all compared methods, improving both metrics over its backbone SimLingo by a large margin. Notably, BLUE trains only a 0.11M-parameter gate on a frozen backbone, uses a single front-view camera without LiDAR, and requires only language annotations. Despite this minimal setup, it surpasses methods that employ six cameras, LiDAR, dense auxiliary labels, or orders-of-magnitude more trainable parameters. Table 2 further reports the multi-ability breakdown. BLUE ranks second in mean ability score, close to the best method that relies on six cameras, with clear improvements in overtaking and emergency braking.

##### Closed-Loop Results on Longest6 v2.

Table 3 presents results on Longest6 v2, a long-horizon benchmark that evaluates sustained driving quality. BLUE achieves the highest driving score and route completion among all compared methods, while requiring substantially fewer GPU hours. The improvement in route completion suggests that our BLUE helps maintain robust driving over long distances, where errors from unnecessary language generation would otherwise compound.

| Method | DS $\uparrow$ | RC $\uparrow$ | IS $\uparrow$ | Time $\downarrow$ |
| --- | --- | --- | --- | --- |
| HiP-AD [^71] | 7 | 56 | \- | \- |
| TF++ [^93] | 23 | 71 | \- | \- |
| SimLingo [^60] | 22 | 70 | 0.38 | 119h |
| CriticVLA [^80] | 34 | 66 | 0.55 | 193h |
| bbb!15BLUE (ours) | 36 | 84 | 0.43 | 56h |
| $\Delta$ vs. SimLingo | +14 | +14 | +0.05 | \-63h |

Table 3: Closed-loop results on Longest6 v2. DS: driving score, RC: route completion, IS: infraction score. Time: total A100 GPU hours to evaluate all routes.

##### Inference Efficiency.

Since the gate skips language generation on most frames, BLUE runs substantially faster than the full language mode. The gate itself adds negligible overhead, as it is a single-hidden-layer MLP applied to the already-computed hidden state. As shown in Table 4, BLUE achieves a 2.54 $\times$ speedup over SimLingo with the lowest latency among all compared methods. We provide additional efficiency analysis in the next section.

| Method | Speed Ratio $\uparrow$ | FPS $\uparrow$ | Latency (ms) $\downarrow$ |
| --- | --- | --- | --- |
| HiP-AD | 0.0625 | 1.25 | 800.3 |
| SimLingo | 0.0358 | 0.72 | 1396.6 |
| CriticVLA | 0.0146 | 0.29 | 3424.7 |
| bbb!15BLUE (ours) | 0.0910 | 1.82 | 549.5 |
| $\Delta$ vs. SimLingo | +154.2% | +154.2% | \-60.7% |

Table 4: Inference efficiency comparison among representative driving models. Higher speed ratio and FPS are better, while lower latency is better.

## 5 Analysis and Ablation Study

We now analyze the language gate from multiple angles: its activation behavior, generalizability to other models, and sensitivity to design choices.

(1) What activation pattern does the gate learn?

We visualize the gate’s frame-level decisions across evaluation routes in Figure 4. The gate skips language generation on most frames, yet BLUE still outperforms the VLA backbone by a large margin, as shown in Section 4. When it does activate language, the activations form contiguous segments rather than scattering across isolated frames, suggesting that the gate captures temporally coherent patterns from the hidden states rather than producing noisy frame-level fluctuations.

![Refer to caption](https://arxiv.org/html/2606.08684v1/x4.png)

Figure 4: Language activation pattern learned by the gate. Each column is one route, grouped by the route-level language benefit measured in Section 2. The color encodes the fraction of frames where language generation is activated at each progress bin. The gate activates language sparingly and in contiguous segments.

(2) Can BLUE be applied to other models?

The main experiments use SimLingo as the backbone. To examine whether BLUE transfers to other architectures, we apply the same pipeline to CriticVLA [^80], a VLA model with a different language integration design. As shown in Table 5, BLUE improves CriticVLA across all metrics on both Bench2Drive and Longest6 v2, with particularly notable gains in route completion on Longest6 v2. The resulting system also surpasses all other listed baselines. This suggests that the language-utility signal in hidden states is not unique to SimLingo, and that the gate mechanism can benefit other language-centric VLA models. Full comparison results are in Appendix D.

<table><tbody><tr><td rowspan="2">Method</td><td colspan="2">Bench2Drive</td><td colspan="2">Longest6 v2</td></tr><tr><td>SR (%) <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>DS <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>DS <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>RC <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td></tr><tr><td>TF++ <sup><a href="#fn:93">93</a></sup></td><td>67.27</td><td>84.21</td><td>23</td><td>71</td></tr><tr><td>HiP-AD <sup><a href="#fn:71">71</a></sup></td><td>69.09</td><td>86.77</td><td>7</td><td>56</td></tr><tr><td>SimLingo <sup><a href="#fn:60">60</a></sup></td><td>67.27</td><td>85.07</td><td>22</td><td>70</td></tr><tr><td>CriticVLA <sup><a href="#fn:80">80</a></sup></td><td>73.33</td><td>88.02</td><td>34</td><td>66</td></tr><tr><td>bbb!15BLUE (CriticVLA)</td><td>76.04</td><td>90.37</td><td>36.2</td><td>80.6</td></tr><tr><td><math><semantics><mi>Δ</mi> <annotation>\Delta</annotation></semantics></math> vs. CriticVLA</td><td>+2.71</td><td>+2.35</td><td>+2.2</td><td>+14.6</td></tr></tbody></table>

Table 5: BLUE applied to CriticVLA, compared with representative methods on Bench2Drive and Longest. Full comparison results are in Appendix D.

(3) Is the gate transferable across models?

Since BLUE trains a separate gate for each model, a natural question is whether a gate learned on one model can be directly reused on another without retraining. We test this by swapping the trained gates between SimLingo and CriticVLA. As shown in Table 6, the matched gate consistently outperforms the transferred gate by a clear margin. This indicates that when language helps is inherently tied to the model itself rather than to the driving scenario. Different models internalize language utility in distinct ways within their hidden states, so each model should train its own gate. Appendix D.2 further analyzes why each model requires its own gate and its retraining cost.

| Configuration | SR (%) $\uparrow$ | DS $\uparrow$ |
| --- | --- | --- |
| SimLingo [^60] | 67.27 $\pm$ 2.11 | 85.07 $\pm$ 0.95 |
| SimLingo + CriticVLA gate | 71.59 $\pm$ 0.96 | 89.23 $\pm$ 0.21 |
| SimLingo + SimLingo gate | 76.18 $\pm$ 0.64 | 90.58 $\pm$ 0.12 |
| CriticVLA [^80] | 73.33 $\pm$ 0.27 | 88.02 $\pm$ 0.17 |
| CriticVLA + SimLingo gate | 73.11 $\pm$ 1.36 | 88.90 $\pm$ 0.53 |
| CriticVLA + CriticVLA gate | 76.04 $\pm$ 0.38 | 90.37 $\pm$ 0.14 |

Table 6: Cross-model transfer of the learned language gate on Bench2Drive. Each transferred gate is evaluated on a model different from the one used for training.

(4) Are rule-based methods sufficient for predicting when language is needed?

We ask whether simple driving signals can replace the learned gate. We design three rule-based gates that activate language when a kinematic feature exceeds a threshold: vehicle speed, acceleration magnitude, or steering angle. See Appendix C.6 for construction details. As shown in Table 7, regardless of which feature or threshold is used, all rule-based gates fall far short of BLUE and offer only marginal gains over single-mode baselines. The random gate performs even worse. This result is expected: kinematic features reflect only the vehicle’s current motion and do not provide enough information to predict whether language will help. The VLA’s hidden states, which jointly encode perceptual and contextual information, are a much stronger signal for this prediction.

| Gate Method | SR (%) $\uparrow$ | DS $\uparrow$ | Lang. (%) |
| --- | --- | --- | --- |
| Speed-based gate | 70.97 $\pm$ 1.21 | 88.71 $\pm$ 0.63 | 55.50 $\pm$ 1.21 |
| Speed-based gate | 71.81 $\pm$ 0.54 | 89.91 $\pm$ 0.88 | 30.21 $\pm$ 0.91 |
| Acceleration-based gate | 70.08 $\pm$ 0.43 | 88.33 $\pm$ 0.43 | 49.12 $\pm$ 0.87 |
| Steering-based gate | 70.71 $\pm$ 0.97 | 89.38 $\pm$ 0.22 | 7.94 $\pm$ 0.02 |
| Complexity-based gate | 70.98 $\pm$ 0.80 | 87.12 $\pm$ 1.75 | 53.61 $\pm$ 1.08 |
| Complexity-based gate | 71.40 $\pm$ 2.26 | 88.02 $\pm$ 0.85 | 17.15 $\pm$ 0.58 |
| Random gate | 67.42 $\pm$ 0.01 | 86.66 $\pm$ 1.18 | 79.90 $\pm$ 0.16 |
| Random gate | 70.96 $\pm$ 0.71 | 88.36 $\pm$ 0.16 | 50.07 $\pm$ 0.41 |
| Random gate | 70.01 $\pm$ 1.53 | 87.10 $\pm$ 1.58 | 20.15 $\pm$ 0.32 |
| bbb!15BLUE (ours) | 76.18 $\pm$ 0.64 | 90.58 $\pm$ 0.12 | 21.44 $\pm$ 0.66 |

Table 7: Comparison of alternative gating strategies on Bench2Drive. Kinematic gates activate language when a motion feature exceeds a threshold; the complexity gate activates language on complex routes. Lang. denotes the percentage of frames with language activation.

(5) Is scenario complexity sufficient for predicting when language is needed?

An intuitive hypothesis is that language generation is needed in complex scenarios and can be skipped in simple ones. To test this, we compute a composite complexity score for each training route based on structured features including the number of sub-scenarios, weather severity, traffic flow density, and opposing vehicle interactions. Routes above a threshold are labeled complex and activate language generation, while the rest skip it and predict actions directly. See Appendix C.6 for more details. As shown in Table 7, the complexity-based gate performs comparably to the kinematic-based gates and remains far below BLUE. This indicates that whether language generation helps is not determined by scenario complexity alone, and frame-level hidden states capture dynamics that coarse route-level labels cannot provide. We discuss why complexity-based gate fall short in Appendix D.2.

(6) Does BLUE improve inference efficiency?

Beyond driving performance, BLUE also improves inference efficiency. Since the gate skips language generation on most frames and the gate itself is a single-hidden-layer MLP with negligible overhead, BLUE reduces the per-frame latency substantially. Table 4 reports the aggregate statistics: BLUE on SimLingo achieves 2.54 $\times$ speedup and reduces mean latency from 1.40 s to 0.55 s. The gain extends to CriticVLA, where BLUE achieves 4.50 $\times$ speedup and lowers the mean latency from 3.42 s to 0.76 s. Figure 5 further shows the per-route latency distributions across all evaluation routes. For both backbones, BLUE shifts the entire distribution toward lower latency. These results confirm that BLUE simultaneously improves both driving quality and inference efficiency.

![Refer to caption](https://arxiv.org/html/2606.08684v1/x5.png)

Figure 5: Distribution of mean per-frame latency across Bench2Drive evaluation routes. Dashed lines indicate overall means. BLUE substantially reduces latency for both backbones, delivering 2.54 × \\times and 4.50 speedup.

(7) How sensitive is the gate to the threshold?

The gate threshold $\theta$ controls how readily the model activates language generation. We select $\theta$ without tuning based on a simple observation: the effect of language on each route falls into three natural categories, helpful, neutral, and harmful, which partition the gate output range $[0,1]$ into three equal intervals. We place $\theta$ at the boundary between neutral and helpful, yielding $\theta{=}0.66$ directly from this categorization. Figure 6 shows the effect of varying $\theta$ across the full range. Language activation ratio decreases monotonically as $\theta$ increases, confirming that the gate produces well-calibrated scores. SR peaks near our chosen value, and thresholds from 0.6 to 0.8 all achieve good results. When $\theta$ is very low, the model generates language reasoning on nearly every frame before acting, and SR reduces to 66.91%. When $\theta$ is very high, the model skips language and produces actions directly at every frame, yielding SR = 69.55%. BLUE at $\theta{=}0.66$ achieves 76.18% SR, surpassing these two settings by a large margin.

(8) How does training data size affect the gate?

We collect approximately 400 training routes from the SimLingo training set, vary the proportion used from 10% to 100%, and evaluate closed-loop driving on Bench2Drive. Training and evaluation routes have no overlap. We detail the data splits in Appendix C.1. As shown in Figure 7, both SR and DS improve steadily as training data increases. With only half of the available routes, the gate already surpasses the SimLingo backbone by a clear margin, indicating that the language-utility signal encoded in hidden states is learnable from moderate amounts of data. Performance continues to improve with additional training routes, and variance across seeds decreases, reflecting more stable gate decisions. We use the full training set in all other experiments for best results.

(9) How does gate design affect performance?

We vary the gate hidden dimension and dropout setting while fixing all other factors. As shown in Table 8, all configurations achieve strong performance, and increasing gate capacity does not bring clear improvement. Among all variants, dropout provides a noticeable benefit. We adopt the smaller gate with dropout as the default, since a smaller capacity combined with regularization better prevents overfitting to training routes. This choice is made purely based on the principle of minimizing overfitting risk, not from test-set tuning.

![Refer to caption](https://arxiv.org/html/2606.08684v1/x6.png)

Figure 6: Effect of threshold θ \\theta on language activation ratio and success rate. Language usage decreases monotonically with, while ∈ \[ 0.6, 0.8 \] \\theta\\in\[0.6,0.8\] yields strong SR.

| Hidden Dim | Dropout | Success Rate $\uparrow$ | Driving Score $\uparrow$ |
| --- | --- | --- | --- |
| 128 | No | 74.67 $\pm$ 0.69 | 90.25 $\pm$ 0.20 |
| bbb!15128 | Yes | 76.18 $\pm$ 0.64 | 90.58 $\pm$ 0.12 |
| 256 | No | 74.62 $\pm$ 0.33 | 89.97 $\pm$ 0.36 |
| 256 | Yes | 75.19 $\pm$ 1.56 | 89.70 $\pm$ 1.15 |

Table 8: Effect of gate hidden dimension and dropout on Bench2Drive. All settings use the same training data, labels, and inference threshold.

(10) Is the impact of language use consistent across experimental settings?

In Section 2, we observed a performance gap and complementarity between driving with and without language generation on SimLingo. Here we test whether this pattern persists across different experimental settings by varying annotation granularity from normal to brief, switching annotation language from English to Chinese, and replacing the backbone from SimLingo to CriticVLA. As shown in Table 9, the same pattern holds across all settings: language improves driving on only a minority of routes, actively degrades it on another subset, and has no clear effect on the majority. This consistent complementarity across annotation settings and backbones supports applying BLUE to different VLA models. We provide statistical testing details for each setting in Appendix D.2.

![Refer to caption](https://arxiv.org/html/2606.08684v1/x7.png)

Figure 7: Effect of training data size on gate performance. Both SR and DS improve quickly with more training data and then approach saturation.

| Category | Setting | Helpful | Neutral | Harmful |
| --- | --- | --- | --- | --- |
| Original | SimLingo | 14.5% | 61.8% | 23.6% |
| Granularity | Brief | 21.8% | 52.7% | 25.5% |
| Language | Chinese | 18.6% | 58.2% | 23.2% |
| Model | CriticVLA | 19.5% | 52.3% | 28.2% |

Table 9: Language usefulness across different settings. Helpful, neutral, and harmful denote the proportion of routes where language helps, has no effect, or hurts.

## 6 Conclusion

We present BLUE, a minimal method for better language use in VLA driving. We reveal that generated language helps in some situations, hurts in others, and is unnecessary most of the time. Based on this observation, BLUE trains a 0.11M-parameter gate on frozen VLA hidden states to decide when to generate language. It achieves 76.2% success rate on Bench2Drive, 36 driving score on Longest6 v2, setting new state of the art while delivering 2.54 $\times$ inference speedup. These results suggest that better language use in VLA driving does not come from generating more language, but from generating language only when it improves driving.

## Limitations

Despite its effectiveness, BLUE has two limitations. First, BLUE introduces uneven per-frame latency, as frames that skip language generation run faster than those that generate language. However, this is not unique to BLUE. Language-generating VLA systems inherently exhibit variable per-frame latency because the number of output tokens differs across frames. Since the gate adds negligible overhead, BLUE barely increases the maximum per-frame latency compared with the original VLA, while substantially reducing the average. Second, BLUE requires training a separate gate for each backbone. However, the gate is a lightweight single-layer MLP and the VLA backbone remains entirely frozen, so the adaptation cost is low. Training labels are a natural byproduct of routine driving evaluation and require no additional human annotation or reward engineering. Applying BLUE to a new backbone is therefore considerably cheaper than retraining or modifying the backbone itself.

## References

## Contents

## Appendix A The Novelty and Contribution of BLUE

In this section, we summarize the main novelty of BLUE and explain how does this work contribute to the NLP community. Detailed comparisons with related methods are provided in Section B.5.

### A.1 Novelty of BLUE

BLUE is related to recent work on language-augmented driving, efficient reasoning, and adaptive computation. We highlight five aspects that distinguish BLUE, while deferring detailed method-by-method comparisons to Section B.5.

##### Systematic diagnosis of language utility.

Prior work on adaptive reasoning focuses on improving, accelerating, or selectively invoking language generation, but none has systematically quantified how generated language affects closed-loop driving outcomes. We conduct large-scale closed-loop evaluations on hundreds of routes and statistically categorize each route as language-helpful, language-neutral, or language-harmful. The results show that language improves driving on only a minority of routes, actively degrades it on another subset, and has no clear effect on the remaining majority. This finding offers a new perspective for the community: selective language use in VLA driving should not only reduce unnecessary computation but also actively prevent harmful language generation.

##### Hidden-state language-utility signal.

Existing methods rely on reinforcement learning rewards, scene complexity features, or architectural modifications to learn when to reason, all of which require retraining or modifying the VLA backbone. BLUE discovers that the pretrained hidden states of a frozen VLA potentially already encode whether language generation will benefit driving at the current frame. This means the gating decision does not require redesigning the model or additional training of the backbone. The gate simply reads out a signal that already exists inside the pretrained representations, keeping the original VLA unchanged and avoiding the cost of backbone-level retraining.

##### Low-cost label collection.

Existing adaptive-reasoning methods typically require constructing reward functions or curating specialized training data. BLUE derives its training labels by simply comparing route success when the VLA generates language versus when it predicts actions directly. This process requires no human annotation and no reward engineering. In autonomous driving, deploying or validating a model already involves collecting driving outcomes under different configurations. BLUE reuses these routine evaluation results as supervision, making label collection a natural byproduct of the standard development pipeline rather than an additional burden.

##### State-of-the-art results with full reproducibility.

Despite training only a 0.11M-parameter MLP gate on a frozen VLA backbone, BLUE achieves state-of-the-art closed-loop driving results on both Bench2Drive and Longest6 v2 while delivering 2.54 $\times$ inference speedup. Many concurrent driving methods remain closed-source, making their results difficult to reproduce or build upon. We fully release the code, training data, model checkpoints, and evaluation logs, providing the community with a fully reproducible baseline for future research on better language use in VLA driving.

##### Toward better language use in vision and robotics.

As language models are increasingly used in visual and robotic systems, language generation must improve downstream behavior under latency constraints. Existing approaches mainly optimize the amount or speed of generation. BLUE demonstrates a complementary principle: better language use requires deciding when to generate, not only how to generate. In embodied settings, unnecessary language can slow the system and steer actions in harmful ways. Maximizing language’s net positive impact therefore requires generation decisions that adapt to each input. This insight extends to any deployed system where language serves as an intermediate computation.

## Appendix B Related Work

BLUE sits at the intersection of end-to-end autonomous driving, adaptive reasoning, and driving evaluation. We first review end-to-end driving methods with a focus on language-augmented models (§B.1), then discuss efficient reasoning techniques in both VLA driving (§B.2) and LLMs (§B.3), followed by the benchmarks used for evaluation (§B.4). Finally, we provide a detailed comparison between BLUE and existing methods (§B.5).

### B.1 End-to-End Autonomous Driving

Autonomous driving has evolved from modular pipelines that chain perception, prediction, and planning into end-to-end models that directly map sensor inputs to driving actions [^6] [^24] [^76]. Within this paradigm, diverse architectural designs have emerged, including multi-modal transformer fusion [^57] [^8] [^64], vectorized scene representation for planning [^24] [^34] [^43] [^32], learning-based planner supervision [^10] [^41] [^77] [^38], and diffusion-based or generative trajectory prediction [^44] [^89] [^48]. A growing line of work further integrates natural language into the driving loop through vision-language models and vision-language-action models [^88] [^91] [^90] [^27], which generate scene descriptions or reasoning chains as intermediate representations before producing control outputs [^67] [^73] [^63] [^28] [^33] [^70]. Complementary directions include world models for predictive simulation [^23] [^16] [^49] and reinforcement learning for driving policy optimization [^37] [^61] [^74].

### B.2 Adaptive Reasoning in Driving VLA

Recent VLA driving models typically generate language reasoning before predicting actions, but language generation introduces significant inference overhead. Several concurrent methods have explored strategies to reduce this cost. One line of work trains VLA models to select reasoning depth adaptively. AutoVLA [^92] builds a unified autoregressive VLA with physical action tokenization and applies supervised fine-tuning followed by GRPO-based reinforcement fine-tuning to reduce reasoning in straightforward scenarios. AdaThinkDrive [^53] trains a dual-mode think and non-think policy through supervised learning and GRPO with an adaptive think reward. Another line of work introduces multi-system architectures. DE-Driver [^79] designs a dual-expert VLA with a scene-aware router that dispatches inputs to a reactive or deliberative expert. SAMoE-VLA [^84] extends this idea with scene-adaptive mixture-of-experts routing, and FASIONAD [^58] adopts a dual-system framework where a fast system handles routine navigation while a slow system reasons from visual prompts and provides feedback in challenging situations. A related direction seeks alternative reasoning representations. FutureX [^45] uses an auto-think switch to choose between instant planning and latent world-model reasoning, invoking CoT-guided latent rollout only when additional reasoning is needed. DynVLA [^62] introduces a dynamics chain-of-thought that compresses future world evolution into compact dynamics tokens. From the efficiency perspective, FastDriveCoT [^18] applies parallel decoding to structured chain-of-thought templates, and Reasoning-VLA [^86] replaces autoregressive action decoding with learnable action queries for parallel trajectory generation. We provide a detailed comparison between BLUE and these methods in §B.5.1.

### B.3 Efficient Reasoning in LLMs

BLUE detects when language generation is needed in VLA driving. This question is related to efficient reasoning in NLP, where recent methods control the length of chain-of-thought through RL with length rewards or SFT on variable-length reasoning data.

#### B.3.1 RL with Length Reward Design

Early RL training for reasoning models focuses on accuracy rewards, which often leads to unnecessarily verbose chains of thought. To address this, recent methods incorporate length-based penalties into the reward function so that shorter correct answers receive higher scores. Kimi k1.5 [^72] adds a length penalty to its policy optimization to control long reasoning activations. O1-Pruner [^52] introduces a length-harmonizing reward with PPO, optimizing the ratio of reasoning lengths between a reference model and the student. L1 [^1] appends explicit token-budget constraints to the input before applying GRPO [^65]. Demystifying Long CoT [^83] proposes a cosine-shaped reward that penalizes excessive length while stabilizing training. DAST [^66] constructs length-preference data and trains with SimPO [^55] to adapt reasoning depth to problem difficulty. Arora et al. [^2] condition length rewards on correctness, assigning higher scores to shorter correct solutions. Other representative efforts include AdaptThink [^87], S-GRPO [^9], and ConciseRL [^14]. These methods share the goal of producing concise reasoning in text-based question answering and mathematics. BLUE addresses a related but distinct setting: instead of shortening textual reasoning, it decides whether to activate language generation at all in an embodied VLA driving model.

| Method | What is Adapted | Core Mechanism | Frozen | Supervision |
| --- | --- | --- | --- | --- |
| AutoVLA [^92] | Reasoning depth | Action token. + SFT + GRPO |  | RL reward |
| AdaThinkDrive [^53] | Think / non-think | Dual-mode SFT + GRPO |  | Adaptive think reward |
| DE-Driver [^79] | Reactive / delib. expert | Dual-expert + scene router |  | Scene-aware routing |
| SAMoE-VLA [^84] | Expert assignment | Scene-adaptive MoE |  | Scene features |
| FASIONAD [^58] | Fast / slow system | Dual-system + VLM feedback |  | Confidence score |
| FutureX [^45] | Instant / latent thinking | Auto-think + world model |  | World-model rollout |
| DynVLA [^62] | Text / dynamics CoT | Dynamics token prediction |  | SFT on dyn. tokens |
| FastDriveCoT [^18] | CoT decoding speed | Parallel structured decoding |  | Template structure |
| Reasoning-VLA [^86] | Action decoding | Action queries + parallel gen. |  | SFT |
| bbb!15 BLUE (ours) | Language gen. (on / off) | Hidden-state gate (0.11M) |  | Language utility |

Table 10: Comparison of BLUE with related adaptive reasoning methods for VLA driving. Frozen indicates whether the original VLA backbone weights remain frozen. BLUE is the only method that performs post-hoc language gating on a frozen VLA, with labels derived from closed-loop driving outcomes.

#### B.3.2 SFT with Variable-Length CoT

Beyond RL, supervised fine-tuning on curated variable-length reasoning data provides another route to efficient reasoning. These methods differ mainly in how the short chains are constructed. In the post-reasoning approach, Distilling System 2 into System 1 [^85] removes the reasoning process entirely and distills only the final answer. C3oT [^36] uses GPT-4 as a compressor to shorten reasoning while retaining key information. TokenSkip [^78] estimates the semantic importance of each reasoning segment and removes low-importance tokens. In the during-reasoning approach, Learn to Skip [^50] first creates concise solutions by manually merging or removing steps, then trains the model to intrinsically skip steps during inference. Token-Budget [^21] uses binary search to find the optimal token budget and trains the model to follow it. Self-Training [^56] samples multiple reasoning paths and selects the shortest correct one as training data. CoT-Valve [^54] progressively mixes parameters of long-reasoning and non-reasoning models to generate variable-length training data. Other related works include ReCUT [^35], ConCISE [^59], NCoTS [^46] and Ada-R1 [^51]. Like RL-based methods, these approaches focus on compressing textual reasoning chains in language tasks. BLUE operates at a coarser granularity: rather than shortening the generated language, it uses a lightweight gate to decide whether language generation should be activated for a given frame.

### B.4 Autonomous Driving Benchmarks

Evaluating autonomous driving models falls into two broad categories depending on whether the model’s own outputs influence future states. Open-loop benchmarks, including nuScenes [^4], Waymo Open [^69], and Argoverse 2 [^75], measure prediction quality against recorded ground truth but do not let the model’s actions affect subsequent observations. As a result, prediction errors that would compound during actual driving remain hidden in open-loop evaluation. Closed-loop benchmarks fill this gap by requiring the model to drive full routes inside a simulator or on replayed logs, where cumulative effects on driving outcomes such as route completion and collision avoidance can be directly measured. On the simulator side, CARLA [^12] provides a configurable urban environment that supports diverse traffic and weather conditions. Built on CARLA, Bench2Drive [^31] defines 220 routes across 44 scenario categories and evaluates models on multiple metrics such as success rate and driving score, covering five distinct driving skills. Longest6 v2 [^3] contains 36 long routes of 1–2 km each and evaluates sustained driving quality over extended distances. On the log-replay side, nuPlan [^5], WayMax [^19], NavSim [^11], and OpenAD [^40] replay real-world driving logs with reactive or non-reactive traffic agents. We evaluate BLUE on two closed-loop benchmarks: Bench2Drive, which tests multi-scenario driving ability, and Longest6 v2, which measures sustained driving quality over long-horizon routes. The two benchmarks are complementary in route length and scenario diversity, allowing us to examine whether our BLUE generalizes across different evaluation conditions.

### B.5 Comparison with Related Methods

We now compare BLUE with adaptive reasoning methods in VLA driving (§B.5.1) and with efficient reasoning methods in LLMs (§B.5.2).

#### B.5.1 Adaptive Reasoning in VLA

Section B.2 surveys the methods covered here. Table 10 provides a structured comparison of BLUE with these methods. As shown in Table 10, BLUE differs from existing methods in four aspects.

First, existing adaptive reasoning methods focus on learning when to activate longer or shorter reasoning, but none of them systematically examines how generated language affects closed-loop driving performance. BLUE fills this gap by conducting extensive evaluations and quantitatively categorizing language impact into helpful, neutral, and harmful cases, offering insights into when and why language generation should be selectively applied.

Second, all prior methods require modifying the VLA backbone or introducing new architectural components, whereas BLUE keeps the VLA entirely frozen and trains only a 0.11M-parameter gate. The minimal parameter count means the gate can be trained in minutes on a single GPU, adds negligible inference overhead, and avoids overfitting on limited calibration data, making BLUE a lightweight plug-in applicable to many existing VLA model without retraining.

Third, existing methods rely on reinforcement learning with custom reward functions, scene-level features, or new training objectives, all of which demand additional supervision and pipeline changes. BLUE instead discovers that pretrained VLA hidden states potentially already encode whether language generation will benefit driving, and the gate simply reads out this existing signal. The training labels are collected automatically by running the VLA and comparing driving outcomes, requiring no human annotation, no reward engineering, and minimal additional cost, as the data collection can be integrated into the routine road testing that deployed driving models already undergo.

Fourth, existing adaptive VLA methods such as AutoVLA and AdaThinkDrive [^92] [^53] base their reasoning decisions on scenario complexity, activating longer reasoning in difficult scenes and reducing it in simple ones. However, as demonstrated in Section 5, complexity-based gates perform comparably to simple kinematic heuristics and remain far below BLUE. The underlying reason, analyzed in detail in Appendix D.2, is that whether language helps depends not only on the scenario but also on how a specific model processes and utilizes language. The same scenario can be language-helpful under one model yet language-harmful under another, making complexity an unreliable proxy. BLUE addresses this limitation by conditioning the gating decision on model-specific hidden states, which jointly encode perceptual context and the model’s internal readiness to benefit from language generation.

#### B.5.2 Efficient Reasoning in LLM

Section B.3 surveys RL-based and SFT-based methods for controlling reasoning length in LLMs. BLUE shares the motivation of avoiding unnecessary reasoning but differs in three ways. First, LLM methods [^72] [^52] [^26] [^83] control how long a reasoning chain should be, producing shorter or longer traces depending on problem difficulty. BLUE makes a binary decision: whether language generation should be activated at all for a given driving frame. This coarser formulation suits VLA driving, where the two inference modes produce qualitatively different action distributions rather than merely longer or shorter textual outputs. Second, in text-based tasks the cost of reasoning is primarily token count and latency, whereas in closed-loop driving, generated language is an intermediate computation that alters the action trajectory. Unnecessary language can therefore actively degrade driving performance, a harmful effect that BLUE systematically quantifies and that has no direct counterpart in text-based settings. Third, LLM methods modify the language model itself through RL or SFT, while BLUE keeps the VLA backbone frozen and trains only a 0.11M-parameter gate supervised by paired closed-loop driving outcomes rather than task accuracy or token cost.

## Appendix C Additional Experimental Details

This section provides implementation and evaluation details needed for reproducibility.

### C.1 Details of Data Splits

We follow the standard data splits and ensure full separation between training and evaluation.

##### Training Set.

All data collection for BLUE, including hidden-state extraction, label construction, and gate training, is performed exclusively on routes from the SimLingo training set. For each training route, we run language mode and direct action mode separately through repeated experiments, recording the per-route success rate of each mode. The success rate gap between the two modes determines the training label for that route. We also extract the hidden state at every frame during these runs, which serves as the input feature for gate training. The specific route counts, seed configurations, and sampling strategy are detailed in Section C.5.

##### Evaluation Set.

We evaluate BLUE on the standard Bench2Drive test split and the full Longest6 v2 benchmark. These evaluation routes are entirely disjoint from the training routes, and our split is consistent with prior work to ensure fair comparison. No training data, labels, or hidden states are derived from evaluation routes.

##### Hyperparameter Selection.

We do not perform systematic hyperparameter search. All hyperparameters are set with simple heuristics. For example, gate threshold $\theta{=}0.66$ follows from the observation that language impact falls into three natural categories: language-harmful, language-neutral, and language-helpful. These three categories map to three equal intervals on the gate output range $[0,1]$, so we place the threshold at the boundary of the upper interval. This simple choice already yields strong performance. We provide threshold sensitivity analysis in Section 5.

### C.2 Details of Benchmarks Considered

We evaluate BLUE on two complementary closed-loop benchmarks built on the CARLA simulator [^12]. Bench2Drive tests multi-scenario driving ability over scenario-focused routes, while Longest6 v2 evaluates sustained driving quality over longer routes. Together they cover different route lengths and scenario diversities, allowing us to examine whether BLUE generalizes across evaluation conditions.

#### C.2.1 Details of Bench2Drive

Bench2Drive [^31] is a closed-loop benchmark built on CARLA v2 for evaluating end-to-end autonomous driving systems across multiple driving skills. It defines 220 evaluation routes distributed across 44 interactive scenario categories, 23 weather conditions, and 12 towns. Each route contains a single safety-critical scenario such as cut-in, overtaking, construction obstacle, or pedestrian crossing. The ego vehicle receives raw sensor inputs and target waypoints, and must drive from the source to the destination within an allotted time without traffic violations.

##### Multi-Ability Evaluation.

Bench2Drive groups the 44 scenarios into five high-level driving skills: Merging, Overtaking, Emergency Brake, Give Way, and Traffic Sign. Each skill score is the success rate over the corresponding subset of routes, and their average forms the multi-ability mean. This design enables disentangled assessment of individual driving capabilities.

##### Metrics.

Bench2Drive reports four evaluation metrics. Success Rate and Driving Score capture goal-achieving ability, while Driving Efficiency and Driving Smoothness measure driving quality.

##### Success Rate (SR).

Success Rate measures the proportion of routes completed within the allotted time without any traffic violation. A route is successful only if the ego vehicle reaches its destination with no recorded infraction. Formally, $\text{SR}=n_{\text{success}}/n_{\text{total}}$, where $n_{\text{success}}$ and $n_{\text{total}}$ denote the number of successful and total routes.

##### Driving Score (DS).

Driving Score follows the CARLA Leaderboard protocol and combines route completion with infraction penalties:

$$
\text{DS}=\frac{1}{n_{\text{total}}}\sum\nolimits_{i=1}^{n_{\text{total}}}\text{RC}_{i}\cdot\prod\nolimits_{j}p_{i,j},
$$

where $\text{RC}_{i}\in[0,100]$ is the route completion percentage for route $i$ and $p_{i,j}\in(0,1]$ is the penalty factor for the $j$ -th infraction. Penalties compound multiplicatively, so a single serious infraction can substantially reduce the score.

##### Driving Efficiency.

Driving Efficiency evaluates whether the ego vehicle maintains a reasonable speed relative to surrounding traffic. At each checkpoint, the ego speed is compared to the average speed of nearby vehicles, yielding a speed percentage. Bench2Drive checks speed every 5% of the route length across 20 checkpoints to reduce measurement variance. The final metric is the mean speed percentage across all checkpoints: $\overline{v}_{\%}=(\sum_{i}v_{\%,i})/C$, where $C$ is the total number of checkpoints.

##### Driving Smoothness.

Driving Smoothness evaluates trajectory comfort following the nuPlan protocol [^5]. At each frame, six kinematic variables are checked against expert-derived thresholds: longitudinal acceleration, lateral acceleration, yaw rate, yaw acceleration, longitudinal jerk, and jerk magnitude. A frame is smooth only if all six variables fall within their bounds. To mitigate the effect of necessary reactions such as emergency braking, Bench2Drive segments the trajectory into intervals of 20 timesteps and evaluates smoothness per segment. The final score is the ratio of smooth segments to total segments: $\text{Smoothness}=S_{\text{smooth}}/S_{\text{total}}$.

#### C.2.2 Details of Longest6 v2

Longest6 v2 [^3] is a closed-loop benchmark derived from the original Longest6 benchmark [^8]. It consists of 36 long routes of 1–2 km each and 7 scenario types. It evaluates whether a driving model can sustain safe and effective performance over longer driving horizons, where early errors and infractions can propagate and compound. Longest6 v2 reports three standard CARLA Leaderboard metrics.

##### Driving Score (DS).

Driving Score is defined identically to the Bench2Drive formulation: the average over all routes of the route completion percentage multiplied by the cumulative infraction penalty. Over longer routes, the multiplicative penalty structure makes Driving Score more sensitive to infractions, since more events can accumulate along the route.

##### Route Completion (RC).

Route Completion measures the average percentage of the prescribed route distance that the ego vehicle successfully traverses before termination. It captures how far the model can drive regardless of infractions.

##### Infraction Score (IS).

Infraction Score isolates the penalty component by reporting the average product of all per-route infraction multipliers: $\text{IS}=\frac{1}{n_{\text{total}}}\sum_{i}\prod_{j}p_{i,j}$. A higher Infraction Score indicates fewer and less severe violations. Together with Route Completion, Infraction Score enables diagnosis of whether low Driving Score stems from incomplete routes or from frequent infractions.

### C.3 Details of Baselines Considered

We compare BLUE against a wide range of published methods spanning end-to-end driving and vision-language-action approaches. Table 1 lists the sensor configuration and training labels for each method. As shown in the table, most baselines rely on surround cameras, LiDAR, or dense auxiliary labels such as 3D object detection, HD maps, semantic segmentation, and depth. In contrast, BLUE uses only front-view camera without LiDAR and requires only language annotations, yet achieves the best closed-loop results on both benchmarks. Below we briefly describe each baseline.

##### UniAD-Base

unifies perception, prediction, and planning into a single network, where all tasks communicate through unified query interfaces and are jointly optimized toward planning. It uses six surround cameras and trains with 3D object detection, map, and segmentation labels.

##### TF++

analyzes training data biases and proposes a label-change-based frame selection criterion to compress datasets without losing important information. It uses a front-view camera with LiDAR and trains with object detection, map, segmentation, and depth labels.

##### MomAD

introduces trajectory momentum and perception momentum to stabilize long-horizon planning by selecting planning queries topologically consistent with historical paths and fusing them with historical context. It uses six cameras with object detection and map labels.

##### DriveTransformer

replaces the sequential perception-prediction-planning pipeline with parallel task interaction, where agent, map, and planning queries directly attend to each other at every block. It uses six cameras with object detection and map labels.

##### Hydra-NeXt

adopts a multi-branch framework that unifies trajectory prediction, control prediction, and trajectory refinement to bridge the gap between open-loop training and closed-loop driving. It uses two cameras without auxiliary perception labels.

##### Raw2Drive

follows a dual-stream model-based reinforcement learning approach, first training a privileged world model and then aligning a raw-sensor world model to it through a guidance mechanism. It uses six cameras with map and segmentation labels.

##### DiffusionDrive

applies a truncated diffusion policy with multi-mode anchors, compressing the denoising process to produce diverse driving actions in only a few steps. It uses multiple cameras with LiDAR and trains with object and segmentation labels.

##### ORION

bridges semantic reasoning and numerical trajectory output by combining a QT-Former for long-term context aggregation, a large language model for scene reasoning, and a generative planner for trajectory prediction. It uses six cameras with object detection, map, and language labels.

##### AutoVLA

unifies reasoning and action generation within a single autoregressive model by tokenizing continuous trajectories into discrete action tokens. It supports fast and slow thinking modes and uses GRPO-based reinforcement fine-tuning to reduce unnecessary reasoning. It uses a front-view camera with language labels.

##### SimLingo

is a camera-only vision-language model that jointly handles closed-loop driving, vision-language understanding, and language-action alignment. It uses a single front-view camera with language annotations and serves as the primary backbone for BLUE.

##### HiP-AD

introduces multi-granularity planning queries that integrate spatial, temporal, and driving-style waypoints, and uses deformable attention to retrieve image features based on physical trajectory locations. It uses six cameras with object detection, map, and depth labels.

##### ReCogDrive

combines an autoregressive cognitive model with a diffusion planner, where the former provides driving reasoning priors and the latter generates continuous trajectories. It uses six cameras with language labels.

##### GeRo

extends VLA models with language-conditioned autoregressive generation of future traffic scenes, encoding ego and agent dynamics into shared latent tokens and stabilizing long-horizon rollouts with a consistency loss. It uses six cameras with object detection, map, and language labels.

##### DeLL

addresses lifelong learning in end-to-end driving through a Dirichlet process mixture model that builds dynamic knowledge spaces, combined with front-door causal adjustment to suppress spurious correlations. It uses a front-view camera with LiDAR and trains with object detection and segmentation labels.

##### R2SE

proposes a three-stage pipeline: generalist pretraining with hard-case identification, residual reinforcement fine-tuning on difficult scenarios, and self-aware adapter expansion that routes between generalist and specialist policies at test time. It uses a front-view camera with LiDAR and trains with object detection, map, segmentation, and depth labels.

##### AutoMoT

unifies reasoning and action generation within a mixture-of-transformers architecture, where a frozen understanding expert and a high-frequency action expert share a joint attention space for asynchronous fast-slow inference. It uses a front-view camera with LiDAR.

##### BevAD

re-examines common architectural patterns for closed-loop driving and identifies effective combinations of spatial bottleneck compression, decoupled trajectory representation, and diffusion-based planning. It uses six cameras with object detection labels.

##### CriticVLA

extends VLA models from acting to judging: it generates a rough trajectory and then uses the VLA as a critic to evaluate and refine the plan through single-step optimization. It uses a single front-view camera with language labels and serves as a secondary backbone for validating BLUE.

##### TakeVLA

improves VLA driving through post-training on expert takeover data, shifting language supervision to the period before takeover moments so that the model learns to anticipate hazards early. It uses a single front-view camera with language labels.

### C.4 Details of Label Construction

The gate requires binary labels indicating whether each frame benefits from language generation. A key advantage of our labeling approach is that it requires no human annotation: all labels are derived automatically from closed-loop evaluation outcomes by comparing the two modes. We construct labels at two granularities, route-level and frame-level, and apply temporal redundancy cleaning before training.

#### C.4.1 Route-Level Labels

We run language mode and direct action mode on each training route with $|\mathcal{S}|{=}5$ random seeds. The cross-seed success rate for mode $m$ on route $r$ is $\overline{\mathrm{SR}}_{m}^{(r)}=\frac{1}{|\mathcal{S}|}\sum_{s\in\mathcal{S}}\mathrm{SR}_{m}^{(r,s)}$. As defined in Eq. 1, a route receives label $y_{r}{=}1$ when the language advantage exceeds a margin threshold:

$$
\Delta\overline{\mathrm{SR}}_{r}=\overline{\mathrm{SR}}_{\text{lang}}^{(r)}-\overline{\mathrm{SR}}_{\text{direct}}^{(r)}>\tau,
$$

where $\tau{=}10\%$ ensures that language mode is activated only when it provides a sufficiently large performance gain. Routes that do not meet this condition are labeled $y_{r}{=}0$, defaulting to the faster direct action mode. Under route-level labeling, all frames within a route share the same label $y_{r}$.

#### C.4.2 Frame-Level Labels

Route-level labels assign a uniform label to all frames in a route, but in practice, even on a language-beneficial route, only certain segments truly require language. To provide finer supervision, we identify critical regions $\mathcal{C}_{r}$ where the two modes exhibit the largest behavioral divergence. The core idea is straightforward: we spatially compare how the vehicle behaves under the two modes and mark the locations where their behaviors differ the most. Concretely, for each language-beneficial route, we collect 2D ego-vehicle trajectories from all seeds under both modes and overlay them on a uniform spatial grid. For each grid cell $\mathbf{g}$, we measure the normalized cross-mode behavioral difference for each signal channel $k$:

$$
\Delta_{k}(\mathbf{g})=\frac{|\bar{v}_{k}^{\text{lang}}(\mathbf{g})-\bar{v}_{k}(\mathbf{g})|}{\max_{\mathbf{g}^{\prime}}|\bar{v}_{k}^{\text{lang}}(\mathbf{g}^{\prime})-\bar{v}_{k}(\mathbf{g}^{\prime})|+\epsilon},
$$

where $\bar{v}_{k}(\mathbf{g})$ and $\bar{v}_{k}^{\text{lang}}(\mathbf{g})$ are the seed-averaged behavioral signals at cell $\mathbf{g}$ under direct action mode and language mode, respectively. The behavioral channels include speed, acceleration, heading, and trajectory spread. Additionally, we construct an infraction channel by placing spatial kernels at infraction locations from simulation logs, measuring where the two modes differ in safety outcomes.

We aggregate all channels into a per-cell criticality score $c(\mathbf{g})=\sum_{k}w_{k}\cdot\Delta_{k}(\mathbf{g})$, threshold the resulting map, and retain the top spatial regions as critical regions $\mathcal{C}_{r}$. A frame receives $y_{r,t}{=}1$ only if both $y_{r}{=}1$ and its spatial position falls within $\mathcal{C}_{r}$ (Eq. 2). During training, we mix route-level and frame-level samples so that the gate learns both coarse route preference and fine-grained activation patterns. We will fully release the labeling code for complete implementation details.

#### C.4.3 Temporal Redundancy Cleaning

When the vehicle remains still, such as while waiting at a red light, many consecutive frames contain almost the same scene and therefore produce almost identical hidden states. If we keep all of them, these low-motion moments would be overrepresented during training. We detect such redundant segments by computing cosine similarity between adjacent hidden-state vectors:

$$
\mathrm{sim}(\mathbf{h}_{t},\mathbf{h}_{t+1})=\frac{\mathbf{h}_{t}^{\top}\mathbf{h}_{t+1}}{\|\mathbf{h}_{t}\|\cdot\|\mathbf{h}_{t+1}\|}\geq 0.99.
$$

Adjacent frames whose similarity exceeds this threshold are treated as one redundant segment of length $L$. From each segment, we keep only $k=\max(2,\;\lceil L^{\alpha}\rceil)$ evenly spaced representative samples, where $\alpha{=}0.5$. This sublinear schedule ensures short segments retain at least two samples while preventing long idle periods from overwhelming the dataset. After cleaning, the training set reduces by approximately 15% in frame count while preserving coverage of all driving states.

### C.5 Implementation and Hyperparameters

This subsection reports all implementation details needed to reproduce the gate training pipeline.

#### C.5.1 Data Collection

We extract the hidden state $\mathbf{h}\in\mathbb{R}^{d}$ from the last transformer layer at the final token position of a prompt-only forward pass, where $d{=}896$ is the hidden dimension of InternVL2-1B [^7]. For both language mode and direct action mode, we forward only the prompt tokens, including visual tokens and system instructions, and collect $\mathbf{h}$ from the same position. This ensures that hidden states from the two modes are aligned and that the gate receives comparable inputs regardless of which mode produced them. All data are collected on approximately 400 routes sampled from the SimLingo training set, stratified by scenario difficulty to cover both common and rare driving situations. For each route, we run both modes separately with 5 random seeds, recording per-frame hidden states and per-route success outcomes. The success rate gap determines training labels as described in Section C.4. BLUE does not require a dedicated data collection effort. In autonomous driving development, validating a model already involves running closed-loop evaluations under different configurations. BLUE simply reuses hidden states and outcomes from these routine runs, making data collection a natural byproduct of the standard pipeline rather than an additional cost.

#### C.5.2 Gate Architecture and Training

The gate is a single-hidden-layer MLP that maps $\mathbf{h}$ to a scalar activation probability:

$$
\mathbf{z}=W_{1}\mathbf{h}+b_{1},\quad p(\mathbf{h})=\sigma(W_{2}\,\tilde{\mathbf{z}}+b_{2}),
$$

where $\tilde{\mathbf{z}}$ denotes $\mathbf{z}$ after ReLU activation and dropout, $W_{1}{\in}\mathbb{R}^{m\times d}$, $W_{2}{\in}\mathbb{R}^{1\times m}$, and $\sigma$ is the sigmoid function. We set hidden dimension $m{=}128$ and dropout rate 0.5, yielding 0.11M total trainable parameters. We optimize with Adam using learning rate 0.001, weight decay 0.05, cosine annealing schedule, and batch size 512 for 100 epochs. The training set contains approximately 670K frames.

#### C.5.3 Computational Cost

The total overhead of BLUE is minimal, largely because its data collection naturally aligns with the standard autonomous driving development workflow. Models must undergo closed-loop road testing before deployment, during which driving outcomes under different configurations are routinely recorded. BLUE simply records hidden states alongside these existing evaluation runs and derives labels automatically by comparing route success rates between the two modes. This means data collection requires minimal additional effort beyond routine validation, no reward engineering, and no human labeling. The raw GPU time consumed by these evaluation runs is approximately 1200 A100 GPU hours, but the vast majority is not a net addition to the development budget. Gate training requires less than 0.1 GPU hours on a single GPU, and the inference overhead is negligible as it adds only a single MLP forward pass per frame. Separately, the language impact analysis in Appendix D.2 consumes approximately $\sim$ 2000 A100 GPU hours in total, covering repeated closed-loop experiments across multiple configurations.

### C.6 Details of Baseline Gates

This section describes the construction of alternative gating strategies evaluated in Table 7. All baseline gates share the same inference procedure as BLUE: at each frame, the gate produces a binary decision that determines whether to activate language generation or predict actions directly.

##### Kinematic Gates.

We design three rule-based gates, each using a single kinematic feature available at inference time: vehicle speed in m/s, acceleration magnitude as the L2 norm of the 3D acceleration vector in m/s <sup>2</sup>, and steering angle as the absolute value of the previous control output normalized to 0–1. A gate activates language generation whenever the corresponding feature exceeds a predefined threshold, selected so that the language activation ratio approximates that of BLUE.

##### Complexity-based Gate.

Rather than relying on frame-level kinematic signals, the complexity-based gate uses route-level scenario complexity as supervision to train a gate. We first compute a composite complexity score for each training route based on structured features extracted from the CARLA scenario configuration file:

$$
s=w_{1}\cdot f_{\text{scen}}+w_{2}\cdot f_{\text{wea}}+w_{3}\cdot f_{\text{flow}}+w_{4}\cdot f_{\text{freq}},
$$

where $f_{\text{scen}}$ reflects the number of sub-scenarios in the route normalized to \[0, 1\], $f_{\text{wea}}$ aggregates fog density, precipitation intensity, and a night-driving indicator, $f_{\text{flow}}$ indicates the presence of dynamic traffic flows requiring gap-finding maneuvers, and $f_{\text{freq}}$ captures periodic opposing vehicle interactions. We set $w_{1}{=}0.30$, $w_{2}{=}0.30$, $w_{3}{=}0.25$, $w_{4}{=}0.15$. Routes with $s\geq\tau$ are labeled as complex and the rest as simple. All frames within a route share the same label. We then train a gate with the same MLP architecture as BLUE, supervised with these complexity-derived labels. At inference, this gate predicts whether the current frame corresponds to a complex scenario and activates language generation accordingly.

##### Random Gate.

The random gate activates language generation for each frame with a fixed probability, matched to the language activation ratio of BLUE. This baseline isolates the effect of selectively choosing when to generate language from the effect of simply reducing language frequency.

<table><tbody><tr><td rowspan="2">Method</td><td colspan="5">Details</td><td colspan="2">Metrics</td></tr><tr><td>Expert</td><td>Camera</td><td>LiDAR</td><td>Labels</td><td>T-Param.</td><td>SR (%) <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>DS <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td></tr><tr><td>TCP* <sup><a href="#fn:76">76</a></sup></td><td>Think2Drive</td><td>3 <math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math></td><td></td><td>-</td><td><math><semantics><mo>≈</mo> <annotation>\approx</annotation></semantics></math>  26 M</td><td>15.00</td><td>40.70</td></tr><tr><td>TCP-traj* <sup><a href="#fn:76">76</a></sup></td><td>Think2Drive</td><td>3 <math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math></td><td></td><td>-</td><td><math><semantics><mo>≈</mo> <annotation>\approx</annotation></semantics></math>  26 M</td><td>30.00</td><td>59.90</td></tr><tr><td>VAD <sup><a href="#fn:34">34</a></sup></td><td>Think2Drive</td><td>6 <math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math></td><td></td><td>O,M</td><td><math><semantics><mo>≥</mo> <annotation>\geq</annotation></semantics></math>  25 M</td><td>15.00</td><td>42.35</td></tr><tr><td>UniAD-Base <sup><a href="#fn:24">24</a></sup></td><td>Think2Drive</td><td>6 <math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math></td><td></td><td>O,M,S</td><td><math><semantics><mo>≥</mo> <annotation>\geq</annotation></semantics></math>  59 M</td><td>16.36</td><td>45.81</td></tr><tr><td>ThinkTwice* <sup><a href="#fn:30">30</a></sup></td><td>Think2Drive</td><td>6 <math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math></td><td></td><td>S,D</td><td><math><semantics><mo>≈</mo> <annotation>\approx</annotation></semantics></math>  120 M</td><td>31.23</td><td>62.44</td></tr><tr><td>DriveAdaptor* <sup><a href="#fn:29">29</a></sup></td><td>Think2Drive</td><td>6 <math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math></td><td></td><td>M,S,D</td><td><math><semantics><mo>≈</mo> <annotation>\approx</annotation></semantics></math>  135 M</td><td>33.08</td><td>64.22</td></tr><tr><td>GenAD <sup><a href="#fn:89">89</a></sup></td><td>Think2Drive</td><td>6 <math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math></td><td></td><td>O,M</td><td><math><semantics><mo>≥</mo> <annotation>\geq</annotation></semantics></math>  25 M</td><td>15.90</td><td>44.81</td></tr><tr><td>TF++ <sup><a href="#fn:93">93</a></sup></td><td>PDM-Lite</td><td>1 <math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math></td><td></td><td>O,M,S,D</td><td><math><semantics><mo>≥</mo> <annotation>\geq</annotation></semantics></math>  39 M</td><td>67.27</td><td>84.21</td></tr><tr><td>MomAD <sup><a href="#fn:68">68</a></sup></td><td>Think2Drive</td><td>6 <math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math></td><td></td><td>O,M</td><td><math><semantics><mo>≥</mo> <annotation>\geq</annotation></semantics></math>  25 M</td><td>18.11</td><td>47.91</td></tr><tr><td>DriveTrans <sup><a href="#fn:32">32</a></sup></td><td>Think2Drive</td><td>6 <math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math></td><td></td><td>O,M</td><td><math><semantics><mo>≈</mo> <annotation>\approx</annotation></semantics></math>  646 M</td><td>35.01</td><td>63.46</td></tr><tr><td>ETA <sup><a href="#fn:20">20</a></sup></td><td>Think2Drive</td><td>1 <math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math></td><td></td><td>-</td><td><math><semantics><mo>≥</mo> <annotation>\geq</annotation></semantics></math>  300 M</td><td>48.33</td><td>74.33</td></tr><tr><td>Hydra-NeXt <sup><a href="#fn:42">42</a></sup></td><td>Think2Drive</td><td>2 <math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math></td><td></td><td>-</td><td><math><semantics><mo>≥</mo> <annotation>\geq</annotation></semantics></math>  25 M</td><td>50.00</td><td>73.86</td></tr><tr><td>Raw2Drive <sup><a href="#fn:81">81</a></sup></td><td>-</td><td>6 <math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math></td><td></td><td>M,S</td><td><math><semantics><mo>≥</mo> <annotation>\geq</annotation></semantics></math>  25 M</td><td>50.24</td><td>71.36</td></tr><tr><td>DiffusionDrive <sup><a href="#fn:44">44</a></sup></td><td>-</td><td>3 <math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math></td><td></td><td>O,S</td><td><math><semantics><mo>≈</mo> <annotation>\approx</annotation></semantics></math>  60 M</td><td>52.72</td><td>77.68</td></tr><tr><td>ORION <sup><a href="#fn:15">15</a></sup></td><td>Think2Drive</td><td>6 <math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math></td><td></td><td>O,M,L</td><td><math><semantics><mo>≥</mo> <annotation>\geq</annotation></semantics></math>  300 M</td><td>54.62</td><td>77.74</td></tr><tr><td>AutoVLA <sup><a href="#fn:92">92</a></sup></td><td>PDM-Lite</td><td>1 <math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math></td><td></td><td>L</td><td><math><semantics><mo>≥</mo> <annotation>\geq</annotation></semantics></math>  1.5 B</td><td>57.73</td><td>78.84</td></tr><tr><td>SimLingo <sup><a href="#fn:60">60</a></sup></td><td>PDM-Lite</td><td>1 <math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math></td><td></td><td>L</td><td><math><semantics><mo>≥</mo> <annotation>\geq</annotation></semantics></math>  300 M</td><td>67.27</td><td>85.07</td></tr><tr><td>HiP-AD <sup><a href="#fn:71">71</a></sup></td><td>Think2Drive</td><td>6 <math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math></td><td></td><td>O,M,D</td><td><math><semantics><mo>≈</mo> <annotation>\approx</annotation></semantics></math>  97 M</td><td>69.09</td><td>86.77</td></tr><tr><td>ReCogDrive <sup><a href="#fn:39">39</a></sup></td><td>Think2Drive</td><td>6 <math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math></td><td></td><td>L</td><td><math><semantics><mo>≥</mo> <annotation>\geq</annotation></semantics></math>  2 B</td><td>45.45</td><td>71.36</td></tr><tr><td>GeRo <sup><a href="#fn:82">82</a></sup></td><td>Think2Drive</td><td>6 <math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math></td><td></td><td>O,M,L</td><td><math><semantics><mo>≥</mo> <annotation>\geq</annotation></semantics></math>  3 B</td><td>60.10</td><td>81.90</td></tr><tr><td>DeLL <sup><a href="#fn:13">13</a></sup></td><td>Think2Drive</td><td>1 <math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math></td><td></td><td>O,S</td><td><math><semantics><mo>≥</mo> <annotation>\geq</annotation></semantics></math>  38 M</td><td>68.63</td><td>86.86</td></tr><tr><td>R2SE <sup><a href="#fn:47">47</a></sup></td><td>PDM-Lite</td><td>1 <math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math></td><td></td><td>O,M,S,D</td><td><math><semantics><mo>≥</mo> <annotation>\geq</annotation></semantics></math>  39 M</td><td>69.54</td><td>86.28</td></tr><tr><td>AutoMoT <sup><a href="#fn:25">25</a></sup></td><td>PDM-Lite</td><td>1 <math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math></td><td></td><td>-</td><td><math><semantics><mo>≈</mo> <annotation>\approx</annotation></semantics></math>  1.6 B</td><td>70.00</td><td>87.34</td></tr><tr><td>BevAD <sup><a href="#fn:22">22</a></sup></td><td>PDM-Lite</td><td>6 <math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math></td><td></td><td>O</td><td><math><semantics><mo>≥</mo> <annotation>\geq</annotation></semantics></math>  25 M</td><td>72.73</td><td>88.11</td></tr><tr><td>CriticVLA <sup><a href="#fn:80">80</a></sup></td><td>PDM-Lite</td><td>1 <math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math></td><td></td><td>L</td><td><math><semantics><mo>≥</mo> <annotation>\geq</annotation></semantics></math>  300 M</td><td>73.33</td><td>88.02</td></tr><tr><td>TakeVLA <sup><a href="#fn:17">17</a></sup></td><td>PDM-Lite</td><td>1 <math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math></td><td></td><td>L</td><td><math><semantics><mo>≥</mo> <annotation>\geq</annotation></semantics></math>  300 M</td><td>73.73</td><td>89.72</td></tr><tr><td>bbb!15BLUE (SimLingo)</td><td>PDM-Lite</td><td>1 <math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math></td><td></td><td>L</td><td>0.11 M</td><td>76.18 <math><semantics><mo>±</mo> <annotation>\pm</annotation></semantics></math> 0.64</td><td>90.58 <math><semantics><mo>±</mo> <annotation>\pm</annotation></semantics></math> 0.12</td></tr><tr><td><math><semantics><mi>Δ</mi> <annotation>\Delta</annotation></semantics></math> vs. SimLingo</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>+8.91</td><td>+5.51</td></tr><tr><td>bbb!15BLUE (CriticVLA)</td><td>PDM-Lite</td><td>1 <math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math></td><td></td><td>L</td><td>0.11 M</td><td>76.04 <math><semantics><mo>±</mo> <annotation>\pm</annotation></semantics></math> 0.38</td><td>90.37 <math><semantics><mo>±</mo> <annotation>\pm</annotation></semantics></math> 0.14</td></tr><tr><td><math><semantics><mi>Δ</mi> <annotation>\Delta</annotation></semantics></math> vs. CriticVLA</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>+2.71</td><td>+2.35</td></tr></tbody></table>

Table 11: Full results on Bench2Drive, showing the complete comparison between BLUE and 26 baselines. BLUE (SimLingo) ranks first and BLUE (CriticVLA) ranks second in terms of closed-loop success rate (SR) and driving score (DS). T-Param. reports driving-task trainable parameters; we use published values ($\approx$) where available and conservative lower bounds ($\geq$) derived from the minimum size of trained components. Both BLUE variants train only a 0.11 M gate while keeping the entire VLA backbone frozen, yet surpass methods that employ multi-camera setups, LiDAR, or dense auxiliary labels (O: 3D object detection, M: map, S: semantic segmentation, D: depth, L: language), using only a single front-view camera with language annotations.

## Appendix D Additional Experimental Results

The main text reports key results on Bench2Drive. This section provides the complete comparison tables, covering 26 baselines alongside our BLUE on both SimLingo and CriticVLA backbones.

### D.1 Full Bench2Drive Comparison

Table 11 presents the full Bench2Drive comparison across 26 methods. For completeness, we include both BLUE (SimLingo) and BLUE (CriticVLA) in a single table. Both variants use only a single front-view camera and language annotations, yet outperform their respective backbones by clear margins. BLUE (SimLingo) achieves the overall best SR and DS among all methods, while BLUE (CriticVLA) also surpasses its backbone and ranks among the top entries. This confirms that the on-demand language use strategy is effective across different VLA architectures under the same evaluation protocol.

<table><thead><tr><th>Category</th><th>Setting</th><th>Method</th><th>Helpful (%)</th><th>Neutral (%)</th><th>Harmful (%)</th></tr></thead><tbody><tr><th rowspan="2">Original</th><th rowspan="2">SimLingo</th><th>Threshold</th><td>14.5</td><td>61.8</td><td>23.6</td></tr><tr><th>Sign test</th><td>14.5</td><td>62.7</td><td>22.7</td></tr><tr><th rowspan="2">Granularity</th><th rowspan="2">Brief</th><th>Threshold</th><td>21.8</td><td>52.7</td><td>25.5</td></tr><tr><th>Sign test</th><td>21.8</td><td>52.7</td><td>25.5</td></tr><tr><th rowspan="2">Language</th><th rowspan="2">Chinese</th><th>Threshold</th><td>18.6</td><td>58.2</td><td>23.2</td></tr><tr><th>Sign test</th><td>15.9</td><td>67.7</td><td>16.4</td></tr><tr><th rowspan="2">Model</th><th rowspan="2">CriticVLA</th><th>Threshold</th><td>19.5</td><td>52.3</td><td>28.2</td></tr><tr><th>Sign test</th><td>18.2</td><td>55.5</td><td>26.4</td></tr></tbody></table>

Table 12: Language impact categorization across four experimental settings, each evaluated with both the threshold method and the sign test. The two methods yield closely aligned results. Across all settings, the majority of routes are neutral, and language-harmful routes consistently equal or outnumber language-helpful routes.

### D.2 Language Impact Analysis

This section details the experimental settings and statistical methods used to categorize each route as language-helpful, language-neutral, or language-harmful, as summarized in Table 9 of the main text. All analyses in this section are conducted on the Bench2Drive evaluation set. The gate training uses only training routes with no overlap with the evaluation set, as detailed in Appendix C.1.

#### D.2.1 Experimental Settings

We evaluate the language impact under four settings that vary annotation granularity, annotation language, and VLA backbone. All settings are evaluated on the full 220 routes of Bench2Drive [^31], running each route with and without language generation across multiple random seeds. We classify each route using two complementary methods: a threshold method based on effect size and a binomial sign test for statistical significance. As Table 12 shows, both methods produce consistent categorizations across all four settings.

##### SimLingo.

The default setting uses the official SimLingo [^60] checkpoint trained with English annotations at normal granularity. The annotations describe surrounding objects, traffic conditions, and intended maneuvers at each frame.

##### Brief.

To test whether annotation granularity affects the language impact distribution, we retrain SimLingo with brief English annotations that retain only action-relevant instructions and remove detailed scene descriptions. The model architecture, training procedure, and evaluation protocol are identical to the default SimLingo setting.

##### Chinese.

To further test whether the pattern holds under additional annotation settings, we retrain SimLingo with Chinese annotations translated from the default English version. The translated annotations preserve the same content and structure. All other training details remain the same.

##### CriticVLA.

To test whether the pattern generalizes across VLA architectures, we replace the backbone with CriticVLA [^80]. Unlike SimLingo, which generates language reasoning before predicting actions in a single pass, CriticVLA first produces an initial trajectory and then uses language-based critique to refine the plan.

#### D.2.2 Statistical Methods

We apply two complementary methods to classify each route. The first provides an intuitive effect-size criterion, and the second offers formal statistical validation.

##### Threshold method.

For each route $r$, we compute the cross-seed average success rate under language mode $\overline{\mathrm{SR}}_{\text{lang}}^{(r)}$ and under direct action mode $\overline{\mathrm{SR}}_{\text{direct}}^{(r)}$, and take their difference $\Delta^{(r)}=\overline{\mathrm{SR}}_{\text{lang}}^{(r)}-\overline{\mathrm{SR}}_{\text{direct}}^{(r)}$. A route is classified as language-helpful if $\Delta>\tau$, language-harmful if $\Delta<-\tau$, and language-neutral otherwise, where $\tau{=}0.1$ requires the success rate gap to exceed 10% for a route to be considered meaningfully affected.

##### Sign test.

For each route $r$, we construct all cross-seed pairings between language seeds and direct action seeds, yielding $n_{\text{lang}}\times n_{\text{direct}}$ pairs per route. Among these, we count the discordant pairs: $n_{+}$ pairs where language succeeds and direct fails, and $n_{-}$ pairs where language fails and direct succeeds. Concordant pairs are excluded. Under the null hypothesis that the two modes are equivalent, $n_{+}$ follows a Binomial $(n_{+}+n_{-},0.5)$ distribution. We apply a one-sided binomial exact test at significance level $\alpha{=}0.15$: a route is classified as language-helpful if $P(X\geq n_{+})<\alpha$, language-harmful if $P(X\geq n_{-})<\alpha$, and language-neutral otherwise. Routes with zero discordant pairs are classified as neutral. We do not apply multiple testing correction because our conclusion depends on the aggregate proportion of the three categories across all routes, not on the classification of any single route.

#### D.2.3 Detailed Results

Table 12 reports the classification results under both methods across all four settings. The two methods produce highly consistent categorizations. In all settings, the majority of routes fall into the neutral category, and the number of language-harmful routes consistently equals or exceeds the number of language-helpful routes. This pattern holds regardless of annotation granularity, annotation language, or VLA backbone, confirming that the complementarity between the two modes is a general property rather than an artifact of a specific configuration. The agreement between the two methods confirms that our categorization is robust to the choice of statistical criterion, and that selective language activation is warranted across diverse configurations.

##### Per-scenario results.

Beyond aggregate route counts, Figure 8 shows how language impact is distributed across scenario categories under the four settings. The scenario-level view reveals that helpful and harmful effects are not confined to a single scenario family. Many categories are dominated by neutral routes, while language-sensitive cases appear as scattered helpful and harmful groups whose locations can change with the setting. Table 13 maps each scenario index in the figure to its Bench2Drive scenario name. These results support the main conclusion that language generation should be selected on demand rather than used by default at every frame.

#### D.2.4 Discussion

The cross-setting results above reveal two practical implications for gate design: why each backbone requires its own gate, and why scenario complexity alone cannot replace hidden-state conditioning.

##### Why each backbone needs its own gate.

Figure 8 shows that the helpful, neutral, and harmful distributions change across settings, with a clear shift when moving from SimLingo to CriticVLA. If language utility were determined only by the route, the same scenario index would show similar patterns across backbones. Instead, the pattern changes, indicating that language utility also depends on how each backbone uses language and represents the scene before acting. A gate trained for one backbone therefore learns a readout tied to that model, rather than a universal rule. BLUE trains a separate gate for each backbone, but this adds little cost. As discussed in Section [Limitations](#Sx1 "Limitations ‣ BLUE: Toward Better Language Use in Efficient Vision-Language-Action Models for Autonomous Driving"), the backbone remains frozen, the gate is a lightweight MLP, and labels come from routine evaluations.

##### Why scenario complexity is insufficient.

Figure 8 also explains why a gate based only on scenario complexity cannot work reliably. All four settings share the same set of driving scenarios, yet where language helps or hurts shifts when the model or annotation changes. This means the driving data alone, including route type and scenario complexity, cannot determine whether language will help. Recent adaptive VLA methods, including AutoVLA and AdaThinkDrive [^92] [^53], often reduce reasoning in straightforward scenes and keep more reasoning in difficult ones. Our results show that this heuristic misses the key signal: language generation may help on a given scenario under one model but hurt under another. Complexity alone is insufficient to predict when language will help. BLUE therefore conditions the decision on model hidden states rather than on scenario complexity alone.

![Refer to caption](https://arxiv.org/html/2606.08684v1/x8.png)

Figure 8: Per-scenario distribution of language effects under four settings. Each bar denotes one Bench2Drive scenario category; all four panels share the same scenario order for direct comparison. Colors indicate routes where language generation is helpful, neutral, or harmful. The scenario index mapping is provided in Table 13.

| bbb!12 ID | Scenario | ID | Scenario |
| --- | --- | --- | --- |
| 1 | MergerIntoSlowTrafficV2 | 23 | ControlLoss |
| 2 | Accident | 24 | CrossingBicycleFlow |
| 3 | HazardAtSideLane | 25 | HazardAtSideLaneTwoWays |
| 4 | InterurbanActorFlow | 26 | VanillaSignalizedTurnEncounterGreenLight |
| 5 | StaticCutIn | 27 | VanillaNonSignalizedTurnEncounterStopsign |
| 6 | ParkingExit | 28 | MergerIntoSlowTraffic |
| 7 | ConstructionObstacleTwoWays | 29 | T\_Junction |
| 8 | OppositeVehicleTakingPriority | 30 | SignalizedJunctionLeftTurnEnterFlow |
| 9 | DynamicObjectCrossing | 31 | SignalizedJunctionRightTurn |
| 10 | VanillaNonSignalizedTurn | 32 | EnterActorFlow |
| 11 | VanillaSignalizedTurnEncounterRedLight | 33 | NonSignalizedJunctionLeftTurn |
| 12 | ParkingCutIn | 34 | ParkedObstacleTwoWays |
| 13 | AccidentTwoWays | 35 | SignalizedJunctionLeftTurn |
| 14 | InvadingTurn | 36 | OppositeVehicleRunningRedLight |
| 15 | HighwayExit | 37 | ParkingCrossingPedestrian |
| 16 | InterurbanAdvancedActorFlow | 38 | SequentialLaneChange |
| 17 | NonSignalizedJunctionLeftTurnEnterFlow | 39 | VehicleTurningRoutePedestrian |
| 18 | YieldToEmergencyVehicle | 40 | PedestrianCrossing |
| 19 | ConstructionObstacle | 41 | BlockedIntersection |
| 20 | HighwayCutIn | 42 | VehicleOpensDoorTwoWays |
| 21 | NonSignalizedJunctionRightTurn | 43 | VehicleTurningRoute |
| 22 | HardBreakRoute | 44 | ParkedObstacle |

Table 13: Mapping between scenario IDs in Figure 8 and Bench2Drive scenario names.

## Appendix E Additional Analysis

This section provides additional analysis on the language content generated by SimLingo.

### E.1 Language Content Visualization

We visualize the language content generated during closed-loop driving through word clouds, providing readers with an intuitive impression of the vocabulary produced by language generation. Figure 9 presents the word clouds of generated language, where the left panel corresponds to the original SimLingo and the right panel corresponds to BLUE with language generated when the gate activates.

## Appendix F Additional Statements

We provide statements on artifact licensing, potential risks, broader impact, and the use of large language models in preparing this work.

##### Potential Risks

BLUE is evaluated entirely in simulation and is not intended for direct deployment on real vehicles. The gate is trained on statistical driving outcomes in CARLA, and deploying it in real-world conditions would require additional domain-specific training and thorough safety verification to account for sensor noise, distribution shift, and safety-critical edge cases. Any real-world application should follow established engineering protocols for autonomous driving systems.

##### Broader Impact

This work shows that selectively generating language in VLA driving models improves both driving performance and inference efficiency. By reducing unnecessary language generation, BLUE lowers the computational cost of VLA models, making them more practical for real-world vehicle deployment where hardware budgets and energy consumption are constrained. This also reduces the carbon footprint associated with running large language models at inference time. All code, data, and evaluation logs are fully released to facilitate reproducible research on efficient language use in embodied agents. Since BLUE operates thought simulation and studys when language benefits driving, we do not foresee direct negative societal consequences from this research.

##### LLM Use Statement

Large language models were used in this work exclusively for English language polishing and assisting with code implementation. All research ideas, experimental designs, analyses, and scientific conclusions are original contributions of the authors. The authors carefully reviewed all LLM-assisted outputs and take full responsibility for the final content of this paper.

![Refer to caption](https://arxiv.org/html/2606.08684v1/x9.png)

