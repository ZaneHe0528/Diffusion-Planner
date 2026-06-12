Yinan Zheng <sup>1</sup> <sup>1</sup>, Ruiming Liang <sup>2</sup> <sup>1</sup>   <sup>3</sup>, Kexin Zheng <sup>3</sup> <sup>1</sup>   <sup>3</sup>, Jinliang Zheng <sup>1</sup>, Liyuan Mao <sup>4</sup> <sup>3</sup>,  
Jianxiong Li <sup>1</sup>, Weihao Gu <sup>5</sup>, Rui Ai <sup>5</sup>, Shengbo Eben Li <sup>1</sup>, Xianyuan Zhan <sup>1,6</sup> <sup>2</sup>, Jingjing Liu <sup>1</sup> <sup>2</sup>  
<sup>1</sup> Tsinghua University   <sup>2</sup> Institute of Automation, Chinese Academy of Sciences  
<sup>3</sup> The Chinese University of Hong Kong   <sup>4</sup> Shanghai Jiao Tong University  
<sup>5</sup> HAOMO.AI   <sup>6</sup> Shanghai Artificial Intelligence Laboratory  
zhengyn23@mails.tsinghua.edu.cn, zhanxianyuan@air.tsinghua.edu.cn

###### Abstract

Achieving human-like driving behaviors in complex open-world environments is a critical challenge in autonomous driving. Contemporary learning-based planning approaches such as imitation learning methods often struggle to balance competing objectives and lack of safety assurance, due to limited adaptability and inadequacy in learning complex multi-modal behaviors commonly exhibited in human planning, not to mention their strong reliance on the fallback strategy with predefined rules. We propose a novel transformer-based Diffusion Planner for closed-loop planning, which can effectively model multi-modal driving behavior and ensure trajectory quality without any rule-based refinement. Our model supports joint modeling of both prediction and planning tasks under the same architecture, enabling cooperative behaviors between vehicles. Moreover, by learning the gradient of the trajectory score function and employing a flexible classifier guidance mechanism, Diffusion Planner effectively achieves safe and adaptable planning behaviors. Evaluations on the large-scale real-world autonomous planning benchmark nuPlan and our newly collected 200-hour delivery-vehicle driving dataset demonstrate that Diffusion Planner achieves state-of-the-art closed-loop performance with robust transferability in diverse driving styles. Project website: [https://zhengyinan-air.github.io/Diffusion-Planner/](https://zhengyinan-air.github.io/Diffusion-Planner/).

## 1 Introduction

Autonomous driving as a cornerstone technology, is poised to usher transportation into a safer and more efficient era of mobility [^44]. The key challenge is achieving human-like driving behaviors in complex open-world environment, while ensuring safety, efficiency, and comfort [^32]. Rule-based planning methods have demonstrated initial success in industrial applications [^16], by defining driving behaviors and establishing boundaries derived from human knowledge. However, their reliance on predefined rules limits adaptability to new traffic situations [^18], and modifying rules demands extensive engineering effort. In contrast, learning-based planning methods acquire driving skills by cloning human driving behaviors from collected datasets [^5], a process made simpler through straightforward imitation learning losses. Additionally, the capabilities of these models can potentially be enhanced by scaling up training resources [^7].

Though promising, current learning-based planning methods still face several limitations. Firstly, human drivers often exhibit multi-modal behaviors in planning scenarios [^33]. Existing methods that rely on behavior cloning lack a guarantee of fitting such complex data distributions, even when utilizing large transformer-based model architecture or sampling multiple trajectories [^8]. Secondly, when encountering out-of-distribution (OOD) scenarios, directly using model output may result in low-quality planning outcomes, forcing many methods to fall back on rule-based approaches for trajectory refinement optimization or filtering [^49] [^22], inevitably facing the same inherent limitations associated with rule-based methods. Thirdly, imitation learning alone is inadequate to capture the vast diversity of driving behaviors required for autonomous driving. For example, penalizing unsafe planning via auxiliary loss, as employed in existing methods [^2] [^9], often results in multi-objective conflicts and poor safety performance due to the lack of learning signals that can teach the agent to recover from mistakes [^52] [^6]. Additionally, well-trained models may be difficult to adapt behaviors to meet specific needs.

In this study, we discover that diffusion model [^19] possesses huge potential to address the aforementioned issues. Its ability to model complex data distributions [^10] allows for effective capturing of multi-modal human driving behavior. Additionally, the high-quality generation capability of the diffusion model also provides opportunities for improving the output trajectory quality through appropriate structural design, removing the reliance on rule-based refinement. The best part of diffusion lies in its flexible guidance mechanism [^15], which allows adaptation to various planning behavioral needs without additional training. Inspired by these observations, we introduce a novel learning-based approach, Diffusion Planner, which pioneers the use of diffusion models [^19] for enhancing closed-loop planning performance without any rule-based refinement. Diffusion Planner is realized by learning the gradient of vehicles’ trajectory score function [^39] to model the multi-modal data distribution, and further enables personalized planning behavior adaptation through a classifier guidance mechanism. Specifically, we propose a new network architecture built upon the diffusion transformer [^35]. The diffusion loss is employed to jointly train both prediction and planning tasks within the same architecture, enabling cooperative behaviors between vehicles without the need for additional loss functions. Moreover, the versatility of classifier guidance is further demonstrated by its ability to modify the planning behavior of the trained model, such as enhancing safety and comfort, or controlling the vehicle’s speed. The differentiable classifier score can be computed in parallel and is flexible for combination, without requiring additional training. Evaluation results on the large-scale real-world autonomous planning benchmark nuPlan [^5] demonstrate that Diffusion Planner achieves state-of-the-art closed-loop performance among learning-based baselines, comparable to or even surpassing rule-based methods, directly using the model’s output without any additional post-processing. By appending a existing post-processing module to the model, we further achieved state-of-the-art performance among all baselines. Additionally, we collected $200$ hours of long-term delivery-vehicle driving data in various city-driving scenarios that further validate the transferability and robustness of the model in diverse driving styles.

In summary, our contributions are:

- To the best of our knowledge, we are the first to fully harness the power of diffusion models with a specifically designed architecture for high-performance motion planning, without overly reliant on rule-based refinement.
- We achieve state-of-the-art performance on the real-world nuPlan dataset, generating more robust and smoother trajectories compared to the baselines.
- We demonstrate that our model can achieve personalized driving behavior at runtime by utilizing a flexible guidance mechanism, which is a desirable feature for real-world applications.
- We have collected and evaluated a new 200-hour delivery-vehicle dataset, which is compatible with the nuPlan framework, and we will open-source it.

## 2 Related work

Rule-based Planner. Rule-based methods rely on predefined rules to dictate the driving behavior of autonomous vehicles, offering a highly controllable and interpretable decision-making process [^46] [^16] [^13]. While they have been widely validated in real-world scenarios [^26] [^48], these frameworks are limited in their ability to handle novel complex situations that fall beyond the predefined rules.

Learning-based Planner. Learning-based planning focuses on leveraging methods such as behavior cloning in imitation learning to directly model human driving behaviors, which has emerged as a popular solution in autonomous driving, particularly in recent end-to-end training pipelines [^20] [^44] [^7]. Behavior cloning method was initially implemented using CNN [^3] [^25] [^18] or RNN [^2] networks and has since been extended to Transformer due to its strong performance and efficiency in fitting complex data distributions [^37] [^11]. However, these methods lack theoretical guarantees for modeling multi-modal driving behavior, which can lead to serious error accumulation in closed-loop planning. As a result, most existing approaches still heavily rely on rules to refine [^49] [^22] or select [^9] the generated trajectories, which in some sense, has failed their initial purpose of using learning to replace pre-defined rules. While learning-based methods could offer more human-like driving behavior, their uncontrollable outputs lack safety guarantees and are hard to adjust based on user needs. Existing methods add extra training losses [^2] [^9], but struggle to strike a balance among competing learning objectives. Additionally, these methods also lack flexibility, making post-training behavior adjustments difficult. In practice, it is desirable for a trained planning model to achieve flexible alignment to various safety and personalized driving preferences during inference, which is still lacking in the current literature. In this work, we develop a novel diffusion planner to tackle the above limitations, which enables the generation of high quality planning trajectories without the need for rule-based refinement, and flexible post-training adaptation to various driving styles through the diffusion guidance mechanism.

Diffusion-based Methods Used in Related Domain. Diffusion models have been recently explored in decision-making fields [^23] [^10] [^28], however, their use in autonomous planning has not yet been fully explored. Some existing works employ diffusion models for motion prediction [^24] and traffic simulation [^54] [^53], but their focus is on open-loop performance or diversity in simulation rather than quality or drivability, as the outputs are not directly used for control. There are also studies targeting planning [^21] [^51] [^42], but these approaches only apply diffusion loss to existing frameworks or stack parameters without specific design considerations, making them heavily reliant on post-processing for reasonable performance. In this paper, we demonstrate that with appropriate structural design, the potential of diffusion models can be fully harnessed to enhance closed-loop planning performance in autonomous driving.

## 3 Preliminaries

### 3.1 Autonomous Driving and closed-loop Planning

The primary objective of autonomous driving is to allow vehicles to navigate complex environments with minimal human intervention, where a critical challenge is closed-loop planning [^5]. Unlike open-loop planning [^4] or motion prediction [^34] [^55], which only involves decision making that adapts to static conditions, closed-loop planning requires a seamless integration of real-time perception, prediction, and control. Vehicles must continuously assess their surroundings, predict the behavior of other neighboring vehicles, and implement precise maneuvers. The dynamic nature of real-world driving scenarios, combined with uncertainty in sensor data and environmental factors, makes closed-loop planning a formidable task.

### 3.2 Diffusion Model and Guidance Schemes

Diffusion Model. Diffusion Probabilistic Models [^38] [^19] are a class of generative models that generate outputs by reversing a Markov chain process known as the forward diffusion process. The transition distribution of the forward process satisfies:

$$
q_{t0}(\bm{x}^{(t)}|\bm{x}^{(0)})=\mathcal{N}(\bm{x}^{(t)}\mid\alpha_{t}\bm{x}%
^{(0)},\sigma_{t}^{2}\mathbf{I}),t\in[0,1],
$$

which gradually adds Gaussian noise to generate a series of noised data from $\bm{x}^{(0)}$ to $\bm{x}^{(t)}$ with $t\in[0,1]$. $\sigma_{t}>0$ is a variance term that controls the introduced noise and $\alpha_{t}>0$ is typically defined as $\alpha_{t}=\sqrt{1-\sigma_{t}^{2}}$, ensuring $\bm{x}^{(t)}\rightarrow\mathcal{N}(0,\mathbf{I})$, as $t\rightarrow 1$. The reversed denoising process of Eq. (1) can be equivalently expressed as a diffusion ODE [^40]:

$$
({\rm Diffusion\ ODE})\ \ {{\rm d}\bm{x}^{(t)}}=\left[f(t)\bm{x}^{(t)}-\frac{1%
}{2}g^{2}(t)\nabla_{\bm{x}^{(t)}}\log q_{t}(\bm{x}^{(t)})\right]{{\rm d}t},
$$

where $f(t)=\frac{{\rm d}\log\alpha_{t}}{{\rm d}t},g^{2}(t)=\frac{{\rm d\sigma_{t}^{2%
}}}{{\rm d}t}-2\frac{{\rm d}\log\alpha_{t}}{{\rm d}t}\sigma_{t}^{2}$ are determined by the fixed noise schedules $\alpha_{t},\sigma_{t}$, and $q_{t}$ is the marginal distribution of $\bm{x}^{(t)}$. Diffusion model utilizes a neural network $\bm{s}_{\theta}(\bm{x}^{(t)},t)$ to fit the probability score $\nabla_{\bm{x}^{(t)}}\log q_{t}(\bm{x}^{(t)})$. By learning the score function, diffusion models enjoy the strong expressiveness of modeling arbitrary complex distributions [^10], making it highly versatile and adaptable for challenging tasks such as autonomous driving.

Classifier Guidance. Classifier guidance [^15] is a technique used to generate preferred data by guiding the sampling process with a classifier $\mathcal{E}_{\phi}(\bm{x}^{(t)},t)$. The gradient of the classifier score is used to modify the original diffusion score:

$$
\tilde{\bm{s}}_{\theta}(\bm{x}^{(t)},t)=\bm{s}_{\theta}(\bm{x}^{(t)},t)-\nabla%
_{\bm{x}^{(t)}}\mathcal{E}_{\phi}(\bm{x}^{(t)},t)
$$

In autonomous driving, this approach offers greater flexibility compared to rule-based refinement because it directly improves the model’s inherent ability, rather than overly relying on sub-optimal post-processing that requires significant human effort and targeted data collection.

## 4 Methodology

In this section, we redefine the planning task as a future trajectory generation task, which jointly generates the ego vehicle’s planning and the prediction of neighboring vehicles. We then introduce the Diffusion Planner, a novel approach that leverages the expressive and flexible diffusion model for enhanced autonomous planning. Lastly, we demonstrate how the guidance mechanism in diffusion models can be utilized to align planning behavior with safe or human-preferred driving styles.

![Refer to caption](https://arxiv.org/html/2501.15564v2/x1.png)

Figure 1: Model architecture of Diffusion Planner.

### 4.1 Task Redefinition

Autonomous driving requires considering the close interaction between the ego and neighboring vehicles, resulting in a cooperative relationship between planning and motion prediction tasks [^34]. Supervising the future trajectories of neighboring vehicles has been shown to be helpful to enhance the ability of closed-loop planning models to handle complex interaction scenarios [^20]. For real-world deployment, motion prediction can also enhance safety by providing more controllable measures, facilitating the implementation of the system [^16]. Consequently, the trajectories of neighboring vehicles have become crucial privileged information for model training. However, the common approaches that use a dedicated sub module [^22] or additional loss design [^8] [^22] to capture privileged information limit their modeling power during training and also lead to a more complex framework.

In this work, we address this issue by collectively considering the status of key participants in the driving scenario and jointly modeling the motion prediction and closed-loop planning tasks as a future trajectory generation task. Specifically, given conditions $\bm{C}$, which include current vehicle states, historical data, lane information, and navigation information, our goal is to generate future trajectories for all key participants simultaneously, enabling the modeling of cooperative behaviors among them. However, this joint modeling of complex distributions is challenging to solve with a simple behavior cloning approach. Benefiting from the strong expressive power of diffusion models, we adopt a diffusion model for this task and formulate the target as:

$$
\bm{x}^{(0)}=\begin{bmatrix}x^{(0)}_{\mathrm{ego}}\\
x^{(0)}_{\mathrm{neighbor}_{1}}\\
\vdots\\
x^{(0)}_{\mathrm{neighbor}_{M}}\par\end{bmatrix}=\begin{bmatrix}x^{1}_{\mathrm%
{ego}}&x^{2}_{\mathrm{ego}}&...&x^{\tau}_{\mathrm{ego}}\\
x^{1}_{\mathrm{neighbor}_{1}}&x^{2}_{\mathrm{neighbor}_{1}}&...&x^{\tau}_{%
\mathrm{neighbor}_{1}}\\
\vdots&\vdots&\ddots&\vdots\\
x^{1}_{\mathrm{neighbor}_{M}}&x^{2}_{\mathrm{neighbor}_{M}}&...&x^{\tau}_{%
\mathrm{neighbor}_{M}}\\
\end{bmatrix},
$$

where we use superscripts with parentheses to represent the timeline of diffusion denoising, and regular superscripts to indicate the timeline of the future trajectory, which contains $\tau$ time steps. For each state $x$, we only consider the coordinates and the sine and cosine of the heading angle, which are sufficient for the downstream LQR controller. We select the nearest $M$ neighboring vehicles to predict their possible future trajectories. By parameterizing our Diffusion Planner with $\theta$, the training target can be expressed as:

$$
\mathcal{L}_{\theta}=\mathbb{E}_{\bm{x}^{(0)},t\sim\mathbb{U}(0,1),\bm{x}^{(t)%
}\sim q_{t0}(\bm{x}^{(t)}|\bm{x}^{(0)})}\left[||\mu_{\theta}(\bm{x}^{(t)},t,%
\bm{C})-\bm{x}^{(0)}||^{2}\right],
$$

where the goal is to recover the data distribution from noisy data [^36]. We can get the score function as $\bm{s}_{\theta}=(\alpha_{t}\mu_{\theta}-\bm{x}^{(t)})/\sigma_{t}^{2}$ and apply it during the denoising process. The joint prediction of multiple vehicles is similar to motion prediction [^24] and traffic simulation [^54] [^53] tasks, but we focus more on the ego vehicle’s closed-loop planning performance and real-time deployment. We will introduce the specific designs as follows.

### 4.2 Diffusion Planner

Diffusion Planner is a model based on the DiT architecture [^35], with a core design focusing on the fusion mechanism between noised future vehicle trajectories $\bm{x}$ and conditional information $\bm{C}$. Figure 1 provides an overview of the complete architecture. A detailed description of these interaction and fusion modules is provided as follows.

Vehicle Information Integration. In the first step, the future vehicle trajectory $\bm{x}$ is concatenated with the current state of each vehicle, represented as $x^{0}=[x^{0}_{\mathrm{ego}},x^{0}_{\mathrm{neighbor1}},\dots,x^{0}_{\mathrm{%
neighbor_{M}}}]^{T}$. This concatenation acts as a constraint to guide the model, simplifying the planning task by providing a clear starting point. Notably, velocity and acceleration information for the ego vehicle is excluded, which has been shown to enhance closed-loop performance, as highlighted in previous works [^8] [^27]. Integration of the information from different vehicles during model execution is achieved through multi-head self-attention mechanisms.

Historical Status and Lane Information Fusion. The historical status of neighboring vehicles and lane information is represented using vectors [^17]. Specifically, each neighboring vehicle is represented as $\bm{S}_{\mathrm{neighbor}}\in\mathbb{R}^{L\times D_{\mathrm{neighbor}}}$, and lanes as $\bm{S}_{\mathrm{lane}}\in\mathbb{R}^{P\times D_{\mathrm{lane}}}$, where $L$ refers to the number of past timestamps, and $P$ indicates the number of points per polyline. $D_{\mathrm{neighbor}}$ contains data such as vehicle coordinates, heading, velocity, size, and category, while $D_{\mathrm{lane}}$ provides lane details such as coordinates, traffic light status, and speed limits. Since these vectors are information-sparse, directly fusing them would make training challenging. To address this, we use MLP-Mixer network [^45] to extract information-dense representations. Compared to existing work [^22] [^8] that uses complex structural designs, we offer a more unified and simplified solution. This is achieved by iteratively passing the vectors through the MLP mixing layers, which operate on both the vector and feature dimensions. The forward process of each mixing layer can be formulated as follows:

$$
\bm{S}=\bm{S}+{\mathrm{MLP}(\bm{S}^{T})}^{T},\bm{S}=\bm{S}+\mathrm{MLP}(\bm{S})
$$

We use two separate MLP-Mixer networks for neighboring vehicles and lanes. Here, $\bm{S}$ represents the features for each neighboring vehicle or lane. After passing through multiple mixing layers, we apply pooling on the final output along the vector dimension. We also consider the static objects information $\bm{S}_{\mathrm{static}}\in\mathbb{R}^{D_{\mathrm{static}}}$, where $D_{\mathrm{static}}$ includes data such as coordinates, heading, size, and category. For this, we use an MLP to extract the representation. Finally, we concatenate all representations and feed them into a vanilla transformer encoder for further aggregation, resulting in the encoder representation $\bm{Q}_{f}$. The fusion of $\bm{Q}_{f}$ with $\bm{x}$ proceeds as follows: f

$$
\bm{x}=\bm{x}+\mathrm{MHCA}(\bm{x},\bm{Q}_{f}),\bm{x}=\bm{x}+\mathrm{FFN}(\bm{%
x}),
$$

where MHCA donate multi-head cross-attention.

Navigation Information Fusion. Navigation information is crucial for autonomous driving planning, as it provides essential guidance on the intended route, enabling the vehicle to make informed decisions. In the nuPlan benchmark [^5], navigation information is represented as a set of lanes along a route, $\bm{S}_{\mathrm{route}}\in\mathbb{R}^{(K\times P)\times D_{\mathrm{route}}}$, where $K$ denotes the number of route lanes, and $D_{\mathrm{route}}$ contains only coordinate information. We first employ an MLP-Mixer network, as described in equation 6, to extract the essential guidance representations $\bm{Q}_{n}$. $\bm{Q}_{n}$ is then added to the diffusion timestep condition $\bm{Q}_{t}$ and applied through an adaptive layer norm block [^35] to guide trajectory generation across all tokens.

### 4.3 Planning Behavior alignment via Classifier Guidance

![Refer to caption](https://arxiv.org/html/2501.15564v2/x2.png)

Figure 2: Starting from the same position, the trajectories driven under different guidance settings.

Enforcing versatile and controllable driving behavior is crucial for real-world autonomous driving. For example, vehicles must ensure safety and comfort while adjusting speeds to align with user preferences. Thanks to its close relationship to Energy-Based Models [^30], diffusion model can conveniently inject such preferences via classifier guidance. It can steer the model outputs via gradient surgery during inference, offering significant potential for customized adaptation.

Specifically, given the original driving behavior $q_{0}(\bm{x}^{(0)})$, we aim to encode additional guidance to reinforce some preferred behavior upon the existing behavior $q_{0}$. This operation can be formulated as generating a target behavior: $p_{0}(\bm{x}^{(0)})\propto q_{0}(\bm{x}^{(0)})e^{-\mathcal{E}(\bm{x}^{(0)})}$, where $\mathcal{E}(\bm{x}^{(0)})$ can be some form of energy function that encodes safety or preferred behavior. As mentioned in Section 3.2, the gradient of the intermediate energy [^30] is employed to adjust the original probability score, promoting the generation of trajectories within the target distribution. This process often necessitates an additional trained classifier to provide an accurate approximation. However, diffusion posterior sampling [^12] [^50] offers a training free method that only uses the trained diffusion model $\mu_{\theta}$ in Eq. (5) to approximate the guidance energy, bypassing the classifier training, which incurs additional computational overhead:

$$
\displaystyle\nabla_{\bm{x}^{(t)}}\log p_{t}(\bm{x}^{(t)})
$$
 
$$
\displaystyle\approx\nabla_{\bm{x}^{(t)}}\log q_{t}(\bm{x}^{(t)})-\nabla_{\bm{%
x}^{(t)}}\mathcal{E}\left(\mathbb{E}_{q_{0t}(\bm{x}^{(0)}|\bm{x}^{(t)})}[\bm{x%
}^{(0)}]\right)
$$
 
$$
\displaystyle=\nabla_{\bm{x}^{(t)}}\log q_{t}(\bm{x}^{(t)})-\nabla_{\bm{x}^{(t%
)}}\mathcal{E}\left(\mu_{\theta}(\bm{x}^{(t)},t,\bm{C})\right).
$$

One restriction of this method is that Eq. (8) needs to use a pre-defined differentiable energy function $\mathcal{E}(\cdot)$ to calculate the guidance energy. Fortunately, in autonomous driving scenarios, many trajectory evaluation protocols can be defined using differentiable functions. Next, we briefly describe some applicable energy functions that can be used to customize the planning behavior of the model, more details are shown in Appendix C.3.

- Target speed maintenance: The speed difference is used as the energy, calculated by comparing the planned average speed with the set target speed.
- Comfort: The energy function is calculated by measuring the amount by which the vehicle’s state exceeds the predefined limits.
- Collision avoidance: The signed distance between the ego vehicle and neighboring vehicles is computed at each timestamp.
- Staying within drivable area: The distance the ego vehicle deviates outside the lane at each time step is calculated.

Additionally, this training-free approach supports flexible combinations during inference time, providing a solution for controllable trajectory generation in complex scenarios. For example, as shown in Figure 2, under collision guidance alone, the ego vehicle veers off the road to avoid a rear-approaching vehicle. However, when drivable guidance is added, the vehicle stays on the road while maintaining safety. For more case studies, please refer to Section 5.1 and the Appendix B.2.

### 4.4 Practical Implementation for Closed-Loop Planning

Data augmentation can help alleviate the out-of-distribution issue and is widely used in planning. Before training, we add random perturbations to the current state [^8]. Then, interpolation is applied to create a physically feasible transition, enabling the model to resist perturbations and regress to the ground-truth trajectory [^2]. After that, we transform the data from the global coordinate system into an ego-centric formulation through coordinate transformation. Considering the significant difference between the longitudinal and lateral distances traveled by the vehicle, z-score normalization is used to ensure the mean of the data distribution is close to zero, thereby further stabilizing the training process. During inference, DPM-Solver [^29] is employed to achieve faster sampling, while low-temperature sampling [^1] enhances determinism in the planning process. We can complete trajectory planning for the next $8$ seconds at $10$ Hz, along with predictions for neighboring vehicles, with an inference frequency of approximately $20$ Hz. Please see Appendix C for implementation details.

![[Uncaptioned image]](https://arxiv.org/html/2501.15564v2/x3.png)

Table 1: Closed-loop planning results on nuPlan dataset. a: The highest scores of baselines in various types. \*: Using pre-searched reference lines as model input provides prior knowledge, reducing the difficulty of planning compared to standard learning-based methods. NR: non-reactive mode. R: reactive mode.

## 5 Experiments

Evaluation Setups. We conduct extensive evaluations on the large-scale real-world autonomous planning benchmark, nuPlan [^5], to compare Diffusion Planner with other state-of-the-art planning methods. The Val14 [^14], Test14, and Test14-hard benchmarks [^8] are utilized, with all experimental results tested in both closed-loop non-reactive and reactive modes. The final score is calculated as the average across all scenarios, ranging from $00$ to $100$, where a higher score indicates better algorithm performance. To further validate the algorithm’s performance across diverse driving scenarios and with vehicles exhibiting different driving behaviors, we collected 200 hours of real-world data using a delivery vehicle from Haomo.AI. Unlike nuPlan, the delivery vehicle demonstrates more conservative planning behavior and operates in bike lanes, which involve dense human-vehicle interactions and unique traffic regulations. The collected data were integrated into the nuPlan framework, and the same evaluation metrics were applied in closed-loop simulations, as detailed in Appendix D.

Baselines. The baselines are categorized into three groups [^14]: Rule-based, Learning-based, and Hybrid, which incorporate additional refinement to the outputs of the learning-based model. To enable a more comprehensive comparison, we utilize an existing refinement module [^43], which applies offsets to the model outputs and scores all trajectories [^14]. Without any parameter tuning, we integrate this module as post-processing for the Diffusion Planner (Diffusion Planner w/ refine.). We compare the Diffusion Planner against the following baselines, with more implementation details provided in Appendix C.4.

- IDM [^47]: A classic rule-based method implemented by nuPlan.
- PDM [^14]: The first-place winner of the nuPlan challenge offers a rule-based version that follows the centerline (PDM-Closed), a learning-based version conditioned on the reference line (PDM-Open), and a hybrid approach that combines both (PDM-Hybrid).
- UrbanDriver [^37]: A learning-based method using policy gradient optimization and implemented by nuPlan.
- GameFormer [^22]: Modeling ego and neighboring vehicle interactions using game theory (GameFormer w/o refine.), followed by rule-based refinement.
- PlanTF [^8]: A state-of-the-art learning-based method built on a transformer architecture, exploring various designs suitable for closed-loop planning.
- PLUTO [^9]: Building on PDM-Open, a complex model with contrastive learning enhances environmental understanding (PLUTO w/o refine.), followed by post-processing.

Main Results. Evaluation results on the nuPlan benchmark are presented in Table 2. The Diffusion Planner achieves state-of-the-art performance across more benchmarks compared to all learning-based baselines. With the addition of post-processing, Diffusion Planner w/ refine. outperforms hybrid and rule-based baselines, achieving scores that even surpass human performance. This is due to our model’s ability to output high-quality trajectories, which are further enhanced by post-processing. Notably, compared to the transformer-based PlanTF and PLUTO, Diffusion Planner leverages the power of diffusion to achieve better performance. GameFormer, which models the interactions between the ego vehicle and neighboring vehicles using game theory, exhibits limited model capabilities, making it overly reliant on rule-based refinements. We further present the planning results on delivery-vehicle driving dataset as shown in Table 2. PDM, GameFormer, and PLUTO include certain designs specifically tailored to the nuPlan benchmark, which limits their ability to transfer to delivery-vehicle driving tasks, resulting in a drop in performance. In contrast, Diffusion Planner demonstrates strong transferability across different driving behaviors. We also compared works that utilize diffusion for planning, as shown in Table 4 in Appendix B.1. The Diffusion Planner better leverages the powerful capabilities of diffusion and is more practical.

Qualitative Results. To further demonstrate the capabilities of learning-based models, we show the trajectory generation results of representative baselines (without refinement) as shown in Figure 3. Diffusion Planner shows high-quality trajectory generation, with accurate predictions for neighboring vehicles and smooth ego planning trajectories that reasonably account for the speed of the vehicle ahead, demonstrating the advantages of joint modeling of both prediction and planning tasks. More closed-loop planning results are shown in Appendix A. In contrast, GameFormer w/o refine produces less smooth trajectories and inaccurate predictions for neighboring vehicles, which explains why it heavily relies on refinement. Although PlanTF and PLUTO w/o refine. sample multiple trajectories at once, most of them are of low quality.

### 5.1 Empirical Studies of Diffusion Planner Properties

![Refer to caption](https://arxiv.org/html/2501.15564v2/extracted/6190205/fig/guidance.png)

Figure 4: Target speed and comfort guidance: For target speed guidance, the speed changes before and after guidance are visualized. For comfort guidance, the longitudinal jerk changes are compared before and after applying comfort guidance on top of collision avoidance guidance.

![Refer to caption](https://arxiv.org/html/2501.15564v2/extracted/6190205/fig/multimodel.png)

Figure 5: Multi-modal planning behavior of Diffusion Planner.

Multi-modal Planning Behavior. We selected an intersection scenario and performed multiple inferences without low temperature sampling from the same initial position to obtain different possible outputs, in order to evaluate the model’s ability to fit multi-modal driving behaviors. As shown in Figure 5, without navigation information, the vehicle can exhibit three distinct driving behaviors—left turn, right turn, and straight ahead—with clear differentiation. When navigation information is provided, the model accurately follows it to make a left turn, demonstrating the diffusion model’s ability to fit driving behaviors with varying distributions and its capacity for switching between them.

Flexible guidance mechanism. Based on the trained Diffusion Planner model, different types of classifier guidance, as described in Section 4.3, are added during inference time without requiring additional training. We present two cases to demonstrate the effectiveness of guidance and its flexible composability, as shown in Figure 4. 1) For target speed setting, we masked all lane speed limit information to prevent it from influencing the model’s planning, ensuring that speed adjustments are made solely through guidance. As a result, the model exhibited a lower speed without guidance. By setting the speed between $10\text{m/s}$ and $14\text{m/s}$, the model closely matches the desired speed range while maintaining smooth speed transitions. 2) For comfort guidance, we effectively alleviate discomfort and can even use it simultaneously with collision guidance. We also provide additional case studies on collision and drivable guidance, as shown in Appendix B.2, as well as cases demonstrating the flexible combination of collision and drivable guidance, as illustrated in Figure 2.

### 5.2 Ablation Studies

![Refer to caption](https://arxiv.org/html/2501.15564v2/extracted/6190205/fig/predictneighbor.png)

Table 3: Ablation of each modules during the training process on nuPlan Test14 Benchmark.

Design Choices for training. We demonstrate the effectiveness of key components of our method: data processing, the handling approach of ego current state, and the number of predicted vehicles. 1) We ablate the model’s performance without using z-score normalization (w/o z-score norm), as well as without data augmentation (w/o augmentation), or by only perturbing the current state without applying interpolation to future trajectories (w/o interpolation). The results are summarized in Table LABEL:tab:ablationtraining. For the w/o z-score norm variant, even with ego-centric transformation, the data range remains large, making it difficult for the model to fit the distribution. The w/o augmentation variant faces out-of-distribution issues, leading to poor performance. Results also show that future trajectory interpolation is essential compared to perturbing only the current state. 2) We analyze the impact of the ego vehicle’s current state on the model. Retaining velocity, acceleration, and yaw rate (w/ ego state) may lead to learning shortcuts, resulting in decreased planning capability. While a state dropout encoder [^8] (w/ SDE) mitigates this, directly discarding the information is more effective. Additionally, the w/o current state shows that adding current state information to the decoder improves planning capability. 3) We also ablate the choice of the number of $M$. Figure 7 shows that including too many neighboring vehicles in the decoder introduces noise, affecting the performance of the ego vehicle. However, most choices still outperform PlanTF.

Design Choices for Inference. We sweep two hyperparameters: the number of denoise steps and the magnitude of low-temperature sampling, as shown in Figure 7. Low temperature helps improve the stability of the output trajectories. Additionally, the model leverages DPM-Solver to achieve efficient denoising and remains robust across different step counts. We report the detailed parameter selection in Table 5.

## 6 Conclusion

We propose Diffusion Planner, a learning-based approach that fully exploits the expressive power and flexible guidance mechanism of diffusion models for high-quality autonomous planning. A transformer-based architecture is introduced to jointly model the multi-modal data distribution in motion prediction and planning tasks through a diffusion objective. Classifier guidance is employed to align planning behavior with safe or user preferred driving styles. Diffusion Planner achieves state-of-the-art closed-loop performance without relying on any rule-based refinement on the nuPlan benchmark and a newly collected 200-hour delivery-vehicle driving dataset, demonstrating strong adaptability across diverse driving styles. Due to space limit, more discussion on limitations and future direction can be found in Appendix E.

## Acknowledgement

This work is supported by National Key Research and Development Program of China under Grant (2022YFB2502904), and funding from Haomo.AI.

## References

## Appendix A Visualization of closed-loop planning results

![Refer to caption](https://arxiv.org/html/2501.15564v2/extracted/6190205/fig/appendix/closedloop.png)

Figure 8: Closed-loop planning results: each row represents a scenario at 0, 5, 10, and 15 seconds intervals. Each frame includes the future planning of the ego vehicle, predictions for neighboring vehicles, the ground truth ego trajectory, and the driving history of the ego vehicle.

## Appendix B Additional results

### B.1 Compared to diffusion-based planning methods

To further demonstrate the advantages of our model, we compared it with two recent works using diffusion models for motion planning. Diffusion-es [^51] enhances a diffusion model by incorporating an LLM as a trajectory filter. STR-16M [^42] uses a diffusion model as a decoder. STR2-CPKS-800M [^43] builds on the former with 800M parameters and includes a PDM-like refinement module. We compared the model’s performance in non-reactive mode and recorded the inference time, as shown in Table 4. We observe that current diffusion-based methods also experience significant performance degradation when detached from LLMs or rule-based refinement. Another important point is that these methods, due to their reliance on LLMs or a large number of model parameters, have higher computational costs, making them difficult to deploy in real-world applications.

Table 4: Closed-loop non-reactive planning results on the nuPlan dataset among diffusion-based planners.

| Planner | Test14 | Test14-hard | Val14 | Inference Time (s) |
| --- | --- | --- | --- | --- |
| Diffusion-es w/o LLM | \- | \- | 50 | \- |
| Diffusion-es w/ LLM | \- | \- | 92 | 0.5 |
| STR-16M | \- | 27.59 | 45.06 | \- |
| STR2-CPKS-800M w/o refine. | 68.74 | 52.57 | 65.16 | $>$ 11 |
| Diffusion Planner (ours) | 89.19 | 75.99 | 89.87 | 0.04 |

### B.2 More case studies for the guidance mechanism.

![Refer to caption](https://arxiv.org/html/2501.15564v2/extracted/6190205/fig/appendix/guidancecase.png)

Figure 9: Case studies for collision and drivable guidance. Starting from the same position, we visualized the closed-loop test results: the dashed line represents the results without guidance, with hollow car markers indicating locations where safety incidents occurred. The solid line represents the results with guidance, and the solid car markers indicate the final positions.

## Appendix C Experimental details

This section outlines the experimental details to reproduce the main results in our papers.

### C.1 Training Details

Datasets. We use the training data from the nuPlan <sup>\par</sup> dataset and sample $1$ million scenarios for our training set. The number of different scenarios is shown in Figure 11. For each scenario, we consider the lane and navigation information within a $100m$ radius around the ego vehicle at the current time, including the neighboring vehicles’ history from the past two seconds. Each type of data is padded to a unified dimension for model input, and attention masking is used to effectively eliminate irrelevant information.

Data augmentation. The current state of the ego vehicle is first perturbed slightly in terms of its $x$, $y$ coordinates, orientation angle $\theta$, speed $v$, acceleration $a$.

$$
\Delta{x^{0}}\sim\mathbbm{U}\left([-\Delta{x},-\Delta{y},-\Delta{\theta},-%
\Delta{v},-\Delta{a}],[\Delta{x},\Delta{y},\Delta{\theta},\Delta{v},\Delta{a}]%
\right).
$$

For the augmented state $\tilde{x}_{ego}^{0}=x_{ego}^{0}+\Delta{x^{0}}$, we ensure that the speed $v$ always remains greater than 0 to prevent the vehicle from learning to move in reverse. After that, a quintic polynomial interpolation is applied between current state $\tilde{x}_{ego}^{0}$ and $x_{ego}^{\tau_{2s}}$ to generate a new trajectory that adheres to the dynamic constraints, replacing the ground truth trajectory.

Normalization. Following previous works [^22] [^8] [^9], we apply an ego-centric transformation to process the original dataset. The global coordinates are converted into the ego vehicle’s local coordinate system, using the vehicle’s heading and position. Afterward, we observe that the ego vehicle’s longitudinal progress is significantly larger than its lateral progress. To improve training stability, we apply z-score normalization to all x-axis coordinates, while the y-axis is scaled to the same magnitude to avoid distortion:

$$
\tilde{x}=\frac{x-\mu}{\sigma},\quad\tilde{y}=\frac{y}{\sigma},
$$

where $\mu=10$, $\sigma=20$. The same approach is applied other scenario inputs.

Training was conducted using 8 NVIDIA A100 80GB GPUs, with a batch size of $2048$ over $500$ epochs, with a $5$ -epoch warmup phase. We use AdamW optimizer with a learning rate of $5e^{-4}$. We report the detailed setup in Table 5.

### C.2 Inference Details

We utilize DPM-Solver++ as diffusion reverse process solver, adopting variance-preserving(VP) noise schedule where the noise is $\sigma_{t}=(1-t)\beta_{\mathrm{min}}+t\beta_{\mathrm{max}}$. Low-temperature sampling is employed to further enhance the stability of the denoising process. We found that directly using the model output with a higher temperature facilitates generating high-quality trajectories. Conversely, if a refinement module is applied after the model output, a lower temperature helps produce more stable trajectories, which supports more accurate judgments by the refinement module. In addition, the model achieves an inference frequency of 20 Hz on a single A6000 GPU. We also report the detailed setup in Table 5.

### C.3 Classifier Guidance Details

We then specifically introduce the mathematical formulation of the different energy functions, as mentioned in Section 4.3.

Collision Avoidance. Based on the ego vehicle’s planning and the neighboring vehicles’ predictions from the decoder at diffusion timestamp $t$, we calculate the signed distance $\mathbf{D}$ between the ego vehicle and each neighboring vehicle at each timestamp $\tau$. When the bounding boxes of the vehicles overlap, we use the minimum separation distance, otherwise, we use the distance between the nearest points. The energy function for collision avoidance is then defined as:

$$
\displaystyle\mathcal{E}_{\mathrm{collision}}=
$$
 
$$
\displaystyle\frac{1}{\omega_{\mathrm{c}}}\cdot\frac{\sum\limits_{M,\tau}{%
\mathbbm{1}_{\mathbf{D}_{M}^{\tau}>0}\cdot\Psi\left(\omega_{\mathrm{c}}\cdot%
\mathrm{max}\left(1-\frac{\mathbf{D}_{M}^{\tau}}{r},0\right)\right)}}{\sum%
\limits_{M,\tau}{\mathbbm{1}_{\mathbf{D}_{M}^{\tau}>0}}+\mathrm{eps}}
$$
 
$$
\displaystyle+
$$
 
$$
\displaystyle\frac{1}{\omega_{\mathrm{c}}}\cdot\frac{\sum\limits_{M,\tau}{%
\mathbbm{1}_{\mathbf{D}_{M}^{\tau}<0}\cdot\Psi\left(\omega_{\mathrm{c}}\cdot%
\mathrm{max}\left(1-\frac{\mathbf{D}_{M}^{\tau}}{r},0\right)\right)}}{\sum%
\limits_{M,\tau}{\mathbbm{1}_{\mathbf{D}_{M}^{\tau}<0}}+\mathrm{eps}},
$$

where $\Psi(x):=e^{x}-x$, $r$ represents the collision-sensitive distance, which controls the maximum distance at which gradients are produced, and $\mathrm{eps}$ is added to ensure numerical stability [^24].

Target Speed Maintenance. We calculate the energy function based on the difference between the average speed of the generated trajectory and the target speed range:

$$
\mathcal{E}_{\mathrm{target\_speed}}=\max\left(\overline{\frac{\mathrm{d}x^{%
\tau}_{\mathrm{ego}}}{\mathrm{d}\tau}}-v_{\mathrm{low}},0\right)^{2}+\max\left%
(v_{\mathrm{high}}-\overline{\frac{\mathrm{d}x^{\tau}_{\mathrm{ego}}}{\mathrm{%
d}\tau}},0\right)^{2}.
$$

Where $v_{\mathrm{low}}$ is the setting lower bound of speed, $v_{\mathrm{low}}$ is the setting higher bound of speed.

Comfort. Taking longitudinal jerk as an example, the difference between each point and the comfort threshold is calculated, ignoring cases where the comfort requirements are met:

$$
\mathcal{E}_{\mathrm{comfort}}=\mathbbm{E}\left[\max\left(\left(j_{\mathrm{max%
}}-\left|\frac{\mathrm{d}^{3}x^{\tau}_{\mathrm{ego}}}{\mathrm{d}\tau^{3}}%
\right|\right)\Delta\tau^{3},0\right)^{2}\right].
$$

Where $j_{\mathrm{max}}$ is the maximum longitude jerk limit.

Staying within Drivable Area. We construct the differentiable cost map $\mathbf{M}$ by using Euclidean Signed Distance Field with parallel computation [^9], which can compute the distance the ego vehicle goes beyond the lane at each timestamp. Then the energy is defined as:

$$
\mathcal{E}_{\mathrm{drivable}}=\frac{1}{\omega_{\mathrm{d}}}\cdot\frac{\sum%
\limits_{\tau}{\Psi\left(\omega_{\mathrm{d}}\cdot\mathbf{M}\left(x^{\tau}_{%
\mathrm{ego}}\right)\right)}}{\sum\limits_{\tau}{\mathbbm{1}_{\mathbf{M}\left(%
x^{\tau}_{\mathrm{ego}}\right)>0}}+\mathrm{eps}}.
$$

Given the diverse options for energy function design, our choices were made primarily to validate whether the model could support various types of guidance and may not be optimal. However, through extensive empirical experiments, we can share some of our insights and experiences regarding energy function selection to assist future work in exploring more effective options:

- Smooth and continuous gradients: Guidance functions with smooth and continuous gradients facilitate the generation of stable trajectories.
- Gradient sparsity: It is preferable for the guidance function to generate gradients only in specific situations, such as when trajectory points approach potential collisions.
- Indirect guidance for higher-order state derivatives: For higher-order state derivatives, such as velocity, acceleration, or angular velocity, indirect guidance through position and heading is preferable. For instance, to control trajectory speed, we can guide trajectory length instead.
- Consistent gradient magnitude: The guidance function should ensure that the magnitude of gradients remains approximately consistent across different conditions. It can be achieved by averaging cost values over the number of points contributing to the cost.

Table 5: Hyperparameters of Diffusion Planner

<table><tbody><tr><td>Type</td><td>Parameter</td><td>Symbol</td><td>Value</td></tr><tr><td rowspan="11">Training</td><td>Num. neighboring vehicles</td><td>-</td><td>32</td></tr><tr><td>Num. past timestamps</td><td><math><semantics><mi>L</mi> <ci>𝐿</ci> <annotation>L</annotation> <annotation>italic_L</annotation></semantics></math></td><td>21</td></tr><tr><td>Dim. neighboring vehicles</td><td><math><semantics><msub><mi>D</mi> <mi>neighbor</mi></msub> <apply><csymbol>subscript</csymbol> <ci>𝐷</ci> <ci>neighbor</ci></apply> <annotation>D_{\mathrm{neighbor}}</annotation> <annotation>italic_D start_POSTSUBSCRIPT roman_neighbor end_POSTSUBSCRIPT</annotation></semantics></math></td><td>11</td></tr><tr><td>Num. lanes</td><td>-</td><td>70</td></tr><tr><td>Num. points per polyline</td><td><math><semantics><mi>P</mi> <ci>𝑃</ci> <annotation>P</annotation> <annotation>italic_P</annotation></semantics></math></td><td>20</td></tr><tr><td>Dim. lanes vehicles</td><td><math><semantics><msub><mi>D</mi> <mi>lane</mi></msub> <apply><csymbol>subscript</csymbol> <ci>𝐷</ci> <ci>lane</ci></apply> <annotation>D_{\mathrm{lane}}</annotation> <annotation>italic_D start_POSTSUBSCRIPT roman_lane end_POSTSUBSCRIPT</annotation></semantics></math></td><td>12</td></tr><tr><td>Num. navigation lanes</td><td><math><semantics><mi>D</mi> <ci>𝐷</ci> <annotation>D</annotation> <annotation>italic_D</annotation></semantics></math></td><td>25</td></tr><tr><td>Num. predicted neighboring vehicles</td><td><math><semantics><mi>M</mi> <ci>𝑀</ci> <annotation>M</annotation> <annotation>italic_M</annotation></semantics></math></td><td>10</td></tr><tr><td>Num. encoder/decoder block</td><td>-</td><td>3</td></tr><tr><td>Dim. hidden layer</td><td>-</td><td>192</td></tr><tr><td>Num. multi-head</td><td>-</td><td>6</td></tr><tr><td rowspan="4">Inference</td><td>Noise schedule</td><td>-</td><td>Linear</td></tr><tr><td>Noise coefficient</td><td><math><semantics><mrow><msub><mi>β</mi> <mi>min</mi></msub><mo>,</mo><msub><mi>β</mi> <mi>max</mi></msub></mrow> <list><apply><csymbol>subscript</csymbol> <ci>𝛽</ci> <ci>min</ci></apply> <apply><csymbol>subscript</csymbol> <ci>𝛽</ci> <ci>max</ci></apply></list> <annotation>\beta_{\mathrm{min}},\beta_{\mathrm{max}}</annotation> <annotation>italic_β start_POSTSUBSCRIPT roman_min end_POSTSUBSCRIPT, italic_β start_POSTSUBSCRIPT roman_max end_POSTSUBSCRIPT</annotation></semantics></math></td><td>0.1, 20.0</td></tr><tr><td>Temperature</td><td>-</td><td>0.5</td></tr><tr><td>Temperature (w/ refine.)</td><td>-</td><td>0.1</td></tr><tr><td></td><td>Denoise step</td><td>-</td><td>10</td></tr></tbody></table>

### C.4 Baselines setup

nuPlan Datasets Evaluation. For IDM and UrbanDriver, we use the official nuPlan code <sup>\par</sup>, with the UrbanDriver checkpoint sourced from the PDM codebase <sup>\par</sup>, which also provides the checkpoints for PDM-Hybrid and PDM-Open. For PlanTF and PLUTO, we use the checkpoints from their respective official codebases <sup>\par</sup> <sup>\par</sup>. In the case of PLUTO w/o refine, we skip the post-processing code and rerun the simulation without retraining. Following the guidelines from the official codebase <sup>\par</sup>, we train GameFormer and skip the refinement step to obtain GameFormer w/o refine.

Delivery-vehicle Datasets Evaluation. We adopt the same metrics and models as those used on nuPlan, but by modifying various vehicle-related parameters to adapt the baselines to the delivery-vehicle training. Based on this, we retrain and test the models following the official training code.

## Appendix D Details on delivery vehicle experiments

We collected approximately $200$ hours of real-world data using an autonomous logistics delivery vehicle from Haomo.AI. The task of the delivery vehicle is similar to that of a robotaxi in nuPlan, as it autonomously navigates a designated route. During operation, the vehicle must comply with traffic regulations, ensure safety, and complete the delivery as efficiently as possible. Compared to the vehicles in the nuPlan dataset, the delivery vehicle is smaller, as shown in Table 6, and operates at lower speeds. As a result, it is able to travel on both main roads and bike lanes. During deliveries, it frequently interacts with pedestrians and cyclists, and the driving rules differ from those for motor vehicles, as shown in 10. This dataset serves as a supplement to nuPlan, allowing for the evaluation of algorithm performance under diverse driving scenarios.

Table 6: Vehicle parameter details

| Parameter (m) | Delivery Vehicle | nuPlan Vehicle |
| --- | --- | --- |
| Width | 1.03 | 2.30 |
| Length | 2.34 | 5.18 |
| Height | 1.65 | 1.78 |
| Wheel base | 1.20 | 3.09 |

Specifically, we transform the original data into the nuPlan data structure, allowing it to be stored as DB files compatible with the nuPlan API for seamless integration and usage. We use the same training pipeline from the nuPlan benchmark to train both the model and baselines. For some baselines that require crosswalk information, we replace it with stop line data. Additionally, the vehicle parameters are substituted with those of the delivery vehicle. The model’s performance is evaluated using the nuPlan metrics.

![Refer to caption](https://arxiv.org/html/2501.15564v2/x4.png)

Figure 10: Scenario count by type in the delivery-vehicle driving dataset, with representative visualizations.

## Appendix E Limitations & Discussions & Future Work

Here, we discuss our limitations, potential solutions and interesting future works.

- Scenario Inputs. Our method relies on vectorized map information and detection results of neighboring vehicles. Compared to mainstream end-to-end pipelines, this approach involves some information loss and requires a data processing module. However, unlike end-to-end methods, our focus is more on the planning stage, particularly on the ability for closed-loop planning.
	Solution and future work: We demonstrate the performance of the diffusion model for closed-loop planning without rule-based refinement. An interesting future direction would be to modify the encoder architecture and use images as inputs, enabling an end-to-end training pipeline.
- Lateral Flexibility. We find that learning-based methods struggle with flexibility, particularly when significant lateral movement is required. In contrast, rule-based methods perform better in this aspect due to the provision of a reference trajectory. Being consistent with findings from previous work [^27], we find this is mostly because that the dataset mainly consists of straight-driving scenarios, with few instances of lane changes or avoidance maneuvers. This makes it challenging for learning-based methods to generalize and acquire these skills. Additionally, since the model only outputs the planned trajectories instead of the controlling signal such as brake and throttle, there is a gap between the planned trajectory and the results from the downstream controller [^8]. This discrepancy also leads to potential poor performance, or even out-of-distribution behavior, in scenarios that require more flexible actions.
	Solution and future work: We find that data augmentation can somewhat alleviate the issue of the vehicle being reluctant to make lateral movements, but it still performs poorly in cases requiring significant lane changes. This could be improved by incorporating more data involving large lateral progress, leveraging reinforcement learning with a reward mechanism, or designing a more effective diffusion guidance mechanism to help the model learn lane-changing behaviors. We believe this is an interesting observation and leave this direction for future work.
- Sample Efficiency. The high performance of Diffusion comes at the cost of requiring multiple model inferences, leading to reduced sample efficiency.
	Solution and future work: We addressed this issue to a large extent by using a high-order ODE solver, enabling trajectory planning for 8 seconds at 10 Hz in 0.05 seconds. Considering real-world application requirements, techniques such as consistency models [^41] or distillation-based sampling methods [^31] could be employed for further acceleration.

Overall, although some design choices may appear simple and certain limitations exist, we have thoroughly demonstrated the capabilities of diffusion models for closed-loop planning in autonomous driving through extensive experiments. Moreover, we demonstrate the potential of the diffusion model to align with safety or human-preferred driving behaviors. It provides a high-performance, highly adaptable planner for autonomous driving systems.

![Refer to caption](https://arxiv.org/html/2501.15564v2/extracted/6190205/fig/appendix/dataset.png)

Figure 11: Scenario count by type in the nuPlan dataset.

[^1]: Anurag Ajay, Yilun Du, Abhi Gupta, Joshua B Tenenbaum, Tommi S Jaakkola, and Pulkit Agrawal. Is conditional generative modeling all you need for decision making? In *The Eleventh International Conference on Learning Representations*, 2022.

[^2]: Mayank Bansal, Alex Krizhevsky, and Abhijit Ogale. Chauffeurnet: Learning to drive by imitating the best and synthesizing the worst. *arXiv preprint arXiv:1812.03079*, 2018.

[^3]: Mariusz Bojarski, Davide Del Testa, Daniel Dworakowski, Bernhard Firner, Beat Flepp, Prasoon Goyal, Lawrence D Jackel, Mathew Monfort, Urs Muller, Jiakai Zhang, et al. End to end learning for self-driving cars. *arXiv preprint arXiv:1604.07316*, 2016.

[^4]: Holger Caesar, Varun Bankiti, Alex H. Lang, Sourabh Vora, Venice Erin Liong, Qiang Xu, Anush Krishnan, Yu Pan, Giancarlo Baldan, and Oscar Beijbom. nuscenes: A multimodal dataset for autonomous driving. *arXiv preprint arXiv:1903.11027*, 2019.

[^5]: Holger Caesar, Juraj Kabzan, Kok Seang Tan, Whye Kit Fong, Eric Wolff, Alex Lang, Luke Fletcher, Oscar Beijbom, and Sammy Omari. nuplan: A closed-loop ml-based planning benchmark for autonomous vehicles. *arXiv preprint arXiv:2106.11810*, 2021.

[^6]: Dian Chen, Vladlen Koltun, and Philipp Krähenbühl. Learning to drive from a world on rails. In *Proceedings of the IEEE/CVF International Conference on Computer Vision*, pp. 15590–15599, 2021.

[^7]: Li Chen, Penghao Wu, Kashyap Chitta, Bernhard Jaeger, Andreas Geiger, and Hongyang Li. End-to-end autonomous driving: Challenges and frontiers. *arXiv preprint arXiv:2306.16927*, 2023.

[^8]: Jie Cheng, Yingbing Chen, Xiaodong Mei, Bowen Yang, Bo Li, and Ming Liu. Rethinking imitation-based planners for autonomous driving, 2023.

[^9]: Jie Cheng, Yingbing Chen, and Qifeng Chen. Pluto: Pushing the limit of imitation learning-based planning for autonomous driving. *arXiv preprint arXiv:2404.14327*, 2024.

[^10]: Cheng Chi, Siyuan Feng, Yilun Du, Zhenjia Xu, Eric Cousineau, Benjamin Burchfiel, and Shuran Song. Diffusion policy: Visuomotor policy learning via action diffusion. *arXiv preprint arXiv:2303.04137*, 2023.

[^11]: Kashyap Chitta, Aditya Prakash, Bernhard Jaeger, Zehao Yu, Katrin Renz, and Andreas Geiger. Transfuser: Imitation with transformer-based sensor fusion for autonomous driving. *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 45(11):12878–12895, 2022.

[^12]: Hyungjin Chung, Jeongsol Kim, Michael T Mccann, Marc L Klasky, and Jong Chul Ye. Diffusion posterior sampling for general noisy inverse problems. *arXiv preprint arXiv:2209.14687*, 2022.

[^13]: Daniel Dauner, Marcel Hallgarten, Andreas Geiger, and Kashyap Chitta. Parting with misconceptions about learning-based vehicle motion planning. In *Conference on Robot Learning (CoRL)*, 2023a.

[^14]: Daniel Dauner, Marcel Hallgarten, Andreas Geiger, and Kashyap Chitta. Parting with misconceptions about learning-based vehicle motion planning. In *Conference on Robot Learning*, pp. 1268–1281. PMLR, 2023b.

[^15]: Prafulla Dhariwal and Alexander Nichol. Diffusion models beat gans on image synthesis. *Advances in neural information processing systems*, 34:8780–8794, 2021.

[^16]: Haoyang Fan, Fan Zhu, Changchun Liu, Liangliang Zhang, Li Zhuang, Dong Li, Weicheng Zhu, Jiangtao Hu, Hongye Li, and Qi Kong. Baidu apollo em motion planner, 2018.

[^17]: Jiyang Gao, Chen Sun, Hang Zhao, Yi Shen, Dragomir Anguelov, Congcong Li, and Cordelia Schmid. Vectornet: Encoding hd maps and agent dynamics from vectorized representation. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pp. 11525–11533, 2020.

[^18]: Jeffrey Hawke, Richard Shen, Corina Gurau, Siddharth Sharma, Daniele Reda, Nikolay Nikolov, Przemysław Mazur, Sean Micklethwaite, Nicolas Griffiths, Amar Shah, et al. Urban driving with conditional imitation learning. In *2020 IEEE International Conference on Robotics and Automation (ICRA)*, pp. 251–257. IEEE, 2020.

[^19]: Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. *Advances in neural information processing systems*, 33:6840–6851, 2020.

[^20]: Yihan Hu, Jiazhi Yang, Li Chen, Keyu Li, Chonghao Sima, Xizhou Zhu, Siqi Chai, Senyao Du, Tianwei Lin, Wenhai Wang, et al. Planning-oriented autonomous driving. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pp. 17853–17862, 2023.

[^21]: Yihan Hu, Siqi Chai, Zhening Yang, Jingyu Qian, Kun Li, Wenxin Shao, Haichao Zhang, Wei Xu, and Qiang Liu. Solving motion planning tasks with a scalable generative model. *arXiv preprint arXiv:2407.02797*, 2024.

[^22]: Zhiyu Huang, Haochen Liu, and Chen Lv. Gameformer: Game-theoretic modeling and learning of transformer-based interactive prediction and planning for autonomous driving. In *Proceedings of the IEEE/CVF International Conference on Computer Vision*, pp. 3903–3913, 2023.

[^23]: Michael Janner, Yilun Du, Joshua Tenenbaum, and Sergey Levine. Planning with diffusion for flexible behavior synthesis. In *International Conference on Machine Learning*, pp. 9902–9915. PMLR, 2022.

[^24]: Chiyu Jiang, Andre Cornman, Cheolho Park, Benjamin Sapp, Yin Zhou, Dragomir Anguelov, et al. Motiondiffuser: Controllable multi-agent motion prediction using diffusion. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pp. 9644–9653, 2023.

[^25]: Alex Kendall, Jeffrey Hawke, David Janz, Przemyslaw Mazur, Daniele Reda, John-Mark Allen, Vinh-Dieu Lam, Alex Bewley, and Amar Shah. Learning to drive in a day. In *2019 international conference on robotics and automation (ICRA)*, pp. 8248–8254. IEEE, 2019.

[^26]: John J. Leonard, Jonathan P. How, Seth J. Teller, Mitch Berger, Stefan Campbell, Gaston A. Fiore, Luke Fletcher, Emilio Frazzoli, Albert S. Huang, Sertac Karaman, Olivier Koch, Yoshiaki Kuwata, David C. Moore, Edwin Olson, Steven C. Peters, Justin Teo, Robert Truax, Matthew R. Walter, David Barrett, Alexander K Epstein, Keoni Maheloni, Katy Moyer, Troy Jones, Ryan Buckley, Matthew E. Antone, Robert Galejs, Siddhartha Krishnamurthy, and Jonathan Williams. A perception‐driven autonomous urban vehicle. *Journal of Field Robotics*, 25, 2008. URL [https://api.semanticscholar.org/CorpusID:1906145](https://api.semanticscholar.org/CorpusID:1906145).

[^27]: Zhiqi Li, Zhiding Yu, Shiyi Lan, Jiahan Li, Jan Kautz, Tong Lu, and Jose M Alvarez. Is ego status all you need for open-loop end-to-end autonomous driving? In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pp. 14864–14873, 2024.

[^28]: Tenglong Liu, Jianxiong Li, Yinan Zheng, Haoyi Niu, Yixing Lan, Xin Xu, and Xianyuan Zhan. Skill expansion and composition in parameter space. In *The Thirteenth International Conference on Learning Representations*, 2025. URL [https://openreview.net/forum?id=GLWf2fq0bX](https://openreview.net/forum?id=GLWf2fq0bX).

[^29]: Cheng Lu, Yuhao Zhou, Fan Bao, Jianfei Chen, Chongxuan Li, and Jun Zhu. Dpm-solver: A fast ode solver for diffusion probabilistic model sampling in around 10 steps. *Advances in Neural Information Processing Systems*, 35:5775–5787, 2022.

[^30]: Cheng Lu, Huayu Chen, Jianfei Chen, Hang Su, Chongxuan Li, and Jun Zhu. Contrastive energy prediction for exact energy-guided diffusion sampling in offline reinforcement learning. *arXiv preprint arXiv:2304.12824*, 2023.

[^31]: Chenlin Meng, Robin Rombach, Ruiqi Gao, Diederik Kingma, Stefano Ermon, Jonathan Ho, and Tim Salimans. On distillation of guided diffusion models. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pp. 14297–14306, 2023.

[^32]: Khan Muhammad, Amin Ullah, Jaime Lloret, Javier Del Ser, and Victor Hugo C de Albuquerque. Deep learning for safe autonomous driving: Current challenges and future directions. *IEEE Transactions on Intelligent Transportation Systems*, 22(7):4316–4336, 2020.

[^33]: Nigamaa Nayakanti, Rami Al-Rfou, Aurick Zhou, Kratarth Goel, Khaled S Refaat, and Benjamin Sapp. Wayformer: Motion forecasting via simple & efficient attention networks. In *2023 IEEE International Conference on Robotics and Automation (ICRA)*, pp. 2980–2987. IEEE, 2023.

[^34]: Jiquan Ngiam, Benjamin Caine, Vijay Vasudevan, Zhengdong Zhang, Hao-Tien Lewis Chiang, Jeffrey Ling, Rebecca Roelofs, Alex Bewley, Chenxi Liu, Ashish Venugopal, et al. Scene transformer: A unified architecture for predicting multiple agent trajectories. *arXiv preprint arXiv:2106.08417*, 2021.

[^35]: William Peebles and Saining Xie. Scalable diffusion models with transformers. In *Proceedings of the IEEE/CVF International Conference on Computer Vision*, pp. 4195–4205, 2023.

[^36]: Aditya Ramesh, Prafulla Dhariwal, Alex Nichol, Casey Chu, and Mark Chen. Hierarchical text-conditional image generation with clip latents. *arXiv preprint arXiv:2204.06125*, 1(2):3, 2022.

[^37]: Oliver Scheel, Luca Bergamini, Maciej Wołczyk, Błażej Osiński, and Peter Ondruska. Urban driver: Learning to drive from real-world demonstrations using policy gradients, 2021.

[^38]: Jascha Sohl-Dickstein, Eric A. Weiss, Niru Maheswaranathan, and Surya Ganguli. Deep unsupervised learning using nonequilibrium thermodynamics, 2015.

[^39]: Yang Song and Stefano Ermon. Generative modeling by estimating gradients of the data distribution. *Advances in neural information processing systems*, 32, 2019.

[^40]: Yang Song, Jascha Sohl-Dickstein, Diederik P. Kingma, Abhishek Kumar, Stefano Ermon, and Ben Poole. Score-based generative modeling through stochastic differential equations, 2021.

[^41]: Yang Song, Prafulla Dhariwal, Mark Chen, and Ilya Sutskever. Consistency models. *arXiv preprint arXiv:2303.01469*, 2023.

[^42]: Qiao Sun, Shiduo Zhang, Danjiao Ma, Jingzhe Shi, Derun Li, Simian Luo, Yu Wang, Ningyi Xu, Guangzhi Cao, and Hang Zhao. Large trajectory models are scalable motion predictors and planners. *arXiv preprint arXiv:2310.19620*, 2023.

[^43]: Qiao Sun, Huimin Wang, Jiahao Zhan, Fan Nie, Xin Wen, Leimeng Xu, Kun Zhan, Peng Jia, Xianpeng Lang, and Hang Zhao. Generalizing motion planners with mixture of experts for autonomous driving. *arXiv preprint arXiv:2410.15774*, 2024.

[^44]: Ardi Tampuu, Tambet Matiisen, Maksym Semikin, Dmytro Fishman, and Naveed Muhammad. A survey of end-to-end driving: Architectures and training methods. *IEEE Transactions on Neural Networks and Learning Systems*, 33(4):1364–1384, 2020.

[^45]: Ilya O Tolstikhin, Neil Houlsby, Alexander Kolesnikov, Lucas Beyer, Xiaohua Zhai, Thomas Unterthiner, Jessica Yung, Andreas Steiner, Daniel Keysers, Jakob Uszkoreit, et al. Mlp-mixer: An all-mlp architecture for vision. *Advances in neural information processing systems*, 34:24261–24272, 2021.

[^46]: Martin Treiber, Ansgar Hennecke, and Dirk Helbing. Congested traffic states in empirical observations and microscopic simulations. *Physical Review E*, 62(2):1805–1824, August 2000a. ISSN 1095-3787. doi: 10.1103/physreve.62.1805. URL [http://dx.doi.org/10.1103/PhysRevE.62.1805](http://dx.doi.org/10.1103/PhysRevE.62.1805).

[^47]: Martin Treiber, Ansgar Hennecke, and Dirk Helbing. Congested traffic states in empirical observations and microscopic simulations. *Physical review E*, 62(2):1805, 2000b.

[^48]: Chris Urmson, Joshua Anhalt, J. Andrew Bagnell, Christopher R. Baker, Robert Bittner, M. N. Clark, John M. Dolan, David Duggins, Tugrul Galatali, Christopher Geyer, Michele Gittleman, Sam Harbaugh, Martial Hebert, Thomas M. Howard, Sascha Kolski, Alonzo Kelly, Maxim Likhachev, Matthew McNaughton, Nick Miller, Kevin M. Peterson, Brian Pilnick, Ragunathan Raj Rajkumar, Paul E. Rybski, Bryan Salesky, Young-Woo Seo, Sanjiv Singh, Jarrod M. Snider, Anthony Stentz, William Whittaker, Ziv Wolkowicki, Jason Ziglar, Hong Bae, Thomas Brown, Daniel Demitrish, Bakhtiar Litkouhi, James N. Nickolaou, Varsha Sadekar, Wende Zhang, Joshua Struble, Michael Taylor, Michael Darms, and Dave Ferguson. Autonomous driving in urban environments: Boss and the urban challenge. *Journal of Field Robotics*, 25, 2008. URL [https://api.semanticscholar.org/CorpusID:11849332](https://api.semanticscholar.org/CorpusID:11849332).

[^49]: Matt Vitelli, Yan Chang, Yawei Ye, Ana Ferreira, Maciej Wołczyk, Błażej Osiński, Moritz Niendorf, Hugo Grimmett, Qiangui Huang, Ashesh Jain, et al. Safetynet: Safe planning for real-world self-driving vehicles using machine-learned policies. In *2022 International Conference on Robotics and Automation (ICRA)*, pp. 897–904. IEEE, 2022.

[^50]: Tongda Xu, Jian Li, Xinjie Zhang, Xingtong Ge, Dailan He, Xiyan Cai, Ming Sun, Yan Wang, Jingjing Liu, and Ya-Qin Zhang. Rethinking diffusion posterior sampling: From conditional score estimator to maximizing a posterior. In *The Thirteenth International Conference on Learning Representations*, 2025. URL [https://openreview.net/forum?id=GcvLoqOoXL](https://openreview.net/forum?id=GcvLoqOoXL).

[^51]: Brian Yang, Huangyuan Su, Nikolaos Gkanatsios, Tsung-Wei Ke, Ayush Jain, Jeff Schneider, and Katerina Fragkiadaki. Diffusion-es: Gradient-free planning with diffusion for autonomous driving and zero-shot instruction following. *arXiv preprint arXiv:2402.06559*, 2024.

[^52]: Yinan Zheng, Jianxiong Li, Dongjie Yu, Yujie Yang, Shengbo Eben Li, Xianyuan Zhan, and Jingjing Liu. Safe offline reinforcement learning with feasibility-guided diffusion model. In *The Twelfth International Conference on Learning Representations*, 2024. URL [https://openreview.net/forum?id=j5JvZCaDM0](https://openreview.net/forum?id=j5JvZCaDM0).

[^53]: Ziyuan Zhong, Davis Rempe, Yuxiao Chen, Boris Ivanovic, Yulong Cao, Danfei Xu, Marco Pavone, and Baishakhi Ray. Language-guided traffic simulation via scene-level diffusion. In *Conference on Robot Learning*, pp. 144–177. PMLR, 2023a.

[^54]: Ziyuan Zhong, Davis Rempe, Danfei Xu, Yuxiao Chen, Sushant Veer, Tong Che, Baishakhi Ray, and Marco Pavone. Guided conditional diffusion for controllable traffic simulation. In *2023 IEEE International Conference on Robotics and Automation (ICRA)*, pp. 3560–3566. IEEE, 2023b.

[^55]: Zikang Zhou, Jianping Wang, Yung-Hui Li, and Yu-Kai Huang. Query-centric trajectory prediction. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pp. 17863–17873, 2023.