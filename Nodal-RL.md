### Copyright &copy; 2020 Jacob Valdez

under the [MIT License](https://opensource.org/licenses/MIT)

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

# Nodal Reinforcement Learning

Abstract and pictures/videos *en simulo*. Maybe also simplified diagrams of AGI node and network.

# Introduction

AGI-0 rests on the hypotheses that:
- encoding the principles of [free energy minimization](https://en.wikipedia.org/wiki/Free_energy_principle) in a multilayer hidden Markov chain prediction unit will enable unsupervised, open-ended learning
- competitive ensembles of prediction units will succeed in overcoming the [no free lunch theorom](https://en.wikipedia.org/wiki/No_free_lunch_theorem)
- ~~supervised multiagent simulations will enable learning mirroring features that transfer successfully to humans~~
- embedded natural language self examination objectives will automatically align agent behavior with encoded human values from environment reinforcement without any supervised reward administration
- **the realization of these hypotheses will achieve artificial general intelligence**

Discuss my theory of hierarchial conditioning here

Each child only provides additional conditioning context but is not required for a module to produce an output. (Kind of like how [conditioning gpt-2 with a longer input sentence gives you more control over its output](https://colab.research.google.com/drive/12_EzHGVE8ZnJ-MA9W_8tQuP6xHpWb1X3?usp=sharing))

# Methodology

$\newcommand{\ab}{\mathcal{A}}$
$\newcommand{\pr}{\mathcal{P}}$
$\newcommand{\pol}{\mathcal{\pi}}$
$\newcommand{\o}[1][t]{{o}_{#1}}$
$\newcommand{\otrue}[1][t]{{o}_{#1}^{true}}$
$\newcommand{\opast}[1][t-T_o:t]{{o}_{#1}}$
$\newcommand{\otruepast}[1][t-T_o:t]{{o}_{#1}^{true}}$
$\newcommand{\s}[1][t]{{s}_{#1}}$
$\newcommand{\spred}[1][t+1]{{s}_{#1}^{pred}}$
$\newcommand{\oimagpast}[1][\tau-T_o:\tau]{{o}_{#1}^{imag}}$
$\newcommand{\oimag}[1][\tau]{{o}_{#1}^{imag}}$
$\newcommand{\simag}[1][\tau]{{s}_{#1}^{imag}}$
$\newcommand{\spredimag}[1][\tau+p]{{s}_{#1}^{pred,imag}}$
$\newcommand{\stargetimag}[1][\tau+p]{{s}_{#1}^{*,imag}}$
$\newcommand{\starget}[1][t+1]{{s}_{#1}^*}$
$\newcommand{\otarget}[1][t+1]{{o}_{#1}^*}$
$\newcommand{\sideal}{{s}_{t+p}^{ideal}}$
$\newcommand{\sblind}[1][\tau+p]{{s}_{#1}^{blind}}$
$\newcommand{\pe}[1][t]{pe_{#1}}$
$\newcommand{\n}[1][i]{n_{#1}}$
$\newcommand{\nparent}{\mathcal{N_{pa}}}$
$\newcommand{\nchildren}{\mathcal{N_{ch}}}$
$\newcommand{\U}{\mathcal{U}}$
$\newcommand{\Ex}[1][ ]{ \underset{#1}{ \mathbb{E} } }$
$ \newcommand{\vect}[1]{\boldsymbol{#1}} $
$\def\falert{f_{alert}}$
$\def\NN{N\kern-0.2em N}$
$\def\DS{\mathcal{D}}$
$\def\IE{\mathcal{E}_I}$
$\def\CER{\mathcal{C}}$
$\newcommand{norm}[1]{\bigl\lvert\bigl\rvert{#1}\bigr\lvert\bigr\rvert}$
$\def\loss{\mathcal{L}}$
$\def\obj{\mathcal{J}}$

## Architecture

Nodal-RL operates by minimizing predictive entropy. It does this by decomposing the state space with a [structured probabblistic model](https://en.wikipedia.org/wiki/Graphical_model) into sensory, information, and actuator nodes. Each *information node* on the graph only models a subset of all information. The directed and potentially cyclic model $\langle\mathcal{N}\kern-0.15em,\mathcal{E}\rangle$ may look like:

![example network](https://mermaid.ink/img/eyJjb2RlIjoiZ3JhcGggTFJcbiAgUzAwW3BpeGVsc10gLS0-IFMwMSh2aXN1YWwgZmVhdHVyZXMpXG4gIFMwMSAtLT4gUDAoY29tcG9zaXRlIGZlYXR1cmVzKVxuXG4gIFMxMFthdWRpbyBzcGVjdHJvZ3JhcGhdIC0tPiBTMTEodG9rZW5zKVxuICBTMTEgLS0-IFAwKGNvbXBvc2l0ZSBmZWF0dXJlcylcblxuICBTMDEgLS0-IFNYMDEwXG4gIFMxMSAtLT4gU1gwMTBcbiAgU1gwMTAobmF0dXJhbCBsYW5ndWFnZSBJKSAtLT4gU1gwMTFcbiAgU1gwMTEobmF0dXJhbCBsYW5ndWFnZSBJSSkgLS0-IFAwXG5cbiAgUzIwW2FjdHVhdG9yIHNpZ25hbHNdIC0tPiBTMjEobW90aW9uIHByaW1pdGl2ZXMpXG4gIFMyMSAtLT4gUDAoY29tcG9zaXRlIGZlYXR1cmVzKVxuXG4gIFAwIC0tPiBQMDAoYWJzdHJhY3QgcGVyc3BlY3RpdmUgSSlcbiAgUDAgLS0-IFAwMShhYnN0cmFjdCBwZXJzcGVjdGl2ZSBJSSlcbiAgUDAgLS0-IFAwMihhYnN0cmFjdCBwZXJzcGVjdGl2ZSBJSUkpXG5cdFxuICBQMDAgLS0-IFAwMDAoYWJzdHJhY3QgcGVyc3BlY3RpdmUgSVYpXG4gIFAwMCAtLT4gUDAwMShhYnN0cmFjdCBwZXJzcGVjdGl2ZSBWKVxuICBQMDIgLS0-IFAwMDFcblxuICBQMDEgLS0-IFAwMDBcblxuICBQMDAwIC0tPiBQMCIsIm1lcm1haWQiOnsidGhlbWUiOiJkZWZhdWx0In0sInVwZGF0ZUVkaXRvciI6ZmFsc2V9) <!-- https://mermaid-js.github.io/mermaid-live-editor/#/edit/eyJjb2RlIjoiZ3JhcGggTFJcbiAgUzAwW3BpeGVsc10gLS0-IFMwMSh2aXN1YWwgZmVhdHVyZXMpXG4gIFMwMSAtLT4gUDAoY29tcG9zaXRlIGZlYXR1cmVzKVxuXG4gIFMxMFthdWRpbyBzcGVjdHJvZ3JhcGhdIC0tPiBTMTEodG9rZW5zKVxuICBTMTEgLS0-IFAwKGNvbXBvc2l0ZSBmZWF0dXJlcylcblxuICBTMDEgLS0-IFNYMDEwXG4gIFMxMSAtLT4gU1gwMTBcbiAgU1gwMTAobmF0dXJhbCBsYW5ndWFnZSBJKSAtLT4gU1gwMTFcbiAgU1gwMTEobmF0dXJhbCBsYW5ndWFnZSBJSSkgLS0-IFAwXG5cbiAgUzIwW2FjdHVhdG9yIHNpZ25hbHNdIC0tPiBTMjEobW90aW9uIHByaW1pdGl2ZXMpXG4gIFMyMSAtLT4gUDAoY29tcG9zaXRlIGZlYXR1cmVzKVxuXG4gIFAwIC0tPiBQMDAoYWJzdHJhY3QgcGVyc3BlY3RpdmUgSSlcbiAgUDAgLS0-IFAwMShhYnN0cmFjdCBwZXJzcGVjdGl2ZSBJSSlcbiAgUDAgLS0-IFAwMihhYnN0cmFjdCBwZXJzcGVjdGl2ZSBJSUkpXG5cdFxuICBQMDAgLS0-IFAwMDAoYWJzdHJhY3QgcGVyc3BlY3RpdmUgSVYpXG4gIFAwMCAtLT4gUDAwMShhYnN0cmFjdCBwZXJzcGVjdGl2ZSBWKVxuICBQMDIgLS0-IFAwMDFcblxuICBQMDEgLS0-IFAwMDBcblxuICBQMDAwIC0tPiBQMCIsIm1lcm1haWQiOnsidGhlbWUiOiJkZWZhdWx0In0sInVwZGF0ZUVkaXRvciI6ZmFsc2V9 -->

Above, among many other partial models, is modeled the information that $\n[tokens]$ and $\n[visual\ features]$ supply to $\n[nat\ lang\ I]$, the information $\n[nat\ lang\ I]$ supplies to $\n[nat\ lang\ II]$, and the partial information $\n[nat\ lang\ II]$ adds to a composite understanding of the world. This composite understanding is captured in the state variable $\n[comp].\s[ ]$. Instead of producing its state from directly observing sensory data, $\n[comp]$ only observes its parent nodes' states $\n[comp].\o[ ]=\{\n[nat\ lang\ II].\s[ ],\n[motion\ prim].\s[ ],\n[visual\ features].\s[ ]\}$.

An information node $\n$ attempts to build an internal model of its observed variables $P(\o[t+p]|\opast)$. It does this by first *abstracting* the information from its observable data trajectory into an internal *state* $\s=\ab(\opast)$. It then attempts to model the trajectory of that internal state with a *predictor* function $\spred=\pr(\s)$. Predictive error $D_{KL}(\spred,\s[t+1])$ and predictive uncertainty $H(\spred)$ then become the objectives to cumulatively minimize.

The *policy* $\pol$ specifies *target observations* $\otarget$ that approach these objectives. Target observations belonging to a child layer influence their parent's *target state* (e.g.: $\n[nat\ lang\ II].\otarget$ affects $\n[nat\ lang\ I].\starget$). Since a node may have multiple children, the target state is determined by a $\n[ch].w_{\mathcal{Pa}_i}$ weighted mean of the node children's target observations $\{\n[ch].\otarget\ |\ \forall\n[ch]\in\n\smash{.}\nchildren\}$ concatenated with the node's own predicted state $\n\smash{.}\spred$. Additionally, probabilistic certainty weighting, ($\CER(x)=0=$ very indecisive) weakens the influence of vague target distributions.

$\text{weighted ave}(\{\langle w_{pred},\spred\rangle\}\cup\{\langle \n[ch].w_{\mathcal{Pa}_i}\CER(\n[ch].\otarget), \n[ch].\otarget\rangle\ |\ \forall\n[ch]\in\nchildren\})$

Nodes do not compute updates simultaneously or even at the same frequency though. They are individually updated at periods of $\n\smash{.}p=\n\smash{.}f^{-1}$. As a result, $\n[ch].\otarget[t+\n\smash{.}p]$ may not exist. To avoid blocking, child node target observations are instead queried for the most recent existing record $\n[ch].\otarget[\tau]\ |\ \exists\n[ch].\otarget[\tau];\ \tau=\max [t-N_o:t+p];$ An exponential recency decay $e^{-\n[ch].\lambda(t+p-\tau)}$ additionally weights the target observation. The target state is thus:

$\starget[t+p]=\text{weighted ave}(\{\langle w_{pred},\spred[t+p]\rangle\} \cup \{\langle \n[ch].w_{\mathcal{Pa}_i}\CER(\n[ch].\otarget[\tau])\times e^{-\n[ch].\lambda(t+p-\tau)}, \n[ch].\otarget[\tau]\rangle\ |\ \exists\n[ch].\otarget[\tau];\ \tau=\max [t-N_o:t+p];\ \forall\n[ch]\in\nchildren\})$

Target state in turn biases the target observation via the policy $\otarget=\pol(\s,\starget)$. Since the target state may be a weighted mean of several stochastic variables, child target observations with a high entropy will have less influence in the weighted mean distribution-even if its weight is large. The more certain (lower entropy) a target observation is, the greater influence it will have on its parent target state.

This makes sibling nodes form an implicit team. One node $\n[abs\ per\ I]$ may have a very clear understanding of the dynamics of one subspace of the observation space $\mathcal{O}_i\subset\mathcal{O}$, hence producing low entropy predictions and strong opinions on what $\otarget[t+p]$ should be. At the same time, its sibling $\n[abs\ per\ II]$ may not have knowledge of that domain and therefore produce highly irrational, entropic observation targets. The result of averaging the two nodes' target observations is a distribution with relatively little influence by $\n[abs\ per\ II]$. Together, nodes form a network of experts each specializing in unique domains to give the agent open-ended intelligence.

## Stochastic Models

To allow information nodes to produce multimodal and multivariate observation and state distributions, the following reparametrization trick is employed to layer input concatenated with a multidimensional uniform distribution:

$\vect{y}=\sigma(W[\vect{x};\vect{\zeta}]+b)\qquad\vect{\zeta}\thicksim\{\mathcal{U(0,1)}\}^{N_\zeta}$
 <!--$y=\sigma(\zeta\vect{w}_\sigma\cdot\vect{x}+\vect{w}_\mu\cdot\vect{x}+b) \qquad \zeta\thicksim\mathcal{U}(0,1)$-->

<!--or for a dense layer:-->

<!--$\vect{y}=\sigma(\vect{\zeta}W_\sigma\vect{x}+\vect{w}_\mu\cdot\vect{x}+b) \qquad \zeta\thicksim\mathcal{U}(0,1)$-->  

Since this design employs continuous distributions, certainty $\CER$ is measured by the relative entropy of the uniform distribution with respect to a target distribution $P$:

$\CER(P)=-D_{KL}(\mathcal{U}(a,b)||P)$

This above definition demands a definition of activation bounds $a$ and $b$ for each distribution. These are regularized distribution-wise by:

${\loss}_{bounds}(x)=\begin{cases}f(a-x)\qquad&x\lt a\\0\qquad& a\le x\le b\\f(x-b)\qquad& x\gt b\end{cases}$

with $f(x)=\{x,x^2,x^n,e^x,\dots\}$ being any differentiable monotonically increasing function on $(0,\infty)$.

The certainty of various distributions zero as they are more unpredictable and increases to above one as they become more certain of the likelihood of a random variable.

## Dreaming

Not all observations collected online are equally relevant to future tasks. Additionally, when observations add little information to an already highly unpredictable state trajectory, it becomes pointless to informatively utilize the next state prediction. In these situations, informations nodes maintain productivity by *dreaming*.

Dreaming makes a shift from closed to open loop control. It involves relying less on observed data and more on anticipated observations. It mixes the true observation $\otrue$ with last timestep's target observation $\otarget[t]$ by a nonlinear function $\falert$ to produce the observed processed by all other functions:

$\o=\falert\otrue+(1-\falert)\otarget[t]$

where $\falert$ increases alertness when surprising, unexpected, or informative observations occur $\falert\propto I(\otrue)$ and decreases alertness when predictability is low $\falert\propto (H(\s[t-p]))^{-1}$. To ensure the transition to dreaming and back to alertness is not jittery, these factors are smoothed with a rolling average and transformed with $\tanh$:

$\overline{\underline{\textbf{Algorithm:}\ \text{Percieved Observation }\o}}$

$1.\ d_t^{ave}=d_t^{ave}+\alpha_{d,1}\frac{I(\otrue)}{H(\s[t-p])}$

$2.\ \falert=\tanh[\alpha_{d,2}(d_t^{ave}-\alpha_{d,3})]$

$3.\ \mathbf{return}\ \o=\falert\otrue+(1-\falert)\otarget[t]$

where $\alpha_{d,1}\lt1$ is small number that influences the rolling average latency, $\alpha_{d,2}\gt1$ is large and makes the transition from dream to alert state abrupt, and $\alpha_{d,3}$ is the minimum average ratio of information to entropy that must be provided by observations for $\n$ to pay attention to them.

Since dreamed state trajectories will generally have a larger entropy than states based on true observations, dreaming node target observations have a correspondingly high entropy and generally do not compete with target observations from sibling nodes that are not dreaming.

## Running

Information nodes join an asynchronously updating network with sensory nodes and actuator nodes. Unless specified, variables are implied to belong to $\n$ ($\n.p=p$).

### Sensory Nodes

Sensory and simply publish states:

$\overline{\underline{\textbf{Algorithm:}\ \text{Sensory Node }\n{\ Update}}}$

$1.\ \s=\text{get sensory data}$

$6.\ \textbf{wait}\ p\ \text{timesteps}$ 

$3.\ \textbf{repeat}$

### Actuator Nodes

Actuator nodes attempts to fulfill their children's target observations. At the actuator level, distributions must finally be sampled into real numbers.

$\overline{\underline{\textbf{Algorithm:}\ \text{Actuator Node }\n{\ Update}}}$

$1.\ \starget[t+p]=\text{weighted ave}(\{\langle w_{pred},\spred[t+p]\rangle\} \cup \{\langle \n[ch].w_{\mathcal{Pa}_i}\CER(\n[ch].\otarget[\tau])\times e^{-\n[ch].\lambda(t+p-\tau)}, \n[ch].\otarget[\tau]\rangle\ |\ \exists\n[ch].\otarget[\tau];\ \tau=\max [t-N_o:t+p];\ \forall\n[ch]\in\nchildren\})$

$2.\ \text{execute } \overset{\small{N_s}}{\Ex}[\starget[t+p]] \quad\rhd\text{execute weighted target state sampling }N_s\text{ times}$

$3.\ \textbf{wait}\ p\ \text{timesteps}$

$4.\ \s[t+p]=\text{get actual result}$

$5.\ \textbf{repeat}$

### Information Nodes

Information nodes simultaneously engage in bottom-up processing and propagate top-down conditioning:

$\overline{\underline{\textbf{Algorithm:}\ \text{Information Node }\n{\ Update}}}$

$1.\ \mathbf{store}\ \otrue=\rm{concat}(\n[parent].\s|\forall\n[parent]\in\nparent)$

$2.\ d_t^{ave}=d_t^{ave}+\alpha_{d,1}\frac{I(\otrue)}{H(\s[t-p])}$

$3.\ \falert=\tanh[\alpha_{d,2}(d_t^{ave}-\alpha_{d,3})]$

$4.\ \o=\falert\otrue+(1-\falert)\otarget[t]$

$5.\ \s=\ab(\opast[t-N_o:t])$

$6.\ \spred[t+p]=\pr(\s)$

$7.\ \starget[t+p]=\text{weighted ave}(\{\langle w_{pred},\spred[t+p]\rangle\} \cup \{\langle \n[ch].w_{\mathcal{Pa}_i}\CER(\n[ch].\otarget[\tau])\times e^{-\n[ch].\lambda(t+p-\tau)}, \n[ch].\otarget[\tau]\rangle\ |\ \exists\n[ch].\otarget[\tau];\ \tau=\max [t-N_o:t+p];\ \forall\n[ch]\in\nchildren\})$

$8.\ \otarget[t+p]=\pol(\s,\starget[t+p])$

$9.\ \textbf{wait}\ p\ \text{timesteps}$ 

$10.\ \textbf{repeat}$

## Training

Training is performed online, on-policy, and on-predictor with new, prioritized, and imagined data. Minibatches are only one rollout sequence of observations ($N=1$) from every point in time in an episode to allow training large architectures and keep policies fresh. Dreaming provides novel observation trajectories for the abstractor and child information nodes to learn. However information nodes do not train their policy or predictor on its own dreamed trajectories ($\o=\otrue$ when optimizing).

### Learning the Policy

The policy aims to produce target observations $\otarget[t+p]=\pol(\s,\starget[t+p])$ which cumulatively:
- maximize target state achievement $D_{KL}\bigl(\starget[t+p]||\ab([\otruepast;\pol(\s,\starget[t+p])])\bigr)$
- maximize resulting state certainty $\CER\bigl(\ab([\otruepast;\pol(\s,\starget[t+p])])\bigr)$
- maximize informative observations targeted (increase in state certainty) $\CER\bigl(\ab([\otruepast;\pol(\s,\starget[t+p])])\bigr)-\CER(\s)$

These objectives are combined as:

$\loss_\pol=\underbrace{\beta_{\pol,D}\norm{\ln{D_{KL}\bigl(\starget[t+p]||\ab([\otruepast;\pol(\s,\starget[t+p])])\bigr)}}}_\text{target state achievement}-\underbrace{\beta_{\pol,\CER}\norm{\ln{\CER\bigl(\ab([\otruepast;\pol(\s,\starget[t+p])])\bigr)}}}_\text{resulting state certainty}-\underbrace{\beta_{\pol,I}\norm{f_{\pol,I}{[\CER\bigl(\ab([\otruepast;\pol(\s,\starget[t+p])])\bigr)-\CER(\s)]}}}_\text{informative observations targeted}$

This attempts to make the policy descend the gradient of predictive error and entropy while attempting to please its children's desired target observations. That is, it should produce a target observation which will make the abstractor produce an abstraction $\ab([\opast;\pol(\s,\starget[t+p])])$ as close as possible to $\starget[t+p]$. The last objective is added because informative events often reduce cumulative predictive entropy. By searching out unexpected, *informative* events, an agent is able to learn a more robust internal model $\Pr(\o[t+p]|\opast)$. $f_{\pol,I}$ is a positive, continuous monotonically decreasing function like $-e^{-x}$.

### Learning the Predictor

Ideally, the predictor aims to model its observations perfectly $\spred[t+p]=\s[t+p]$. This means it should be able to maintain an accurate predicted state trajectory s.t. $\pr(\pr(\pr(\s)))\approxeq \s[t+3p]$. That requires maximizing predictive certainty $\CER(\pr(\s))$. Jointly, these objectives compose:

$\obj_\pr=\underbrace{\beta_{\pr,D}\norm{\ln{\bigl[D_{KL}(\pr(\s)||\s[t+p])\bigr]}}}_\text{accuracy} - \underbrace{\beta_{\pr,\CER}\norm{\ln{ \CER(\pr(\s))}}}_\text{certainty}$

Objectives are computed component-wise and only then normalized to encourage modeling only a few variables accurately rather than attempting to poorly model a joint distribution.

### Learning the Abstractor

In some information nodes, the abstractor may be frozen with pretrained weights. When trainable however, its optimization attempts to capture variables relevant to both the policy's and predictor's losses. Although dreaming may provide novel observation trajectories for the abstractor to represent only true observations are used to optimize $\ab$. However, child information nodes do optimize on the novel state trajectories generated by dreaming.

### Putting It All Together

Optimization begins at each record in time of an information episode $\IE=\langle\otruepast,\nchildren \text{(target observation sequences only)}\rangle$ by predicting the rollout for $T_{rollout}$ timesteps. Then cumulative policy and predictor losses are minimized by optimizing policy, predictor, and abstractor weights. Importantly, this optimization process takes place for nodes individually, allowing very large node architectures to integrate in an agent that no one processing unit could support.

$\begin{equation}\overline{\underline{\mathbf{Algorithm}:\ \text{Train Information Node }\n}} \\
1.\ \ \mathbf{for}\ t\in[\IE.T_{start}+p,\IE.T_{start}+2p,\dots,\IE.T_{end}]:\qquad\rhd\text{precompute }\s[ ],\spred[ ],\starget[ ],\otarget[ ] \\
2.\ \ \quad \s=\ab(\IE.\otruepast) \\
3.\ \ \quad \spred[t+p]=\pr(\s) \\
4.\ \ \quad \starget[t+p]=\text{weighted ave}(\{\langle w_{pred},\spred[t+p]\rangle\} \cup \\
\kern+7em\{\langle \n[ch].w_{\mathcal{Pa}_i}\CER(\n[ch].\otarget[\tau]) e^{-\n[ch].\lambda(t+p-\tau)}, \n[ch].\otarget[\tau]\rangle\ |\ \exists\n[ch].\otarget[\tau];\ \tau=\max [t-N_o:t+p];\ \forall\n[ch]\in\IE.\nchildren\}) \\
5.\ \ \quad\otarget[t+p]=\pol(\s,\starget[t+p]) \\
6.\ \ \quad\sideal=\ab([\otruepast;\otarget]) \\
7.\ \ \mathbf{end\ for} \\
8.\ \ \mathbf{for}\ t\in[\IE.T_{start}+p,\IE.T_{start}+2p,\dots,\IE.T_{end}]: \\
9.\ \ \quad \loss_t=\underbrace{\beta_{\pol,D}\norm{\ln{D_{KL}\bigl(\starget[t+p]||\ab(\sideal)\bigr)}}}_\text{target state achievement}-\underbrace{\beta_{\pol,\CER}\norm{\ln{\CER(\sideal)}}}_\text{resulting state certainty}\\
\kern+8em-\underbrace{\beta_{\pol,I}\norm{f_{\pol,I}{[\CER(\sideal)-\CER(\s)]}}}_\text{informative observations targeted} \\
9.\ \ \quad \tau=t \\
10.\ \quad \oimagpast=\otruepast \\
11.\ \quad \sblind[\tau]=\s \\
12.\ \quad \mathbf{while}\ \tau\le t+T_{rollout}\ \mathbf{and}\ \tau\le\IE.T_{end}-p:\qquad\rhd\text{compute rollout comparing against real data} \\
13.\ \qquad \simag=\ab(\oimagpast) \\
14.\ \qquad \spredimag[\tau+p]=\pr(\simag) \\
15.\ \qquad \stargetimag[\tau+p]=\text{weighted ave}(\{\langle w_{predimag},\spredimag[\tau+p]\rangle\} \cup \{\langle \n[ch].w_{\mathcal{Pa}_i}\CER(\n[ch].\o[\tau_{ch}])e^{-\n[ch].\lambda(\tau+p-\tau_{ch})}, \n[ch].\o[\tau_{ch}]\rangle \\
\kern+10em |\ \exists\n[ch].\o[\tau_{ch}];\ \tau_{ch}=\max [\tau-N_o:\tau+p];\ \forall\n[ch]\in\IE.\nchildren\}) \\
16.\ \qquad \oimag[\tau+p]=\pol(\simag,\stargetimag) \\
17.\ \qquad \sblind=\pr(\sblind[\tau]) \\
18.\ \qquad \loss_t=\loss_t+\gamma^{t-\tau}\Biggl[ \underbrace{\beta_{\pol,D}\norm{\ln{D_{KL}(\stargetimag[\tau]||\simag)}}}_\text{target state achievement} - \underbrace{\beta_{\pol,\CER}\norm{\ln{\CER(\simag)}}}_\text{resulting state certainty}\\
\kern+6em - \underbrace{\beta_{\pol,I}\norm{f_{\pol,I}{[\CER(\sblind)-\CER(\simag)]}}}_\text{informative observations targeted} \\
\kern+6em + \underbrace{\beta_{\pr,D}\norm{\ln{\bigl[D_{KL}(\sblind||\s[t+p])}}}_\text{blind prediction accuracy} - \underbrace{\beta_{\pr,\CER}\norm{\ln{ \CER(\sblind)}}}_\text{blind prediction certainty}\Biggr]\\
19.\ \qquad \tau = \tau + p \\
20.\ \ \quad \mathbf{end\ while} \\
22.\ \ \mathbf{end\ for} \\
21.\ \underset{\theta_\ab,\theta_\pr,\theta_\pol}\min\loss_{\forall t}\qquad\rhd\text{minimize discounted sum loss and reward for timestep }t \end{equation}$

To minimize uncertainty, $\beta_{\pol,\CER}>1$, $\beta_{\pr,\CER}>1$.

## Knowledge Transfer and Swarm Intelligence

AGI-0's node-based architecture allows different networks to share underlying nodes and individual nodes to transfer in knowledge from other neural networks. For example, a software automation agent and a writing agent might benefit by sharing nodes that process natural language. Some natural language processing nodes might also utilize existing language models like gpt for their abstractor and a conditional language modeling head for the policy. Nodes do not belong to any one agent but instead a *swarm cloud* which trains and serves individual nodes. *Information episodes* $\IE$ are collected from nodes after the agent's episode completes. This allows agents to synergize each other's heterogeneous architectures.

![swarm intelligence](https://mermaid.ink/img/eyJjb2RlIjoiZ3JhcGggVERcbiAgQVtwaXhlbHNdIC0tPiBCKHZpc3VhbCBkYXRhKVxuICBBIC0tPiBDXG4gIEIgLS0-IEMoc3ltYm9sIHN0cmluZ3MpXG4gIEMgLS0-IEQobGFuZ3VhZ2UgcHJvY2Vzc2luZylcblx0RCAtLT4gRShhYnN0cmFjdCB1bmRlcnN0YW5kaW5nKVxuICBGW2tleSBldmVudHNdIC0tPiBHXG4gIEhbbW91c2UgZXZlbnRzXSAtLT4gR1xuICBHKGFjdGlvbiBwcmltaXRpdmVzKVxuICBHIC0tPiBFXG4gIEYgLS0-IEkobGFuZ3VhZ2UgcHJvY2Vzc2luZylcbiAgSSAtLT4gRVxuXG4gIEpbdGV4dF0gLS0-IEtcbiAgSyhsYW5ndWFnZSBwcm9jZXNzaW5nKVxuICBLIC0tPiBMKGxhbmd1YWdlIHVuZGVyc3RhbmRpbmcgSSlcbiAgSyAtLT4gTShsYW5ndWFnZSB1bmRlcnN0YW5kaW5nIElJKVxuICBLIC0tPiBOKGxhbmd1YWdlIHVuZGVyc3RhbmRpbmcgSUlJKVxuICBMIC0tPiBOXG4gIE0gLS0-IE4iLCJtZXJtYWlkIjp7InRoZW1lIjoiZGVmYXVsdCJ9LCJ1cGRhdGVFZGl0b3IiOmZhbHNlfQ)
<!-- https://mermaid-js.github.io/mermaid-live-editor/#/edit/eyJjb2RlIjoiZ3JhcGggVERcbiAgQVtwaXhlbHNdIC0tPiBCKHZpc3VhbCBkYXRhKVxuICBBIC0tPiBDXG4gIEIgLS0-IEMoc3ltYm9sIHN0cmluZ3MpXG4gIEMgLS0-IEQobGFuZ3VhZ2UgcHJvY2Vzc2luZylcblx0RCAtLT4gRShhYnN0cmFjdCB1bmRlcnN0YW5kaW5nKVxuICBGW2tleSBldmVudHNdIC0tPiBHXG4gIEhbbW91c2UgZXZlbnRzXSAtLT4gR1xuICBHKGFjdGlvbiBwcmltaXRpdmVzKVxuICBHIC0tPiBFXG4gIEYgLS0-IEkobGFuZ3VhZ2UgcHJvY2Vzc2luZylcbiAgSSAtLT4gRVxuXG4gIEpbdGV4dF0gLS0-IEtcbiAgSyhsYW5ndWFnZSBwcm9jZXNzaW5nKVxuICBLIC0tPiBMKGxhbmd1YWdlIHVuZGVyc3RhbmRpbmcgSSlcbiAgSyAtLT4gTShsYW5ndWFnZSB1bmRlcnN0YW5kaW5nIElJKVxuICBLIC0tPiBOKGxhbmd1YWdlIHVuZGVyc3RhbmRpbmcgSUlJKVxuICBMIC0tPiBOXG4gIE0gLS0-IE4iLCJtZXJtYWlkIjp7InRoZW1lIjoiZGVmYXVsdCJ9LCJ1cGRhdGVFZGl0b3IiOmZhbHNlfQ -->

![similarities](https://mermaid.ink/img/eyJjb2RlIjoiZ3JhcGggVERcbiAgQVtvYnNlcnZhdGlvbnMgdXAgdG8gdF1cbiAgQltiaWFzaW5nIGluZm9ybWF0aW9uXVxuICBDW3RhcmdldCBvYnNlcnZhdGlvbl1cblxuICBEKGFic3RyYWN0b3IpXG4gIEUocHJlZGljdG9yKVxuICBGW3RhcmdldCBzdGF0ZV1cbiAgRyhwb2xpY3kpXG5cbiAgQSAtLT4gRFxuICBEIC0tPiBFXG4gIEUgLS0-IEZcbiAgQiAtLT4gRlxuXG4gIEYgLS0-IEdcbiAgRyAtLT4gQ1xuXG4gIElbdG9rZW4gc2VxdWVuY2VdIC0tPiBKKHRyYW5zZm9ybWVyKVxuICBKIC0tPiBLKGxhbmd1YWdlIG1vZGVsaW5nIGhlYWQpXG4gIEsgLS0-IExbbmV4dCB0b2tlbl0iLCJtZXJtYWlkIjp7InRoZW1lIjoiZGVmYXVsdCJ9LCJ1cGRhdGVFZGl0b3IiOmZhbHNlfQ)
<!-- https://mermaid-js.github.io/mermaid-live-editor/#/edit/eyJjb2RlIjoiZ3JhcGggVERcbiAgQVtvYnNlcnZhdGlvbnMgdXAgdG8gdF1cbiAgQltiaWFzaW5nIGluZm9ybWF0aW9uXVxuICBDW3RhcmdldCBvYnNlcnZhdGlvbl1cblxuICBEKGFic3RyYWN0b3IpXG4gIEUocHJlZGljdG9yKVxuICBGW3RhcmdldCBzdGF0ZV1cbiAgRyhwb2xpY3kpXG5cbiAgQSAtLT4gRFxuICBEIC0tPiBFXG4gIEUgLS0-IEZcbiAgQiAtLT4gRlxuXG4gIEYgLS0-IEdcbiAgRyAtLT4gQ1xuXG4gIElbdG9rZW4gc2VxdWVuY2VdIC0tPiBKKHRyYW5zZm9ybWVyKVxuICBKIC0tPiBLKGxhbmd1YWdlIG1vZGVsaW5nIGhlYWQpXG4gIEsgLS0-IExbbmV4dCB0b2tlbl0iLCJtZXJtYWlkIjp7InRoZW1lIjoiZGVmYXVsdCJ9LCJ1cGRhdGVFZGl0b3IiOmZhbHNlfQ -->
In the above comparison between a language modeling transformer and an information node, note that the predictor must be trained on supervised data. This could be implimented by initializing or even freezing the abstractor's weights with the transformer network, training the predictor to predict embedding transitions, and forcing the policy to produce the required next token from that predicted embedding (although top-down influence can change network output). 

# Experiments

To evaluate the open ended performance of Nodal-RL, experiments test 

- **Architecture Deviations**: different information node functions (dense, LSTM, sparse transformer, sparse VAE, VQ-VAE, EBM, and specialized: convolutional, language models, mel-tacotron), node structures (hetero/homogeneous ensembles), and network superstructures (integrated chatbot, google search, KB query)
- **Training**: whether the policy is allowed to learn unsupervised with the above training loop ("Putting It All Togethor") or is supervised by expert demonstration. also experimenting with various optimization tweeks such as rewarding entropic target observations, various hyperparameters
- **In-context and Multiple Domain learning**: how quickly the agent can adapt from pure observation without any gradient updates and measuring the performance of this adaptation across diverse skill domains 
- **Diverse Agent-Environments**: some on the experimentation list: board games (chess, checkers, go), kinesthetic (one legged hopper, spider, humanoid), content completion (video from frame sequence, natural language modeling, image inpainting, music generation), computer interface (web surfing, VM Bash interface, VM gui interface, world of bits++), 3D worlds (MALMO, Second Life), design (natural language documents, artwork, mechanical engineering, electrical engineering using VM interface), second-hand research (using VM interface with Internet), programming (using VM interface, Desmos, GeoGebra, Bash REPL, Jupyter Python, selfML), and conditioning with only natural language conversation and interactive demonstrations to complete real world objectives.

Specific experiments are published in public Google Colaboratory Notebooks. Initially, agents will consist of only specialized nodes. As more environments are experimented with, however, nodes will be reused with greater frequency and less architecture change. Ultimately, the agent will consist of a virtual machine interface where tasks are conditioned with an axillary natural language interface and at times by overriding the policy with manual demonstrations.

# Results

# Analysis

# Discussion

# Conclusion

General conclusion here

## Questions

- **How generalized can Nodel RL agents become?** Will agent performance continue improving as it acquires skills specialized for increasingly diverse domains? Can a humanoid robot controlled by Nodal RL perform reliably in the general human activity domain?
- **Can an [imitator-imitator](https://arxiv.org/abs/1912.02875) framework be used to condition Nodel-RL agent behavior?**
- **How could transformer neural networks learn when to dream ($\NN_{transformer}(\opast)$) instead of employing a blind rolling average ($\text{Percieved Observation }\o$)?**
- **Can humans learn to dynamically condition agents to control their behavior?** Since high level conditioning data only provides a general handle on agent behavior, would a user commanding the agent in realtime be able to adjust his or her commands dynamically to more reliably achieve high level targets?
- **Will centralized shared-parameter multiagent training efficently improve performance?** Although parameter sharing in reinforcement learning [generally accelerates learning in MARL](https://arxiv.org/abs/2005.13625), is it worth the cost to centralize training these huge architectures?

# References

 - @book{Goodfellow-et-al-2016,
    title={Deep Learning},
    author={Ian Goodfellow and Yoshua Bengio and Aaron Courville},
    publisher={MIT Press},
    note={\url{http://www.deeplearningbook.org}},
    year={2016}
}
