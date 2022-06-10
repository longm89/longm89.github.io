---
layout: essay
type: essay
title: Score-Based Generative Modeling through Stochastic Differential Equations
# All dates must be YYYY-MM-DD format!
date: 2022-06-10
published: true
labels:
  - Generative Model
  - Deep learning
---

# Introduction
In the article Score-based generative modeling through stochastic differential equations, the authors propose a new method of using stochastic differential equations (SDE) to smoothly transform a complex data distribution to a known prior distribution by slowly injecting noise. Their corresponding reverse-time SDEs transform the prior distribution back into the data distribution by slowly removing the noise. Their work generalizes two previous approaches: Score matching with Langevin dynamics (SMLD) and Denoising diffusion probabilistic modeling (DDPM), which in their framework, correspond to the discretization of two different SDEs. The main idea in these methods is to estimate the time-dependent gradients (scores) of the perturbed data distribution with neural networks, and then use these scores to draw samples using numerical inverse SDEs solvers. In addition, they propose a new method of Predictor-Corrector sampling that improves the sample quality.  They also link their inverse SDEs with neural ODEs, enabling exact likelihood computation and improving sampling efficiency. Finally, their score-based models allow them to solve inverse problems such as class-conditional generation, image inpainting and colorization. In the following, we first summarize their findings and then explain the experiments. 

# Score-based generative modeling through Stochastic Differential Equations            
## Perturbing data with SDEs       
We are given a dataset of i.i.d samples from an unknown data distribution $$p_0$$. The method constructs a diffusion process $$\{x(t)\}_{t = 0}^T$$ indexed by a continuous time variable $$t\in [0, T]$$, such that $$x(0) \sim p_0$$, and $$x(T) \sim p_T$$, a known prior distribution. For example, $$p_T$$ could be a Gaussian distribution with a fixed mean and variance. The diffusion process takes the form of solution to an Itô SDE:
	$$dx = f(x,t)dt + g(t)dw$$
where $$w$$ is the standard Wiener process, $$f(.,t): \mathbb R^r\rightarrow \mathbb R^d$$ is a vector valued function, called the drift coefficient of $$x(t)$$ and $$g(.):\mathbb R\rightarrow \mathbb R$$ is a scalar function, known as the diffusion coefficient of $$x(t).$$ The SDE has a unique strong solution as long as the coefficients are globally Lipschitz in both state and time. In this case, denote by $$p_t(x)$$ the probability density function of $$x(t)$$ and $$p_{st}(x(t)\mid x(s))$$ the transition kernel from $$x(s)$$ to $$x(t)$$ where $$0 \le s < t\le T.$$

## Examples

### First example

Fix $$\sigma$$ big enough, and choose the following SDE:
	$$dx = \sigma^t dw, \ \ \ \ t\in [0, 1]$$
In this case, 
	$$p_{0t}(x(t)\mid x(0)) = N(x(t); x(0), \frac{1}{2\log(\sigma)}(\sigma^{2t} - 1)I)$$
When $$\sigma$$ is large, the prior distribution $$p_{t = 1}$$ is:
	$$\int p_0(y)N(x, y, \frac{1}{2\log \sigma}(\sigma^2 - 1) I) dy \cong N(x, 0, \frac{1}{2\log \sigma}(\sigma^2 - 1) I)$$
which is independent of the data distribution (given $$\sigma$$) and easy to sample from.

### Second example

Choose a function $$\sigma(t)$$, the process, called Variance Exploding SDE is defined by:
	$$dx = \sqrt{\frac{d[\sigma^2(t)]}{dt} dw}$$

### Third example

Choose a function $$\beta(t)$$, the process, called Variance Preserving SDE is defined by:
	$$dx = -\frac{1}{2}\beta(t)xdt + \sqrt{\beta(t)}dw$$

The discretization of the last two SDEs correpond to finite noise pertubation used in SMLD and DDPM. The SDE of the second example always gives a process with exploding variance when $$t\rightarrow \infty$$ while the SDE of the third example gives a process with a fixed variance of $$1$$ when the initial distribution has unit variance. Hence they are called Variance Exploding SDE and Variance Preserving SDE.  Inspired by the Variance Preserving SDE, the authors propose a new type of SDE that performs well with likelihood, called sub-VP SDE.

## Using reversed SDEs to generate images

A result from Anderson (1982) states that the reverse of a diffusion process is also a diffusion process, running backwards in time and given by a reverse-time SDE:
	$$dx = [f(x,t) - g(t)^2\nabla_x \log p_t(x)] dt + g(t)d\overline{w}$$
where $$d\overline{w}$$ is a standard Wiener process when time flows backward from $$T$$ to $$0$$, and $$dt$$ is an infinitesimal negative timestep. If we know the score of each marginal distribution, $$\nabla_x \log p_t(x)$$, we could sample from $$x(T) \sim p_T$$ and use the reverse-time SDE to obtain samples from $$p_0.$$

### Estimating the scores for the SDE

To estimate $$\nabla_x \log p_t(x),$$ we could train a time-dependent score-based model $$s_\theta(x,t)$$ using denoising score matching objectives:
	$$\theta^* = arg\min_\theta \mathbb E_t[\lambda(t)\mathbb E_{x(0)}\mathbb E_{x(t)\mid x(0)}[\|s_\theta(x(t),t) - \nabla_{x(t)}\log p_{0t}(x(t)\mid x(0))\|_2^2]]$$
where $$\lambda: [0, T]\rightarrow \mathbb R_{>0}$$ is a positive weighting function, $$t$$ uniformly sampled over $$[0, T]$$, $$x(0)\sim p_0(x), x(t) \sim p_{0t}(x(t)\mid x(0)).$$ With sufficient data and model capacity, score matching ensures that the optimal solution to the above equation, denoted by $$s_{\theta^*}(x,t) \sim \nabla_x\log p_t(x)$$ for almost all $$x$$ and $$t$$. 

### Solving the reverse SDE

After having the scores, we can use them to generate samples by solving the reverse SDE process. One of the numerial SDE solvers is the Euler Maruyama method. It dicretizes the SDE using finite time steps and small Gaussian noise. In particular, choose a negative time step $$\Delta_t \cong 0$$, initialize $$t \leftarrow T$$, and iterates the following procedure until $$t\cong 0$$:
$$\Delta x \leftarrow [f(x,t) - g^2(t)s_\theta(x,t)]\Delta t + g(t)\sqrt{\|\Delta t\|} z_t$$
$$x\leftarrow x + \Delta x$$
$$t\leftarrow t + \Delta t$$
Here $$z_t \sim N(0, I).$$ Moreover, to improve the quality of sampling, the authors propose the method of Predictor-Corrector samplers. At each step of the Predictor-Corrector sampler, they first use a SDE solver, such as above, to predict $$x(t +\Delta t)$$ based on the current sample $$x(t).$$ Next, they run several corrector step to improve the sample $$x(t + \Delta t)$$ according to the score-based model $$s_\theta(x, t + \Delta t),$$ so that $$x(t + \Delta t)$$ becomes a higher quality sample from $$p_{t + \Delta t}(x).$$ The corrector can be any MCMC procedure that solely relies on the score function. For example, Langevin MCMC operates by running the following iteration rule for $$i = 1,2,...:$$
	$$x_{i+1} = x_i + \epsilon \nabla_x \log p(x_i)  + \sqrt{2\epsilon}z_i$$
where $$z_i \sim N(0, I)$$, $$\epsilon > 0$$ is the step size and $$x_1$$ is initialized from any prior distribution. With this Predictor-Corrector methods and better architecture models, they achieve state of the art sample quality on CIFAR-10, outperforming best GAN model to date (StyleGAN2 + ADA). 

## Probability flow ODE and Controllable generation

They show that their SDE determines an ODE, once the scores are known:
	$$dx = [f(x,t) - \frac{1}{2}g(t)^2 \nabla_x \log p_t(x)] dt$$

Using this equation and an instantaneous change of variables formula, they can compute the unknown density from a known prior density $$p_T$$ with numerical ODE solvers. This method achieves state of the art log-likelihoods on uniformly dequantized CIFAR-10 images, even without maximum likelihood training. 

In the last sections of the paper, they also show how to use score-based generative models to solve inverse problems, with applications in class-conditional generation, image inpainting, and colorization.  

# Experiments on a new dataset

For the experiment, we take a dataset of cats and dogs containing 23,000 images of normal quality, and under various real life settings. We use the SDE:
	$$dx=\sigma^t dw, \ \ \ \ t \in [0, 1]$$

To speed up the training process, we resize the images to $$64 \times 64$$ and normalize them before training. To train for the scores, we use a time-dependent score-based model built upon the U-net architecture that they use for the MNIST data set and change the number of channels in the model to work with RGB images.

As recommended in another article, $$\sigma$$ is chosen to be $$391$$, a little bit less than the maximum $$L^2$$ distance between any two normalized images in the data set to get diversed samples. 

We train the model for 200 epochs and save a check point every 20 epochs. We try each check point to check for the quality of images. We could also use FID scores, computed with tensorflow$$\_$$gan to find the best check point. 

For the Euler-Maruyama sampler, we choose 1000 steps as in the paper of the author and others.  

To apply the predictor-corrector methods, we choose the number of steps to be 1000 as for some datasets in the paper. The signal-to-noise is also chosen to be 0.075 as in the paper for the data set bedroom$$/$$church$$\_$$outdoor. For a more optimal result, it could be searched over a grid that increments at 0.01.

The link to the colab is: https://colab.research.google.com/drive/1TDKfMmJA581gQ-aFw1fwlx6-o9JJI1Xd?usp=sharing

The images generated are not as good as their images, because I use the code from the tutorial for MNIST data set and apply their models and methods for the new data set of cats and dogs which are in colors and under various light and setting conditions. Moreover, in their paper, they applied a lot of techniques and new model architecture design to achieve high sample quality images. 
