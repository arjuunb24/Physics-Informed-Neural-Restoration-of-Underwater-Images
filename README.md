# Physics-Informed-Neural-Restoration-of-Underwater-Images


This is a research project currently in progress.


## Overview:


Underwater image restoration remains a fundamentally ill-posed inverse problem due to wavelength-dependent absorption, forward and back scattering, and spatially varying illumination, which jointly induce severe color cast, contrast degradation, and haze. Classical model-based approaches address these effects through explicit priors derived from simplified underwater imaging models, such as dark channel statistics or color constancy assumptions, but often fail under complex real-world conditions where model parameters are scene-dependent and difficult to estimate. Conversely, recent learning-based methods achieve strong empirical performance by leveraging large-scale data and deep convolutional architectures, yet remain constrained by discrete pixel representations, limited generalization across water types, and weak physical interpretability.

Implicit Neural Representations (INRs) provide an alternative formulation by parameterizing images as continuous functions mapping spatial coordinates to radiance values, enabling resolution-agnostic reconstruction and improved spectral bias control. Recent INR-based restoration and super-resolution methods demonstrate strong capacity for high-frequency detail recovery through coordinate-based learning augmented with Fourier feature mappings or sinusoidal activations. However, most existing INR formulations are agnostic to domain-specific degradation processes, relying primarily on reconstruction-driven objectives that inadequately constrain color correction and scattering artifacts in underwater environments.

Parallel to representational advances, physics-informed learning has re-emerged as an effective mechanism for regularizing inverse problems by embedding analytical priors directly into the training objective. In underwater imaging, such priors are commonly instantiated through dark channel constraints, gray-world assumptions, or simplified image formation models, which act as soft regularizers rather than explicit forward solvers. While recent hybrid methods combine deep networks with physics-inspired losses, they remain largely pixel-centric and do not exploit the continuous nature of scene radiance.

In this work, we unify these directions by proposing a physics-informed implicit neural restoration framework for underwater images. The restoration task is formulated as learning a continuous coordinate-based function mapping spatial locations to restored RGB values using sinusoidal representation networks (SIREN) equipped with Fourier feature encodings to capture high-frequency structure. To address wavelength-dependent degradation, we introduce channel-specific prediction heads with adaptive capacity, allowing differential modeling of attenuation across color channels. Training is guided by a composite objective that integrates reconstruction fidelity, perceptual consistency, and domain-specific physics priors, thereby constraining the solution space without explicit parameter estimation. This formulation enables memory-efficient optimization through sparse coordinate sampling while supporting full-resolution reconstruction at inference time.

By integrating physics-based constraints into an implicit representation framework, our approach bridges discrete data-driven restoration and continuous model-based regularization, offering a scalable and interpretable solution for underwater image enhancement.
