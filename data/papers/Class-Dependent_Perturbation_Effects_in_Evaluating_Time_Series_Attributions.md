# Class-Dependent Perturbation Effects in Evaluating Time Series Attributions

## Authors
Gregor Baer, Isel Grau, Chao Zhang, Pieter Van Gorp

## Year
2025

## Categories
cs.LG

## Abstract
As machine learning models become increasingly prevalent in time series applications, Explainable Artificial Intelligence (XAI) methods are essential for understanding their predictions. Within XAI, feature attribution methods aim to identify which input features contribute the most to a model's prediction, with their evaluation typically relying on perturbation-based metrics. Through systematic empirical analysis across multiple datasets, model architectures, and perturbation strategies, we reveal previously overlooked class-dependent effects in these metrics: they show varying effectiveness across classes, achieving strong results for some while remaining less sensitive to others. In particular, we find that the most effective perturbation strategies often demonstrate the most pronounced class differences. Our analysis suggests that these effects arise from the learned biases of classifiers, indicating that perturbation-based evaluation may reflect specific model behaviors rather than intrinsic attribution quality. We propose an evaluation framework with a class-aware penalty term to help assess and account for these effects in evaluating feature attributions, offering particular value for class-imbalanced datasets. Although our analysis focuses on time series classification, these class-dependent effects likely extend to other structured data domains where perturbation-based evaluation is common.

## Source
http://arxiv.org/abs/2502.17022v2
