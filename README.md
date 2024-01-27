# MNIST Imange Classifier
## Overview

## Features

## Background

## Possible Improvements

## License
MIT

1. params[4]
2. grads[4]
3. momentum1[]
4. momentun2[]
 
Learning-rate (SGD): 0.01
Learning-rate (Adam): 0.001 (1e-3)

for idx, param in enumerate(params):
    <!-- params[idx] = params[idx] - lr * grads[idx] -->
    <!-- grads[idx] = grad[idx] + params[idx] * weight_decay (1e-5)>
    momentum1[idx] = 0.9 * momentum1[idx] + (1 - 0.1) * grad 
    momentum2[idx] = 0.999 * momentum2[idx] + (1 - 0.999) * grad * grad

    momentum1_ = momentum1[idx] / (1 - 0.9 ** step)
    momentum2_ = momentum2[idx] / (1 - 0.999 ** step)

    update = momentum1_ / sqrt(momentum2_ + 1e-8)
    params[idx] = params[idx] - lr * update