# Neural Network for Nanofluid Inference

This repository contains a physics-informed neural network (PINN) framework for inferring multiple physical parameters from nanofluid flow data in porous media.

## Overview

The model is designed to simultaneously estimate four key parameters:
- Ra (Rayleigh number)
- Ha (Hartmann number)
- Q (Heat Source/Sink parameter)
- Da (Darcy number)

It utilizes a ConvLSTM-based architecture to capture spatiotemporal features of the fluid flow while enforcing governing physical equations as loss constraints.

## Core Files

- train_and_infer_multi_v2.py: Main script for training the surrogate model and performing parameter inference.
- models.py: Definition of the neural network architecture (STNNN).
- data.py: Data loading and preprocessing for MATLAB (.mat) files.

## Documentation

For a detailed explanation of the model architecture, data pipeline, and the dual-phase loss strategy (Training vs. Inference), please refer to:
- TRAIN_AND_INFER_MULTI_V2_DETAILED_GUIDE.md

## Requirements

The required libraries are listed in requirements.txt. You can install them using:
pip install -r requirements.txt
