#!/bin/bash

BASEDIR=$(dirname "$0")
wandb sweep --project RayleighBenard-AE "$BASEDIR/sweep_autoencoder.yaml"
