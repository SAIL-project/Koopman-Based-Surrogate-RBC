#!/bin/bash

BASEDIR=$(dirname "$0")
wandb sweep --project RayleighBenard-LRAN "$BASEDIR/sweep_lran.yaml"
