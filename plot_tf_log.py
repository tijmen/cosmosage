#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Helper functions for monitoring the progress of a training loop.
"""
import argparse
import matplotlib.pyplot as plt
import numpy as np
import scipy
from tensorboard.backend.event_processing import event_accumulator
import os
import glob


def most_recent_log(dir):
    logs = glob.glob("/home/tijmen/cosmosage/models/" + dir + "/*/runs/*/*.0")
    sorted_logs = sorted(logs, key=lambda x: os.path.getmtime(x), reverse=True)
    if sorted_logs:
        return sorted_logs[0]
    else:
        raise IndexError("No directories found")


def plot_loss(file_paths, logsmooth=False):
    plt.figure(figsize=(12, 8))

    for idx, file_path in enumerate(file_paths):
        # Load the event accumulator
        ea = event_accumulator.EventAccumulator(file_path)
        ea.Reload()

        # Extract training and evaluation losses
        tloss = ea.scalars.Items("train/loss")
        eloss = ea.scalars.Items("eval/loss")

        # Extract steps and loss values for training loss
        t_steps = np.array([s.step for s in tloss])
        t_losses = np.array([s.value for s in tloss])

        # Extract steps and loss values for evaluation loss
        e_steps = np.array([s.step for s in eloss])
        e_losses = np.array([s.value for s in eloss])

        # Smooth the loss curve
        if logsmooth:
            # gaussian smoothing using edge handling that doesn't change the length
            t_losses = scipy.ndimage.filters.gaussian_filter1d(
                t_losses, sigma=10, mode="nearest"
            )

        # Plotting
        plotting_function = plt.semilogy if logsmooth else plt.plot
        plotting_function(
            t_steps, t_losses, label=f"Training Loss (Run {idx+1})", color=f"C{idx}"
        )
        plotting_function(
            e_steps,
            e_losses,
            label=f"Evaluation Loss (Run {idx+1})",
            color=f"C{idx}",
            linestyle="dashed",
        )

    plt.xlabel("Steps")
    plt.ylabel("Loss")
    plt.title("Training and Evaluation Loss over Time")
    plt.legend()
    plt.grid()
    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Plot training and evaluation loss from TensorFlow event files."
    )
    parser.add_argument(
        "file_paths",
        nargs="+",
        type=str,
        help="Path(s) to the TensorFlow event file(s)",
    )

    args = parser.parse_args()

    plot_loss(args.file_paths)


if __name__ == "__main__":
    main()
