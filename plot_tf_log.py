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
from scipy.stats import linregress
from tensorflow.python.summary.summary_iterator import summary_iterator
from tensorflow.core.util.event_pb2 import Event

def most_recent_log(dir):
    logs = glob.glob("/home/tijmen/cosmosage/models/" + dir + "/*/runs/*/*.0")
    sorted_logs = sorted(logs, key=lambda x: os.path.getmtime(x), reverse=True)
    if sorted_logs:
        return sorted_logs[0]
    else:
        raise IndexError("No directories found")


def plot_loss(file_paths, plot_type="default", detailed_pts_per_eval=10):
    plt.figure(figsize=(12, 8))

    for idx, file_path in enumerate(file_paths):
        # Load the event accumulator
        ea = event_accumulator.EventAccumulator(file_path)
        ea.Reload()

        # # Print all available items inside ea
        # print(ea.scalars.Keys())
        # # available keys are
        # # ['train/loss', 'train/learning_rate', 'train/epoch', 'eval/loss', 'eval/runtime', 'eval/samples_per_second', 'eval/steps_per_second']

        tloss = ea.scalars.Items("train/loss")
        eloss = ea.scalars.Items("eval/loss")
        lr = ea.scalars.Items("train/learning_rate")
        epoch = ea.scalars.Items("train/epoch")

        # Extract steps and loss values for training loss
        t_steps = np.array([s.step for s in tloss])
        t_losses = np.array([s.value for s in tloss])

        # Extract steps and loss values for evaluation loss
        e_steps = np.array([s.step for s in eloss])
        e_losses = np.array([s.value for s in eloss])

        # Extract steps and learning rate values
        lr_steps = np.array([s.step for s in lr])
        lr_values = np.array([s.value for s in lr])

        # Extract steps and epoch values
        epoch_steps = np.array([s.step for s in epoch])
        epoch_values = np.array([s.value for s in epoch])

        plt.figure(figsize=(12, 6))

        # Smooth the loss curve if plot_type is "logsmooth"
        if plot_type == "logsmooth":
            # gaussian smoothing using edge handling that doesn't change the length
            t_losses = scipy.ndimage.filters.gaussian_filter1d(
                t_losses, sigma=10, mode="nearest"
            )

        # Plotting
        if plot_type == "default":
            plt.plot(
                t_steps, t_losses, label=f"Training Loss (Run {idx+1})", color=f"C{idx}"
            )
            plt.plot(
                e_steps,
                e_losses,
                label=f"Evaluation Loss (Run {idx+1})",
                color=f"C{idx}",
                linestyle="dashed",
            )
        elif plot_type == "logsmooth":
            plt.semilogy(
                t_steps, t_losses, label=f"Training Loss (Run {idx+1})", color=f"C{idx}"
            )
            plt.semilogy(
                e_steps,
                e_losses,
                label=f"Evaluation Loss (Run {idx+1})",
                color=f"C{idx}",
                linestyle="dashed",
            )
        elif plot_type == "detailed":
            # Bin the loss values
            bin_size = int(len(t_losses) / (detailed_pts_per_eval * len(e_losses)))
            num_bins = int(len(t_losses) / bin_size)
            t_losses_binned = np.mean(
                t_losses[: num_bins * bin_size].reshape(-1, bin_size), axis=1
            )
            t_steps_binned = np.mean(
                t_steps[: num_bins * bin_size].reshape(-1, bin_size), axis=1
            )

            # Calculate error bars
            t_losses_std = np.std(
                t_losses[: num_bins * bin_size].reshape(-1, bin_size), axis=1
            ) / np.sqrt(bin_size)

            # Plotting
            plt.errorbar(
                t_steps_binned,
                t_losses_binned,
                yerr=t_losses_std,
                label=f"Training Loss (Run {idx+1})",
                color=f"C{idx}",
                capsize=3,
            )
            plt.plot(
                e_steps,
                e_losses,
                label=f"Evaluation Loss (Run {idx+1})",
                color=f"C{idx}",
                linestyle="dashed",
            )
            plt.ylabel("Loss")

            # label each evaluation point with the epoch number
            for i, e_loss in enumerate(e_losses):
                epoch_number = epoch_values[np.where(epoch_steps == e_steps[i])[0][0]]
                plt.text(
                    e_steps[i],
                    e_loss,
                    f"Epoch: {epoch_number:.2f}",
                    color=f"C{idx}",
                    fontsize=9,
                )

            plt.grid()

            # Plotting learning rate on the other axis
            ax2 = plt.gca().twinx()
            ax2.plot(lr_steps, lr_values, label="Learning Rate", color="red", alpha=0.15)
            ax2.set_ylabel("Learning Rate")

        elif plot_type == "slopes":
            num_segments = 10
            segment_length = len(t_steps) // num_segments

            # Initialize arrays to store slopes and midpoints of each segment
            slopes = np.zeros(num_segments)
            midpoints = np.zeros(num_segments)
            indices = np.zeros(num_segments)
            avg_losses = np.zeros(num_segments)

            # Calculate slopes for each segment and determine the midpoints based on the fit
            for i in range(num_segments):
                start_idx = i * segment_length
                end_idx = (i + 1) * segment_length if i != num_segments - 1 else len(t_steps)
                indices[i] = (start_idx+end_idx)/2

                segment_steps = t_steps[start_idx:end_idx]
                segment_losses = t_losses[start_idx:end_idx]

                avg_losses[i] = np.mean(segment_losses)

                slope, intercept, _, _, _ = linregress(segment_steps, segment_losses)
                slopes[i] = slope

                # Calculate midpoint based on the fit
                avg_loss = (np.max(segment_losses) + np.min(segment_losses)) / 2
                x_mid = (avg_loss - intercept) / slope
                midpoints[i] = x_mid

            # Create subplots
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 4))

            # Plot slopes in the top subplot
            ax1.plot(indices, slopes, label="Slopes", color="blue")
            ax1.set_xlabel("Steps")
            ax1.set_ylabel("Slope")
            ax1.grid()

            # Plot training loss in the bottom subplot
            ax2.plot(indices, avg_losses, label=f"Training Loss", color="green")
            ax2.set_xlabel("Steps")
            ax2.set_ylabel("Training Loss")
            ax2.grid()


    plt.xlabel("Steps")
    plt.legend()
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
