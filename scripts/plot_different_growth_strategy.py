# -*- coding: utf-8 -*-
"""
Sandbox for plotting various things
"""

from matplotlib import pyplot as plt
import pickle


if __name__ == "__main__":
    f1 = '../data/s2000_copenhagen_additive_bf200_relative_coverage'
    f2 = '../data/s2000_copenhagen_additive_bf200_random'
    f3 = '../data/s2000_copenhagen_subtractive_bf200_relative_coverage'
    f4 = '../data/s2000_copenhagen_subtractive_bf200_random'
    f_list = [f1, f2, f3, f4]
    coverage_list = []
    directness_list = []
    for folder_name in f_list:
        with open(folder_name + "/arrcov.pickle", "rb") as fp:
            coverage_list.append(pickle.load(fp))
        with open(folder_name + "/arrdir.pickle", "rb") as fp:
            directness_list.append(pickle.load(fp))
    coverage_list[2].reverse()
    directness_list[2].reverse()
    coverage_list[3].reverse()
    directness_list[3].reverse()
    fig, axs = plt.subplots(1, 2, figsize=(20, 10))
    axs[0].set_xlabel("Step")
    axs[0].set_ylabel("Coverage")
    axs[1].set_xlabel("Step")
    axs[1].set_ylabel("Linkwise directness")
    
    axs[0].plot(range(len(coverage_list[0])), coverage_list[0],
                linewidth=5, label='Additive, Relative coverage')
    axs[1].plot(range(len(directness_list[0])), directness_list[0],
                linewidth=5, label='Additive, Relative coverage')
    axs[0].plot(range(len(coverage_list[1])), coverage_list[1],
                linewidth=5, label='Additive, Random')
    axs[1].plot(range(len(directness_list[1])), directness_list[1],
                linewidth=5, label='Additive, Random')
    axs[0].plot(range(len(coverage_list[2])), coverage_list[2],
                linewidth=5, label='Subtractive, Relative coverage')
    axs[1].plot(range(len(directness_list[2])), directness_list[2],
                linewidth=5, label='Subtractive, Relative coverage')
    axs[0].plot(range(len(coverage_list[3])), coverage_list[3],
                linewidth=5, label='Subtractive, Random')
    axs[1].plot(range(len(directness_list[3])), directness_list[3],
                linewidth=5, label='Subtractive, Random')
    
    axs[0].legend()
    axs[1].legend()
    fig.suptitle('s2000 Copenhagen, bf200')
    plt.tight_layout()