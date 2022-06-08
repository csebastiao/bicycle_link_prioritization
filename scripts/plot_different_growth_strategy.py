# -*- coding: utf-8 -*-
"""
Sandbox for plotting various things
"""

from matplotlib import pyplot as plt
import pickle


if __name__ == "__main__":
    f_list = ['../data/s2000_copenhagen_built_connected_additive_bf200_directness',
              '../data/s2000_copenhagen_built_connected_additive_bf200_random',
              '../data/s2000_copenhagen_built_connected_additive_bf200_betweenness',
              '../data/s2000_copenhagen_built_connected_subtractive_bf200_directness',
              '../data/s2000_copenhagen_built_connected_subtractive_bf200_random',
              '../data/s2000_copenhagen_built_connected_subtractive_bf200_betweenness']
    coverage_list = []
    directness_list = []
    for folder_name in f_list:
        with open(folder_name + "/arrcov.pickle", "rb") as fp:
            coverage_list.append(pickle.load(fp))
        with open(folder_name + "/arrdir.pickle", "rb") as fp:
            directness_list.append(pickle.load(fp))
    coverage_list[3].reverse()
    directness_list[3].reverse()
    coverage_list[4].reverse()
    directness_list[4].reverse()
    coverage_list[5].reverse()
    directness_list[5].reverse()
    for vlist in coverage_list:
        print(len(vlist), vlist[0], vlist[-1])
    for vlist in directness_list:
        print(len(vlist), vlist[0], vlist[-1])
    fig, axs = plt.subplots(1, 2, figsize=(20, 10))
    axs[0].set_xlabel("Step")
    axs[0].set_ylabel("Coverage")
    axs[1].set_xlabel("Step")
    axs[1].set_ylabel("Linkwise directness")
    
    #TODO: Put same color different tone for additive and subtractive
    axs[0].plot(range(len(coverage_list[0])), coverage_list[0],
                linewidth=5, label='Additive, Directness',
                color='salmon')
    axs[1].plot(range(len(directness_list[0])), directness_list[0],
                linewidth=5, label='Additive, Directness',
                color='salmon')
    axs[0].plot(range(len(coverage_list[1])), coverage_list[1],
                linewidth=5, label='Additive, Random',
                color='lightblue')
    axs[1].plot(range(len(directness_list[1])), directness_list[1],
                linewidth=5, label='Additive, Random',
                color='lightblue')
    axs[0].plot(range(len(coverage_list[2])), coverage_list[2],
                linewidth=5, label='Additive, Betweenness',
                color='lightgreen')
    axs[1].plot(range(len(directness_list[2])), directness_list[2],
                linewidth=5, label='Additive, Betweenness',
                color='lightgreen')
    axs[0].plot(range(len(coverage_list[3])), coverage_list[3],
                linewidth=5, label='Subtractive, Directness',
                color='darkred')
    axs[1].plot(range(len(directness_list[3])), directness_list[3],
                linewidth=5, label='Subtractive, Directness',
                color='darkred')
    axs[0].plot(range(len(coverage_list[4])), coverage_list[4],
                linewidth=5, label='Subtractive, Random',
                color='darkblue')
    axs[1].plot(range(len(directness_list[4])), directness_list[4],
                linewidth=5, label='Subtractive, Random',
                color='darkblue')
    axs[0].plot(range(len(coverage_list[5])), coverage_list[5],
                linewidth=5, label='Subtractive, Betweenness',
                color='darkgreen')
    axs[1].plot(range(len(directness_list[5])), directness_list[5],
                linewidth=5, label='Subtractive, Betweenness',
                color='darkgreen')
    
    axs[0].legend()
    axs[1].legend()
    fig.suptitle('s2000 Copenhagen, bf200, built and connected')
    plt.tight_layout()