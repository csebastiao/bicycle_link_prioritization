# -*- coding: utf-8 -*-
"""
Sandbox for plotting various things
"""

from matplotlib import pyplot as plt
import matplotlib as mpl
import pandas as pd
import seaborn as sns
import pickle


if __name__ == "__main__":
    f_list = ['../data/s2000_copenhagen_built_connected_additive_bf200_directness',
              '../data/s2000_copenhagen_built_connected_additive_bf200_relative_coverage',
              '../data/s2000_copenhagen_built_connected_additive_bf200_betweenness',
              '../data/s2000_copenhagen_built_connected_subtractive_bf200_directness',
              '../data/s2000_copenhagen_built_connected_subtractive_bf200_relative_coverage',
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
    # coverage_list[6].reverse()
    # directness_list[6].reverse()
    # coverage_list[7].reverse()
    # directness_list[7].reverse()
    # for vlist in coverage_list:
    #     print(len(vlist), vlist[0], vlist[-1])
    # for vlist in directness_list:
    #     print(len(vlist), vlist[0], vlist[-1])



    mpl.rcParams.update({'font.size': 25})
    sns.set_style('ticks')
    fig, axs = plt.subplots(figsize=(20, 10))
    palette = sns.color_palette('Paired')
    sns.despine(fig=fig)
    axs.set_xlabel("Step")
    axs.set_xlim([-1, 203])
    axs.set_ylim([1*10**6, 9*10**6])
    axs.set_ylabel("Coverage ($m^2$)")

    axs.plot(range(len(coverage_list[0])), coverage_list[0],
                linewidth=4, label='Additive, Directness',
                color=palette[0])
    axs.plot(range(len(coverage_list[1])), coverage_list[1],
                linewidth=4, label='Additive, Relative coverage',
                color=palette[2])
    axs.plot(range(len(coverage_list[2])), coverage_list[2],
                linewidth=4, label='Additive, Betweenness',
                color=palette[4])
    axs.plot(range(len(coverage_list[3])), coverage_list[3],
                linewidth=4, label='Subtractive, Directness',
                color=palette[1])
    axs.plot(range(len(coverage_list[4])), coverage_list[4],
                linewidth=4, label='Subtractive, Relative coverage',
                color=palette[3])
    axs.plot(range(len(coverage_list[5])), coverage_list[5],
                linewidth=4, label='Subtractive, Betweenness',
                color=palette[5])
    axs.legend()
    plt.tight_layout()





    # fig, axs = plt.subplots(1, 2, figsize=(20, 10))
    # axs[0].set_xlabel("Step")
    # axs[0].set_ylabel("Coverage")
    # axs[1].set_xlabel("Step")
    # axs[1].set_ylabel("Linkwise directness")

    # axs[0].plot(range(len(coverage_list[0])), coverage_list[0],
    #             linewidth=5, label='Additive, Directness',
    #             color='salmon')
    # axs[1].plot(range(len(directness_list[0])), directness_list[0],
    #             linewidth=5, label='Additive, Directness',
    #             color='salmon')
    # axs[0].plot(range(len(coverage_list[1])), coverage_list[1],
    #             linewidth=5, label='Additive, Random',
    #             color='lightblue')
    # axs[1].plot(range(len(directness_list[1])), directness_list[1],
    #             linewidth=5, label='Additive, Random',
    #             color='lightblue')
    # axs[0].plot(range(len(coverage_list[2])), coverage_list[2],
    #             linewidth=5, label='Additive, Betweenness',
    #             color='lightgreen')
    # axs[1].plot(range(len(directness_list[2])), directness_list[2],
    #             linewidth=5, label='Additive, Betweenness',
    #             color='lightgreen')
    # axs[0].plot(range(len(coverage_list[3])), coverage_list[3],
    #             linewidth=5, label='Subtractive, Directness',
    #             color='darkred')
    # axs[1].plot(range(len(directness_list[3])), directness_list[3],
    #             linewidth=5, label='Subtractive, Directness',
    #             color='darkred')
    # axs[0].plot(range(len(coverage_list[4])), coverage_list[4],
    #             linewidth=5, label='Subtractive, Random',
    #             color='darkblue')
    # axs[1].plot(range(len(directness_list[4])), directness_list[4],
    #             linewidth=5, label='Subtractive, Random',
    #             color='darkblue')
    # axs[0].plot(range(len(coverage_list[5])), coverage_list[5],
    #             linewidth=5, label='Subtractive, Betweenness',
    #             color='darkgreen')
    # axs[1].plot(range(len(directness_list[5])), directness_list[5],
    #             linewidth=5, label='Subtractive, Betweenness',
    #             color='darkgreen')
    
    # axs[0].legend()
    # axs[1].legend()
    # fig.suptitle('s2000 Copenhagen, bf200, built and connected')
    # plt.tight_layout()