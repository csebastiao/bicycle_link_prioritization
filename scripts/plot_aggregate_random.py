# -*- coding: utf-8 -*-
"""

"""

import os
import pickle
from matplotlib import pyplot as plt
import matplotlib as mpl
import numpy as np
import seaborn as sns
from blp import utils

if __name__ == "__main__":
    add = '../data/s2000_copenhagen_built_connected_additive_bf200_random_folder'
    sub = '../data/s2000_copenhagen_built_connected_subtractive_bf200_random_folder'
    add_folders = os.listdir(add)
    sub_folders = os.listdir(sub)
    
    mpl.rcParams.update({'font.size': 20})
    sns.set_style('ticks')
    palette = sns.color_palette('Paired')
    fig, axs = plt.subplots(1, 2, figsize=(20, 10))
    sns.despine(fig=fig)
    coverage_list = []
    for f in add_folders:
        with open(add + '/' + f + "/arrcov.pickle", "rb") as fp:
            val = pickle.load(fp)
            coverage_list.append(val)
            # axs[0].plot(range(len(val)), val, linewidth=1, alpha=0.08,
            #             color='blue')
    average_coverage = np.mean(coverage_list, axis=0)
    std_coverage = np.std(coverage_list, axis=0)
    

    axs[0].set_xlabel("Step")
    axs[0].set_xlim([-1, 203])
    axs[0].set_ylim([1*10**6, 9*10**6])
    axs[0].set_ylabel("Coverage ($m^2$)")
    axs[0].plot(range(len(average_coverage)), average_coverage,
                color=palette[7], label='Additive, Random', linewidth=4)
    axs[0].fill_between(range(len(average_coverage)),
                     average_coverage-std_coverage,
                     average_coverage+std_coverage, color=palette[7],
                     alpha=0.4)
    coverage_list = []
    for f in sub_folders:
        with open(sub + '/' + f + "/arrcov.pickle", "rb") as fp:
            val = pickle.load(fp)
            val.reverse()
            coverage_list.append(val)
            # axs[0].plot(range(len(val)), val, linewidth=1, alpha=0.08,
            #             color='red')
    average_coverage = np.mean(coverage_list, axis=0)
    std_coverage = np.std(coverage_list, axis=0)
    axs[0].plot(range(len(average_coverage)), average_coverage,
                color=palette[9], label='Subtractive, Random', linewidth=4)
    axs[0].fill_between(range(len(average_coverage)),
                     average_coverage-std_coverage,
                     average_coverage+std_coverage, color=palette[9],
                     alpha=0.4)
    
    directness_list = []
    for f in add_folders:
        with open(add + '/' + f + "/arrdir.pickle", "rb") as fp:
            val = pickle.load(fp)
            directness_list.append(val)
            # axs[1].plot(range(len(val)), val, linewidth=1, alpha=0.08,
            #             color='blue')
    average_directness = np.mean(directness_list, axis=0)
    std_directness = np.std(directness_list, axis=0)
    
    axs[1].set_xlabel("Step")
    axs[1].set_xlim([-1, 203])
    axs[1].set_ylim([0.600, 0.9])
    axs[1].set_ylabel("Directness")
    axs[1].plot(range(len(average_directness)), average_directness,
                color=palette[7], label='Additive, Random', linewidth=4)
    axs[1].fill_between(range(len(average_directness)),
                     average_directness-std_directness,
                     average_directness+std_directness, color=palette[7],
                     alpha=0.4)
    directness_list = []
    for f in sub_folders:
        with open(sub + '/' + f + "/arrdir.pickle", "rb") as fp:
            val = pickle.load(fp)
            val.reverse()
            directness_list.append(val)
            # axs[1].plot(range(len(val)), val, linewidth=1, alpha=0.08,
            #             color='red')
    average_directness = np.mean(directness_list, axis=0)
    std_directness = np.std(directness_list, axis=0)
    axs[1].plot(range(len(average_directness)), average_directness,
                color=palette[9], label='Subtractive, Random', linewidth=4)
    axs[1].fill_between(range(len(average_directness)),
                     average_directness-std_directness,
                     average_directness+std_directness, color=palette[9],
                     alpha=0.4)
    
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
    for lis in coverage_list:
        print(utils.get_area_under_curve(
            lis, normalize_x=True, normalize_y=True))
    for lis in directness_list:
        print(utils.get_area_under_curve(
            lis, normalize_x=True, normalize_y=False))
    axs[1].plot(range(len(directness_list[0])), directness_list[0],
                linewidth=4, label='Additive, Directness',
                color=palette[0])
    axs[1].plot(range(len(directness_list[1])), directness_list[1],
                linewidth=4, label='Additive, Relative coverage',
                color=palette[2])
    axs[1].plot(range(len(directness_list[2])), directness_list[2],
                linewidth=4, label='Additive, Betweenness',
                color=palette[4])
    axs[1].plot(range(len(directness_list[3])), directness_list[3],
                linewidth=4, label='Subtractive, Directness',
                color=palette[1])
    axs[1].plot(range(len(directness_list[4])), directness_list[4],
                linewidth=4, label='Subtractive, Relative coverage',
                color=palette[3])
    axs[1].plot(range(len(directness_list[5])), directness_list[5],
                linewidth=4, label='Subtractive, Betweenness',
                color=palette[5])
    
    axs[0].plot(range(len(coverage_list[0])), coverage_list[0],
                linewidth=4, label='Additive, Directness',
                color=palette[0])
    axs[0].plot(range(len(coverage_list[1])), coverage_list[1],
                linewidth=4, label='Additive, Relative coverage',
                color=palette[2])
    axs[0].plot(range(len(coverage_list[2])), coverage_list[2],
                linewidth=4, label='Additive, Betweenness',
                color=palette[4])
    axs[0].plot(range(len(coverage_list[3])), coverage_list[3],
                linewidth=4, label='Subtractive, Directness',
                color=palette[1])
    axs[0].plot(range(len(coverage_list[4])), coverage_list[4],
                linewidth=4, label='Subtractive, Relative coverage',
                color=palette[3])
    axs[0].plot(range(len(coverage_list[5])), coverage_list[5],
                linewidth=4, label='Subtractive, Betweenness',
                color=palette[5])
    
    axs[0].legend()
    # axs[1].legend()
    

    