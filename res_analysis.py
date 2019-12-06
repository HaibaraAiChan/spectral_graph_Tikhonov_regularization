import numpy as np

import cv2
import pandas as pd
import matplotlib.pyplot as plt
import os

def draw_one_image(file):

    df = pd.read_csv(file, delimiter=',')
    fig, ax = plt.subplots()

    df = df.T
    df.columns = ['crop_noisy', 'crop_graph', 'crop_traditional']
    file = os.path.basename(file)
    plot = df.plot(title=file[:-4])
    ax.grid(linestyle='--', linewidth='0.5', color='black')
    fig = plot.get_figure()
    fig.savefig(file[:-4] + '.png', bbox_inches='tight')
    plt.show()


def draw_line_PSNR(file, title):
    df = pd.read_csv(file, delimiter=',')
    fig, ax = plt.subplots()
    df = df.T
    df.columns = ['crop_noisy', 'crop_graph', 'crop_traditional']
    # plt.title(title)
    # ax = plt.gca()
    file = os.path.basename(file)

    plot = df.plot(title=file[:-4],  ax=ax)
    ax.grid(linestyle='--', linewidth='0.5', color='black')
    fig = plot.get_figure()
    fig.savefig(file[:-4] + '.png', bbox_inches='tight')
    plt.show()


def draw_line_graph(file, title):
    df = pd.read_csv(file, delimiter=',')
    plt.title(title)
    ax = plt.gca()
    #
    # df.plot(kind='line', x='name', y='PSNR', color='cyan', ax=ax)
    # df.plot(kind='line', x='name', y='SSIM', color='blue', style='.-', markevery=1, marker='o', markerfacecolor='blue',
    #         ax=ax)
    # df.plot(kind='line', x='name', y='PSIM', color='red', style='.-', markevery=1, marker='o', markerfacecolor='red',
    #         ax=ax)
    # df.plot(kind='line', x='name', y='SIFT', color='green', style='.-', markevery=1, marker='o',
    #         markerfacecolor='green', ax=ax)
    # df.plot(kind='line', x='name', y='EMDI', color='magenta', style='.-', markevery=1, marker='o',
    #         markerfacecolor='magenta', ax=ax)
    # df.plot(linestyle='-', markevery=1, marker='o', markerfacecolor='yellow')

    plt.show()

    return


if __name__ == '__main__':
    file = './res/PSNR.csv'
    file_1 = './res/res.csv'
    file_2 = './res/res_graph_filter.csv'
    file_3 = './res/res_traditional_filter.csv'
    title_1 = 'similarity: cropped image V.S. cropped noisy image'
    title_2 = 'similarity: cropped image V.S. graph filtered image'
    title_3 = 'similarity: cropped image V.S. traditional filtered image'

    # draw_line_PSNR(file, 'PSNR')
    # draw_line_PSNR('coffee.csv', 'not PSNR')
    # draw_line_PSNR('camera.csv', 'not PSNR')
    # draw_line_PSNR('coins.csv', 'not PSNR')
    draw_line_PSNR('page.csv', 'not PSNR')


    # draw_line_graph(file_1, title_1)
    # draw_line_graph(file_2, title_2)
    # draw_line_graph(file_3, title_3)

    # draw_one_image('coffee.csv')
    # draw_one_image('camera.csv')
    # draw_one_image('coins.csv')
    # draw_one_image('page.csv')
    print()
