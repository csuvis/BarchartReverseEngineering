#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import matplotlib

matplotlib.use("Agg")
import random
import string
import itertools
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib import cm
from tqdm import tqdm

# In[2]:


### random bar chart configuration ###
bar_dirction_list = ["horizontal", "vertical"]
bar_per_loc_list = [1, 2]  # how many bars are there in one ordinal position
bar_num_min = 2
bar_num_max = 3
bar_value_min = 10
bar_value_max = 200

axis_label_length_min = 5
axis_label_length_max = 15
axis_label_size_min = 15
axis_label_size_max = 18

legend_position = ["top", "right", "bottom"]
legend_length_min = 3
legend_length_max = 6
legend_size_min = 15
legend_size_max = 18

ticks_label_length_min = 1
ticks_label_length_max = 5
ticks_label_size_min = 14
ticks_label_size_max = 16

dpi_min = 50
dpi_max = 80
figsize_min = 6
figsize_max = 8

title_length_min = 5
title_length_max = 15
title_size_min = 18
title_size_max = 20
title_location = ["left", "center", "right"]

# fonts_list = font_manager.findSystemFonts()
fonts_list = font_manager.findSystemFonts()[0:2]

styles = plt.style.available
if 'dark_background' in styles:
    styles.remove('dark_background')


# In[3]:


def get_random_plot(filename):
    """
    Random bar chart generation method.
    
    Inputs:
    filename(string):  name of the chart image which will be saved
    """

    # Outputs
    ax = None
    fig = None
    bars = []
    data = None
    title = None
    legend = None
    axis_ticks = None
    axis_label = None

    # plot style 
    style = random.choice(styles)
    plt.style.use(style)

    # resolution and figure size
    dpi = random.randint(dpi_min, dpi_max)
    figsize = [random.randint(figsize_min, figsize_max), random.randint(figsize_min, figsize_max)]
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    # bars setting
    bar_num = random.randint(bar_num_min, bar_num_max)
    bar_per_loc = random.choice(bar_per_loc_list)
    bar_direction = random.choice(bar_dirction_list)
    bar_width = random.choice([0.6, 0.7, 0.8, 0.9])

    # generate random data according to bar_per_loc 
    bar_value_range = random.randint(int(bar_value_max * 0.2), bar_value_max)
    y = np.random.rand(bar_per_loc, bar_num)
    y = bar_value_min + y * bar_value_range
    y = y.astype(np.int32)

    bar_dist = random.choice([0.5, 1, 1.5])
    bar_start = random.choice([0, 0.4, 0.8])
    # x stores the start position of every group of bars(bar_per_loc bars in one group)
    x = [bar_start]
    last = bar_start
    for i in range(bar_num - 1):
        last = last + bar_width * bar_per_loc + bar_dist
        x.append(last)
    x = np.array(x)
    data = (x, y)

    if bar_direction == "horizontal":
        bar_generator = ax.barh
        set_hticks = ax.set_yticks
        set_hticklabels = ax.set_yticklabels
        get_vticklabels = ax.get_xticklabels
        # if the bars are horizontal, invert the y axis
        ax.invert_yaxis()
    else:
        bar_generator = ax.bar
        set_hticks = ax.set_xticks
        set_hticklabels = ax.set_xticklabels
        get_vticklabels = ax.get_yticklabels

    colors = cm.jet(np.random.rand(bar_per_loc))
    linewidth = random.choice([0, 1])
    for i in range(bar_per_loc):
        temp = bar_generator(x + bar_width * i, y[i], bar_width, align="edge", color=colors[i], linewidth=linewidth,
                             edgecolor="black")
        bars.append(temp)

    # fonts and fonts size
    font = random.choice(fonts_list)
    title_size = random.choice(range(title_size_min, title_size_max + 1))
    axis_label_size = random.choice(range(axis_label_size_min, axis_label_size_max + 1))
    ticks_label_size = random.choice(range(ticks_label_size_min, ticks_label_size_max + 1))
    legend_size = random.choice(range(legend_size_min, legend_size_max + 1))
    ticks_label_font = font_manager.FontProperties(fname=font, size=ticks_label_size)
    title_font = font_manager.FontProperties(fname=font, size=title_size)
    axis_label_font = font_manager.FontProperties(fname=font, size=axis_label_size)
    legend_font = font_manager.FontProperties(fname=font, size=legend_size)

    # Title and Label text
    letter_weights = np.ones((len(string.ascii_letters) + 1))
    # increase the weight of white space character
    letter_weights[-1] = int(len(letter_weights) * 0.2)
    letter_weights = list(itertools.accumulate(letter_weights))
    letters = string.ascii_letters + " "
    title_length = random.choice(range(title_length_min, title_length_max))
    title_text = "".join(random.choices(letters, cum_weights=letter_weights, k=title_length)).strip()
    xlabel_length = random.choice(range(axis_label_length_min, axis_label_length_max))
    xlabel = "".join(random.choices(letters, cum_weights=letter_weights, k=xlabel_length)).strip()
    ylabel_length = random.choice(range(axis_label_length_min, axis_label_length_max))
    ylabel = "".join(random.choices(letters, cum_weights=letter_weights, k=ylabel_length)).strip()

    ticks_label = []
    for i in range(bar_num):
        ticks_label_length = random.choice(range(ticks_label_length_min, ticks_label_length_max))
        ticks_label.append("".join(random.choices(string.ascii_letters, k=ticks_label_length)).strip())

    legend_char = []
    for i in range(bar_per_loc):
        legend_length = random.choice(range(legend_length_min, legend_length_max))
        legend_char.append("".join(random.choices(letters, k=legend_length)).strip())

    # decide whether the switch of axis label, title and legend
    axis_label_switch = random.choice(["on", "off"])
    title_switch = random.choice(["on", "off"])
    legend_switch = random.choice(["on", "off"])
    legend_pos = random.choice(legend_position)

    if axis_label_switch == "on":
        xlabel = ax.set_xlabel(xlabel, fontproperties=axis_label_font)
        ylabel = ax.set_ylabel(ylabel, fontproperties=axis_label_font)
        axis_label = (xlabel, ylabel)

    # set the ticks and tick labels
    set_hticks(x + (bar_width / 2) * bar_per_loc)
    hticklabels = set_hticklabels(ticks_label, fontproperties=ticks_label_font)
    vticklabels = get_vticklabels()
    for label in vticklabels:
        label.set_fontproperties(ticks_label_font)
    axis_ticks = (hticklabels, vticklabels)

    # set legend, possible positions include: top, bottom, upper right and center right
    ax_bbox = ax.get_position()
    tight_rect = [0, 0, 1, 1]
    if legend_switch == "on":
        if legend_pos == "top":
            ax.set_position([ax_bbox.x0, ax_bbox.y0, ax_bbox.width, ax_bbox.height * 0.85])
            legend = ax.legend(legend_char, prop=legend_font, ncol=bar_per_loc, loc="lower center",
                               bbox_to_anchor=(0.5, 1))
            tight_rect = [0, 0, 1, 0.85]
        if legend_pos == "bottom":
            ax.set_position([ax_bbox.x0, ax_bbox.y0 + ax_bbox.height * 0.15, ax_bbox.width, ax_bbox.height * 0.85])
            tight_rect = [0, 0.15, 1, 1]
            if axis_label_switch == "on":
                legend = ax.legend(legend_char, prop=legend_font, ncol=bar_per_loc, loc="upper center",
                                   bbox_to_anchor=(0.5, -0.12))
            else:
                legend = ax.legend(legend_char, prop=legend_font, ncol=bar_per_loc, loc="upper center",
                                   bbox_to_anchor=(0.5, -0.05))
        if legend_pos == "right":
            ax.set_position([ax_bbox.x0, ax_bbox.y0, ax_bbox.width * 0.85, ax_bbox.height])
            tight_rect = [0, 0, 0.85, 1]
            if random.choice(["top", "center"]) == "center":
                legend = ax.legend(legend_char, prop=legend_font, ncol=1, loc="center left", bbox_to_anchor=(1, 0.5))
            else:
                legend = ax.legend(legend_char, prop=legend_font, ncol=1, loc="upper left", bbox_to_anchor=(1, 1))

    if title_switch == "on":
        title_loc = random.choice(title_location)
        if legend_pos == "top":
            title = ax.set_title(title_text, fontproperties=title_font, loc="center", y=1.1)
        else:
            title = ax.set_title(title_text, fontproperties=title_font, loc=title_loc, y=1.01)

    plt.tight_layout(rect=tight_rect)
    fig.savefig(filename, dpi="figure")
    return fig, ax, bars, data, title, legend, axis_ticks, axis_label, bar_direction


# In[4]:


def get_bar_pixel(fig_height, bars, data):
    """
    method that return the bounding box of the bars that are arranged
    according to their x axis positions.
    Inputs:
    bars: objects of the bars of the plot, of size(bar_per_loc, bar_nums)
    data: list(x, y) containing the x coordinates of the bars and the heights of the bars
    Outputs:
    bar_coord: a list of dict containing bbox and height(data coordinate) of bars
    dict looks like {bbox:[top-left-x, top-left-y, bottom-right-x, bottom-right-y], height:h}
    """
    bar_heights = data[1]
    bar_per_loc = len(bars)
    bar_nums = len(bars[0])
    bar_coord = []

    for i in range(bar_nums):
        for j in range(bar_per_loc):
            h = bar_heights[j][i]
            bar = bars[j][i]
            b_cor = get_bbox_coord(fig_height, bar.get_window_extent())
            bar_coord.append({"bbox": b_cor, "height": h})

    return bar_coord


def get_tick_pixel(fig, fig_height, axis_ticks):
    """
    method that return tick coordinates and tick texts on the xy axes
    Inputs:
    fig: matplotlib object of the figure of the plot
    axis_ticks: tick object
    """
    ticklabel = []
    for i in range(2):
        ticktmp = []
        for t in axis_ticks[i]:
            text = t.get_text()
            # unclear error: raise "cannot get window extent" sometimes unless passed renderer kwarg
            b_cor = get_bbox_coord(fig_height, t.get_window_extent(renderer=fig.canvas.get_renderer()))
            ticktmp.append({"bbox": b_cor, "text": text})
        ticklabel.append(ticktmp)

    return ticklabel


def get_label_pixel(fig_height, axis_label):
    """
    """
    if axis_label is None:
        return None
    al = []
    for label in axis_label:
        text = label.get_text()
        b_cor = get_bbox_coord(fig_height, label.get_window_extent())
        al.append({"bbox": b_cor, "text": text})

    return al


def get_title_pixel(fig_height, title):
    """
    """
    if title is None:
        return None
    text = title.get_text()
    b_cor = get_bbox_coord(fig_height, title.get_window_extent())

    return {"bbox": b_cor, "text": text}


def get_legend_pixel(fig_height, legend):
    """
    """
    if legend is None:
        return None
    legend_list = []
    text_list = legend.get_texts()
    for t in text_list:
        text = t.get_text()
        b_cor = get_bbox_coord(fig_height, t.get_window_extent())
        legend_list.append({"bbox": b_cor, "text": text})

    return legend_list


def get_bbox_coord(fig_height, bbox):
    """
    take bounding boxs(bottom-left, top-right) and return processed boxs(top-left, bottom-right)
    the orgin is at top-left of the plot
    """
    cor = bbox.get_points()
    tmp = cor[0][1]
    cor[0][1] = fig_height - cor[1][1]
    cor[1][1] = fig_height - tmp
    b_cor = [round(cor[0][0]), round(cor[0][1]), round(cor[1][0]), round(cor[1][1])]
    return [int(i) for i in b_cor]


def write_coord(file_obj, plot_name, coord, sep):
    string_prep = "{plot_name}{seperator}".format(plot_name=plot_name, seperator=sep)
    string_prep += "{coord}".format(coord=coord)
    file_obj.write(string_prep)
    file_obj.write("\n")


def get_all_bbox(plot_objs, fig_height):
    fig, ax, bars, data, title, legend, axis_ticks, axis_label, bar_direction = plot_objs

    barbbox = get_bar_pixel(fig_height, bars, data)
    tickbbox = get_tick_pixel(fig, fig_height, axis_ticks)
    axislabelbbox = get_label_pixel(fig_height, axis_label)
    titlebbox = get_title_pixel(fig_height, title)
    legendbbox = get_legend_pixel(fig_height, legend)

    return barbbox, tickbbox, axislabelbbox, titlebbox, legendbbox


def generate_plots(n, train_or_test):
    # the python script directory
    # dir_path = os.path.dirname(os.path.realpath(__file__))
    dir_path = os.getcwd()
    outter_dir = os.path.join(dir_path, "data", train_or_test)
    plot_dir = os.path.join(outter_dir, "plots")
    if not os.path.exists(outter_dir):
        os.makedirs(outter_dir)
        os.mkdir(plot_dir)

    with open(os.path.join(outter_dir, train_or_test + "_barbbox.idl"), "w") as f_bar, open(
            os.path.join(outter_dir, train_or_test + "_tickbbox.idl"), "w") as f_tick, open(
            os.path.join(outter_dir, train_or_test + "_axislabelbbox.idl"), "w") as f_label, open(
            os.path.join(outter_dir, train_or_test + "_titlebbox.idl"), "w") as f_title, open(
            os.path.join(outter_dir, train_or_test + "_legendbbox.idl"), "w") as f_legend, open(
            os.path.join(outter_dir, train_or_test + "_imgsize.idl"), "w") as f_imgsize:

        file_objs = [f_bar, f_tick, f_label, f_title, f_legend, f_imgsize]
        # seperator in idl file
        sep = " -<>- "
        for i in tqdm(range(n)):
            try:
                img_type = random.choice(["jpg", "png"])
                img_name = "{}_{}.{}".format(train_or_test, i, img_type)
                plot_name = os.path.join(plot_dir, img_name)
                plot_objs = get_random_plot(plot_name)
                imgsize = list(map(int, plot_objs[0].get_size_inches() * plot_objs[0].dpi))
                fig_height = imgsize[1]

                bboxs_all = get_all_bbox(plot_objs, fig_height)
                for j, b in enumerate(bboxs_all):
                    write_coord(file_objs[j], img_name, b, sep)
                # write figure size
                write_coord(file_objs[-1], img_name, [imgsize, plot_objs[-1]], sep)
                # close the figure
                plt.close(plot_objs[0])
            except Exception:
                error_file = os.path.join(outter_dir, "error_log.txt")
                if not os.path.exists(error_file):
                    os.makedirs(error_file)
                with open(error_file, "a") as f_error:
                    f_error.write("{} error".format(plot_name))
                    f_error.write("\n")

        print(train_or_test + " plot generation done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="This python script generates random bar charts and their elements' bounding boxes")
    parser.add_argument("--n_train", help="Number of traning images", required=True, type=int)
    parser.add_argument("--n_test", help="Number of test images", required=True, type=int)
    args = vars(parser.parse_args())

    print("generating {} training data:".format(args["n_train"]))
    generate_plots(args["n_train"], "train")
    print("generating {} test data".format(args["n_test"]))
    generate_plots(args["n_test"], "test")
