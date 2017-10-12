import os
from itertools import cycle
import torch
import logging.config
import shutil
import pandas as pd
from bokeh.io import output_file, save, show
from bokeh.plotting import figure
from bokeh.plotting.helpers import DEFAULT_PALETTE
from bokeh.layouts import column
from copy import deepcopy


def setup_logging(log_file='log.txt'):
    """Setup logging configuration
    """
    logging.basicConfig(level=logging.DEBUG,
                        format="%(asctime)s - %(levelname)s - %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S",
                        filename=log_file,
                        filemode='w')
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)


class ResultsLog(object):

    def __init__(self, path='results.csv', plot_path=None):
        self.path = path
        self.plot_path = plot_path or (self.path + '.html')
        self.results = None
        self.clear()

    def clear(self):
        self.figures = []

    def add(self, **kwargs):
        df = pd.DataFrame([kwargs.values()], columns=kwargs.keys())
        if self.results is None:
            self.results = df
        else:
            self.results = self.results.append(df, ignore_index=True)

    def save(self, title='Training Results'):
        if len(self.figures) > 0:
            if os.path.isfile(self.plot_path):
                os.remove(self.plot_path)
            output_file(self.plot_path, title=title)
            plot = column(*self.figures)
            save(plot)
            self.clear()
        self.results.to_csv(self.path, index=False, index_label=False)

    def load(self, path=None):
        path = path or self.path
        if os.path.isfile(path):
            self.results.read_csv(path)

    def show(self):
        if len(self.figures) > 0:
            plot = column(*self.figures)
            show(plot)

    def plot(self, x, y, title=None, xlabel=None, ylabel=None,
             width=800, height=400, colors=None, line_width=2,
             tools='pan,box_zoom,wheel_zoom,box_select,hover,reset,save'):
        xlabel = xlabel or x
        f = figure(title=title, tools=tools,
                   width=width, height=height,
                   x_axis_label=xlabel or x,
                   y_axis_label=ylabel or '')
        if colors is not None:
            colors = iter(colors)
        else:
            colors = cycle(DEFAULT_PALETTE)
        for yi in y:
            f.line(self.results[x], self.results[yi],
                   line_width=line_width,
                   line_color=next(colors), legend=yi)
        self.figures.append(f)

    def image(self, *kargs, **kwargs):
        fig = figure()
        fig.image(*kargs, **kwargs)
        self.figures.append(fig)


def save_checkpoint(state, is_best, path='.', filename='checkpoint.pth.tar', save_all=False):
    filename = os.path.join(path, filename)
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, os.path.join(path, 'model_best.pth.tar'))
    if save_all:
        shutil.copyfile(filename, os.path.join(
            path, 'checkpoint_epoch_%s.pth.tar' % state['epoch']))
