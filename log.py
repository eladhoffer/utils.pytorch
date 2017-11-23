import shutil
import os
from itertools import cycle
import torch
import logging.config
from datetime import datetime
import json

import pandas as pd
from bokeh.io import output_file, save, show
from bokeh.plotting import figure
import bokeh.palettes as pal
from bokeh.layouts import column
from bokeh.models import Div


def setup_logging_and_results(args):
    """
    Calls setup_loggining, exports args and creates a ResultsLog class.
    Assumes args contains save_path, save_name
    Can resume training/logging if args.resume is set
    """
    if args.save_name is '':
        args.save_name = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    save_path = os.path.join(args.results_dir, args.save_name)
    if os.path.exists(save_path):
        shutil.rmtree(save_path)
    os.makedirs(save_path, exist_ok=True)
    log_file = os.path.join(save_path, 'log.txt')

    if hasattr(args, 'resume'):
        resume = args.resume
    else:
        resume = False
    setup_logging(log_file, resume)
    results = ResultsLog(path=save_path, plot_title=args.save_name, resume=False)
    export_args(args, save_path)
    return results, save_path


def export_args(args, save_path):
    """
    args: argparse.Namespace
        arguments to save
    save_path: string
        path to directory to save at
    """
    os.makedirs(save_path, exist_ok=True)
    json_file_name = os.path.join(save_path, 'args.json')
    with open(json_file_name, 'w') as fp:
        json.dump(dict(args._get_kwargs()), fp, sort_keys=True, indent=4)


def setup_logging(log_file='log.txt', resume=False):
    """
    Setup logging configuration
    """
    if os.path.isfile(log_file) and resume:
        file_mode = 'a'
    else:
        file_mode = 'w'

    root_logger = logging.getLogger()
    if root_logger.handlers:
        root_logger.removeHandler(root_logger.handlers[0])
    logging.basicConfig(level=logging.DEBUG,
                        format="%(asctime)s - %(levelname)s - %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S",
                        filename=log_file,
                        filemode=file_mode)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)


class ResultsLog(object):

    def __init__(self, path='', plot_title='', resume=False):
        """
        Parameters
        ----------
        path: string
            path to directory to save json files
        plot_path: string
            path to directory to save plot files
        plot_title: string
            title of HTML file
        resume: bool
            resume previous logging
        """
        os.makedirs(path, exist_ok=True)
        full_path = os.path.join(path, 'results')
        self.date_path = '{}.json'.format(full_path)
        self.plot_path = '{}.html'.format(full_path)
        self.results = None
        self.clear()

        self.first_save = True
        if os.path.isfile(self.date_path):
            if resume:
                self.load(self.date_path)
                self.first_save = False
            else:
                os.remove(self.date_path)
                self.results = pd.DataFrame()
        else:
            self.results = pd.DataFrame()

        self.plot_title = plot_title

    def clear(self):
        self.figures = []

    def add(self, **kwargs):
        """Add a new row to the dataframe
        example:
            resultsLog.add(epoch=epoch_num, train_loss=loss, test_loss=test_loss)
        """
        df = pd.DataFrame([kwargs.values()], columns=kwargs.keys())
        self.results = self.results.append(df, ignore_index=True)

    def smooth(self, column_name, window):
        """Select an entry to smooth over time"""
        # TODO: smooth only new data
        smoothed_column = self.results[column_name].rolling(window=window, center=False).mean()
        self.results[column_name+'_smoothed'] = smoothed_column

    def save(self, title='Training Results'):
        """save the json file.
        Parameters
        ----------
        title: string
            title of the HTML file
        """
        if len(self.figures) > 0:
            if os.path.isfile(self.plot_path):
                os.remove(self.plot_path)
            if self.first_save:
                self.first_save = False
                logging.info('Plot file saved at: {}'.format(os.path.abspath(self.plot_path)))

            output_file(self.plot_path, title=title)
            plot = column(Div(text='<h1 align="center">{}</h1>'.format(self.plot_title)), *self.figures)
            save(plot)
            self.clear()

        self.results.to_json(self.date_path, orient='records', lines=True)

    def load(self, path=None):
        """load a json file
        Parameters
        ----------
        path:
            path to load the json file from
        """
        path = path or self.path
        if os.path.isfile(path):
            self.results = pd.read_json(path)
        else:
            raise ValueError('{} isn''t a file'.format(path))

    def show(self):
        if len(self.figures) > 0:
            plot = column(Div(text='<h1 align="center">{}</h1>'.format(self.plot_title)), *self.figures)
            show(plot)

    def plot(self, x, y, title=None, xlabel=None, ylabel=None,
             width=800, height=400, colors=None, line_width=2,
             tools='pan,box_zoom,wheel_zoom,box_select,hover,reset,save'):
        """
        add a new plot to the HTML file
        example:
            results.plot(x='epoch', y=['train_loss', 'val_loss'],
                         title='Loss', ylabel='loss')
        """
        if not isinstance(y, list):
            y = [y]
        xlabel = xlabel or x
        f = figure(title=title, tools=tools,
                   width=width, height=height,
                   x_axis_label=xlabel or x,
                   y_axis_label=ylabel or '')
        if colors is not None:
            colors = iter(colors)
        else:
            colors = cycle(list(pal.Colorblind[8]))
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
