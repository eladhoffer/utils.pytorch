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
from bokeh.layouts import column
from bokeh.models import Div


def setup_logging_and_results(args):
    """
    Calls setup_loggining, exports args and creates a ResultsLog class.
    Can resume training/logging if args.resume is set
    """
    def set_args_default(field_name, value):
        if hasattr(args, field_name):
            return eval('args.' + field_name)
        else:
            return value

    # Set default args in case they don't exist in args
    resume = set_args_default('resume', False)
    data_format = set_args_default('data_format', 'csv')
    save_name = set_args_default('save_name', '')
    results_dir = set_args_default('results_dir', './results')

    if save_name is '':
        save_name = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    save_path = os.path.join(results_dir, save_name)
    if os.path.exists(save_path):
        shutil.rmtree(save_path)
    os.makedirs(save_path, exist_ok=True)
    log_file = os.path.join(save_path, 'log.txt')

    setup_logging(log_file, resume)
    results = ResultsLog(path=save_path, title=save_name,
                         resume=False, data_format=data_format)
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

    supported_data_formats = ['csv', 'json']

    def __init__(self, path='', title='', resume=False, data_format='csv'):
        """
        Parameters
        ----------
        path: string
            path to directory to save data files
        plot_path: string
            path to directory to save plot files
        title: string
            title of HTML file
        resume: bool
            resume previous logging
        data_format: str('csv'|'json')
            which file format to use to save the data
        """
        if data_format not in ResultsLog.supported_data_formats:
            raise ValueError('data_format must of the following: ' +
                             '|'.join(['{}'.format(k) for k in ResultsLog.supported_data_formats]))

        if data_format == 'json':
            self.data_path = '{}.json'.format(path)
        else:
            self.data_path = '{}.csv'.format(path)
        self.plot_path = '{}.html'.format(path)
        self.results = None
        self.clear()
        self.first_save = True
        if os.path.isfile(self.data_path):
            if resume:
                self.load(self.data_path)
                self.first_save = False
            else:
                os.remove(self.data_path)
                self.results = pd.DataFrame()
        else:
            self.results = pd.DataFrame()

        self.title = title
        self.data_format = data_format

    def clear(self):
        self.figures = []

    def add(self, **kwargs):
        """Add a new row to the dataframe
        example:
            resultsLog.add(epoch=epoch_num, train_loss=loss,
                           test_loss=test_loss)
        """
        df = pd.DataFrame([kwargs.values()], columns=kwargs.keys())
        self.results = self.results.append(df, ignore_index=True)

    def smooth(self, column_name, window):
        """Select an entry to smooth over time"""
        # TODO: smooth only new data
        smoothed_column = self.results[column_name].rolling(
            window=window, center=False).mean()
        self.results[column_name + '_smoothed'] = smoothed_column

    def save(self, title=None):
        """save the json file.
        Parameters
        ----------
        title: string
            title of the HTML file
        """
        title = title or self.title
        if len(self.figures) > 0:
            if os.path.isfile(self.plot_path):
                os.remove(self.plot_path)
            if self.first_save:
                self.first_save = False
                logging.info('Plot file saved at: {}'.format(
                    os.path.abspath(self.plot_path)))

            output_file(self.plot_path, title=title)
            plot = column(
                Div(text='<h1 align="center">{}</h1>'.format(title)), *self.figures)
            save(plot)
            self.clear()

        if self.data_format == 'json':
            self.results.to_json(self.data_path, orient='records', lines=True)
        else:
            self.results.to_csv(self.data_path, index=False, index_label=False)

    def load(self, path=None):
        """load the data file
        Parameters
        ----------
        path:
            path to load the json|csv file from
        """
        path = path or self.data_path
        if os.path.isfile(path):
            if self.data_format == 'json':
                self.results.read_json(path)
            else:
                self.results.read_csv(path)
        else:
            raise ValueError('{} isn''t a file'.format(path))

    def show(self, title=None):
        title = title or self.title
        if len(self.figures) > 0:
            plot = column(
                Div(text='<h1 align="center">{}</h1>'.format(title)), *self.figures)
            show(plot)

    def plot(self, x, y, title=None, xlabel=None, ylabel=None, legend=None,
             width=800, height=400, line_width=2,
             colors=['red', 'green', 'blue', 'orange',
                     'black', 'purple', 'brown'],
             tools='pan,box_zoom,wheel_zoom,box_select,hover,reset,save'):
        """
        add a new plot to the HTML file
        example:
            results.plot(x='epoch', y=['train_loss', 'val_loss'],
                         'title='Loss', 'ylabel'='loss')
        """
        if not isinstance(y, list):
            y = [y]
        xlabel = xlabel or x
        legend = legend or y
        assert len(legend) == len(y)
        f = figure(title=title, tools=tools,
                   width=width, height=height,
                   x_axis_label=xlabel or x,
                   y_axis_label=ylabel or '')
        colors = cycle(colors)
        for i, yi in enumerate(y):
            f.line(self.results[x], self.results[yi],
                   line_width=line_width,
                   line_color=next(colors), legend=legend[i])
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

