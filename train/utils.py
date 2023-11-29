import sys
import csv


def append_history(history_file, metrics, first=False):
    """
    Args:
        history_file: 'path/to/history_xx.csv'
        metrics: dict: {}
        first: bool
    """
    columns = sorted(metrics.keys())
    with open(history_file, 'a') as file:
        writer = csv.writer(file, delimiter=',', quotechar="'", quoting=csv.QUOTE_MINIMAL)
        if first:
            writer.writerow(columns)
        writer.writerow(list(map(lambda x: metrics[x], columns)))


class Logger(object):
    def __init__(self, filename, stream=sys.stdout):
        self.terminal = stream
        self.log = open(filename, 'a')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass




