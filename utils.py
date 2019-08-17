"""A range of utility functions."""

from enum import IntEnum
import os
import sys

def mkdir(directory):
    """Create directory if not exist."""
    if not os.path.exists(directory):
        try:
            os.makedirs(directory)
        except OSError:
            print("Direcotory %s already exists." %directory)

def savecmd(directory, cmd):
    """Cache commands."""
    cmddir = os.path.join(directory, 'CMDs')
    mkdir(cmddir)
    filename = os.path.splitext(cmd[0].split('/')[-1])[0] + '.cmds'
    cmdsep = '------------------------------------\n'
    with open(os.path.join(cmddir, filename), 'a+') as file:
        file.write(cmdsep)
        file.write(' '.join(cmd) + '\n')
        file.write(cmdsep)

def check_file(*args):
    """Check if a file exists."""
    for path in args:
        if not os.path.isfile(path):
            print("ERROR: File does not exist: %s" %path)
            sys.exit(1)

def check_dir(*args):
    """Check if a directory exists."""
    for path in args:
        if not os.path.isdir(path):
            print("ERROR: Directory does not exist: %s" %path)
            sys.exit(1)

def print_options(args):
    """Display all arguments, args is a object from argparse."""
    output_list = []
    print_color_msg('==> All options are displayed below:')
    for arg in vars(args):
        line = "".ljust(4) + "--{0:20}{1}".format(arg, getattr(args, arg))
        output_list.append(line)
        print(line)

    args_file_path = os.path.join(args.resume, 'opts.txt')
    remove_file(args_file_path)
    with open(args_file_path, 'a') as args_file:
            args_file.write('\n'.join(output_list))

def color_msg(msg):
    """Return colored message."""
    return "\033[38;5;108m%s\033[0m" %(msg)

def print_color_msg(msg):
    """Print colored message."""
    print("\033[38;5;108m%s\033[0m" %(msg))

def remove_file(file_path):
    try:
        os.remove(file_path)
    except OSError:
        pass
