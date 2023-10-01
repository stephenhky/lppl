# -*- coding: utf-8 -*-
"""
Created on Tue Dec 24 16:47:38 2013

@author: hok1
"""

from lpplfit import LPPLGeneticAlgorithm, PoolLPPLGeneticAlgorithm
from pele_lpplfit import LPPLPeleAlgorithm
import numpy as np
import argparse
from datetime import datetime as dt
import time

def DiscreteDaytoYearFraction(year, month, day):
    wholeyear = dt(year+1, 1, 1).toordinal() - dt(year, 1, 1).toordinal()
    fraction = dt(year, month, day).toordinal() - dt(year, 1, 1).toordinal()
    return year + float(fraction)/float(wholeyear)

def toContinuousYear(datestr, delimiter='/'):
    month, day, year = map(int, datestr.split(delimiter))
    if year < 100:
        year += 1900 if year>=70 else 2000
    return DiscreteDaytoYearFraction(year, month, day)

def readRawYearFractionData(filename, header=False):
    data = np.loadtxt(filename, unpack=True, skiprows=1 if header else 0,
                      usecols=(0, 1), dtype={'names': ('year', 'price'),
                                             'formats': (np.float, np.float)})
    (tarray, yarray) = data
    return tarray, yarray
    
def readStrYearFractionData(filename, header=True):
    data = np.loadtxt(filename, unpack=True, skiprows=1 if header else 0,
                      usecols=(0, 1), dtype={'names': ('year', 'price'),
                                             'formats': ('S20', np.float)},
                      delimiter=',')
    yarray = data[1]
    tarray = np.array(map(toContinuousYear, data[0]))
    return tarray, yarray
    
def readData(filename, decimal_year=True, lowlimit=None, uplimit=None):
    data = readRawYearFractionData(filename) if decimal_year else readStrYearFractionData(filename)
    if lowlimit==None:
        lowlimit = min(data[0])
    if uplimit==None:
        uplimit = max(data[0])
    data = zip(data[0], data[1])
    data = filter(lambda datum: datum[0]>=lowlimit and datum[0]<=uplimit,
                  data)
    data = sorted(data, key=lambda datum: datum[0])
    tarray = np.array(map(lambda datum: datum[0], data))
    yarray = np.array(map(lambda datum: datum[1], data))
    return tarray, yarray

def lpplfit_workflow(tarray, parray, param_pop_size, max_iter, mutprob, 
                     reproduceprob, numthreads=1):
    yarray = np.log(parray)
    fitalg = LPPLGeneticAlgorithm() if numthreads==1 else PoolLPPLGeneticAlgorithm(numthreads=numthreads)
    param_pop = fitalg.generate_init_population(tarray, yarray,
                                                size=param_pop_size)
    param_pop, costs_iter = fitalg.perform(tarray, yarray, 
                                           init_param_pop=param_pop,
                                           size=param_pop_size,
                                           max_iter=max_iter, mutprob=mutprob, 
                                           reproduceprob=reproduceprob)
    res_param = fitalg.grad_optimize(tarray, yarray, param_pop[0])
    return res_param

def pele_lpplfit_workflow(tarray, parray):
    yarray = np.log(parray)
    fitalg = LPPLPeleAlgorithm()
    res_param = fitalg.rolling_peaks_price_gyrations(tarray, yarray)
    return res_param

def get_argvparser():
    prog_descp = 'Read the financial data from the input file, '
    prog_descp += 'the first column being the year and second column prices, '
    prog_descp += 'and then fit the data to estimate the time of the bubble crash.'
    argv_parser = argparse.ArgumentParser(description=prog_descp)
    argv_parser.add_argument('filename',
                             help='the name of the file for fitting')
    argv_parser.add_argument('--stringdate',
                             help='date given with month and date instead of decimal years',
                             action='store_true')
    argv_parser.add_argument('--parampopsize', type=int, default=500,
                             help='number of sets of parameters (default=500)')
    argv_parser.add_argument('--maxiter', type=int, default=150,
                             help='maximum number of iterations of genetic algorithm (default=150)')
    argv_parser.add_argument('--mutprob', type=float, default=0.75,
                             help='mutation probability (between 0 and 1, default=0.75)')
    argv_parser.add_argument('--reproduceprob', type=float, default=0.25,
                             help='breeding probability (between 0 and 1, default=0.25)')
    argv_parser.add_argument('--lowlimit', type=str, default=None,
                             help='lower limit (default=None)')
    argv_parser.add_argument('--uplimit', type=str, default=None,
                             help='upper limit (default=None)')
    argv_parser.add_argument('--numthreads', type=int, default=1,
                             help='number of threads (default=1)')
    argv_parser.add_argument('--pelealg',
                             help='using Pele''s algorithm',
                             action='store_true')
    return argv_parser
    
if __name__ == '__main__':
    argv_parser = get_argvparser()
    args = argv_parser.parse_args()    
    
    if args.stringdate:
        lowlimit = None if args.lowlimit==None else toContinuousYear(args.lowlimit) 
        uplimit = None if args.uplimit==None else toContinuousYear(args.uplimit)
    else:
        lowlimit = None if args.lowlimit==None else float(args.lowlimit)
        uplimit = None if args.uplimit==None else float(args.uplimit)

    tarray, parray = readData(args.filename, 
                              decimal_year=(not args.stringdate),
                              lowlimit=lowlimit, uplimit=uplimit)
    
    exe_datetime = time.strftime('%x')+' '+time.strftime('%X')

    if args.pelealg:
        res_param = pele_lpplfit_workflow(tarray, parray)
    else:
        res_param = lpplfit_workflow(tarray, parray, args.parampopsize,
                                     args.maxiter, args.mutprob,
                                     args.reproduceprob,
                                     numthreads=args.numthreads)
    
    line_to_print = '\t'.join([exe_datetime, str(lowlimit), str(uplimit),
                               str(args.maxiter)])
    line_to_print += '\t'+'\t'.join(map(lambda key: str(res_param[key]), 
                                        ['A', 'B', 'C', 'phi', 'omega', 'z', 'tc']))
    print(line_to_print)
