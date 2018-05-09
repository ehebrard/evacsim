#! /usr/bin/env python

from rocknrun import *


launchname = 'cp2018big'


def setup():
    
    cmdline = {}
    
    
    base = '/home/ehebrard/work/dev/git/energeticcumulative/code/EvacPlan1 --limit 3600'
    
    options = []
    
    propagator = {}
    propagator['no'] = '--propagator 0'
    # propagator['energ'] = '--propagator 2'
    propagator['flow'] = '--propagator 3 --frequency 0 --pruneratio 0.95 --minspace 50'
    # propagator['flow+'] = '--propagator 3 --frequency 0 --pruneratio 0.95 --minspace 50 --ERcuts 1'
    propagator['simp'] = '--propagator 3 --frequency 0 --pruneratio 1 --minspace 1000'

    
    options.append(propagator)
    
    
    heur = {}
    heur['no'] = '--heur 0'
    heur['root'] = '--heur 1'
    heur['node'] = '--heur 2'
    
    options.append(heur)
    
    
    search = {}
    search['dfs'] = '--search 0'
    # search['res'] = '--search 1'
    search['auto'] = '--search 2'
    
    options.append(search)
    
    
    cmds = [base]
    algos = ['cpo']
    
    for option in options:
        algos = [(algo+'_'+str(key)).strip('_') for algo in algos for key in option.keys()]
        cmds = [cmd+' '+option[key] for cmd in cmds for key in option.keys()]
        
    # # print algorithms
    # # print cmds
    # algorithms = ['cpo_gflow_0.5_100', 'cpo_gflow_0.5_10', 'cpo_gflow_0.5_20', 'cpo_gflow_0.5_50', 'cpo_gflow_0.5_300', 'cpo_gflow_1_100', 'cpo_gflow_1_10', 'cpo_gflow_1_20', 'cpo_gflow_1_50', 'cpo_gflow_1_300', 'cpo_gflow_0.8_100', 'cpo_gflow_0.8_10', 'cpo_gflow_0.8_20', 'cpo_gflow_0.8_50', 'cpo_gflow_0.8_300', 'cpo_gflow_0.99_100', 'cpo_gflow_0.99_10', 'cpo_gflow_0.99_20', 'cpo_gflow_0.99_50', 'cpo_gflow_0.99_300', 'cpo_gflow_0.9_100', 'cpo_gflow_0.9_10', 'cpo_gflow_0.9_20', 'cpo_gflow_0.9_50', 'cpo_gflow_0.9_300', 'cpo_gflow_0.95_100', 'cpo_gflow_0.95_10', 'cpo_gflow_0.95_20', 'cpo_gflow_0.95_50', 'cpo_gflow_0.95_300']
    #
    # algorithms = [a for a in algos if a.find('gflow_1') < 0]
    # algorithms.append('cpo_gflow_1_100')
    #
    # cmdlines = [cmds[a] for a in algorithms]
    #
    # # sys.exit()
        
   
    # method = dict(zip(algorithms, cmdlines))
    method = dict([(a,c) for a,c in zip(algos, cmds) if a.find('gflow_1') < 0 or a.find('cpo_gflow_1_100') >= 0])
    
    print method
    
    
    algorithms = sorted(method.keys())
    
    keyfile = open('%s.key'%launchname, 'w')
    keyfile.write('%i methods\n'%len(algorithms))
    
    for algo in algorithms:
         keyfile.write(algo+'\n')
         keyfile.write(method[algo]+' #BENCHMARK\n')

    # declare the benchmarks (print_benchlist assumes that everything in benchfolder is an instance file)
    benchfolders = ['data']
    print_benchlist(benchfolders, keyfile)

    keyfile.close()


                
if __name__ == '__main__' :
    setup()
    e = Experiment(keys=[launchname])
    e.generate_jobs(timeout='00:45:00',keys=[launchname],full=False) 
    
