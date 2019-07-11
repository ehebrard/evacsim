#!/usr/bin/env python

import subprocess


for graph in ['small']:
    for n in [3]:
        for speed in [30]:
            for pace in [5]:
                for seed in range(1,20):
                    cmd = ['./generator.py', graph, '--evacuation', '--num_evacuations', str(n), '--speed', str(speed), '--firepace', str(pace), '--seed', str(seed), '--tofile', '--writetree']
                    print ' '.join(cmd)
                    g = subprocess.Popen(cmd)
                    g.wait()