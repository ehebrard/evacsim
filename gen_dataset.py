#!/usr/bin/env python

import subprocess


for graph in ['sparse', 'medium', 'dense']:
    for n in [10,15,20,25]:
        for speed in [30]:
            for pace in [3]:
                for seed in range(1,21):
                    cmd = ['./generator.py', graph, '--evacuation', '--num_evacuations', str(n), '--speed', str(speed), '--firepace', str(pace), '--seed', str(seed), '--tofile']
                    print ' '.join(cmd)
                    g = subprocess.Popen(cmd)
                    g.wait()