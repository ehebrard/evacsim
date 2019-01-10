#!/usr/bin/env python

from distutils.core import setup

setup(name='EvacSim',
      version='1.0',
      description='Python instance generator for evacuation planning in case of wildfire',
      author='Emmanuel Hebrard',
      author_email='hebrard@laas.fr',
      #url='https://www.python.org/sigs/distutils-sig/',
      py_modules=['generator'],
      packages=['quadtree'],
     )