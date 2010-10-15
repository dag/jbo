#!/usr/bin/env python
#-*- coding:utf-8 -*-

from __future__ import absolute_import
from __future__ import unicode_literals

from setuptools import setup

import jbo


setup(name='jbo',
      version=jbo.VERSION,
      description=jbo.main.__doc__,
      py_modules=['jbo'],
      scripts=['bin/jbo'],
      install_requires=['progressbar>2.2',
                        'PyStemmer'],
      extras_require={'lxml': ['lxml']},
      author='Dag Odenhall',
      author_email='dag.odenhall@gmail.com',
      license='Simplified BSD',
      url='http://github.com/dag/jbo',
      classifiers=['Development Status :: 4 - Beta',
                   'Environment :: Console',
                   'Intended Audience :: Developers',
                   'Intended Audience :: End Users/Desktop',
                   'Intended Audience :: Science/Research',
                   'License :: OSI Approved :: BSD License',
                   'Operating System :: OS Independent',
                   'Programming Language :: Python',
                   'Programming Language :: Python :: 2.6',
                   'Programming Language :: Python :: 2.7',
                   'Topic :: Text Processing :: Indexing',
                   'Topic :: Text Processing :: Linguistic',])
