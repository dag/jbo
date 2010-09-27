#!/usr/bin/env python
#-*- coding:utf-8 -*-

from __future__ import absolute_import
from __future__ import unicode_literals

from nose.tools import istest as test
from nose import main

from jbo import Entry
import jbo


@test
def dbfilter():
    assert 'donri' in jbo.dbfilter(['daytime'])
    assert len(list(jbo.dbfilter('is at the on'.split()))) == 14
    assert list(jbo.dbfilter(['hello'])) == ['rinsa', 'coi']


@test
def entries():
    with jbo.dbopen('entries') as db:
        assert db[b'donri'].affixes == ['dor', "do'i"]


if __name__ == '__main__':
    main()
