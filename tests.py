#-*- coding:utf-8 -*-

from __future__ import absolute_import
from __future__ import unicode_literals

from nose.tools import istest as test

from jbo import Entry
import jbo


@test
def dbfilter():
    assert 'donri' in jbo.dbfilter(['daytime'])
    assert len(list(jbo.dbfilter('is at the on'.split()))) == 14
    assert list(jbo.dbfilter(['hello'])) == ['rinsa', 'coi']
