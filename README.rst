Use jbovlaste on the command line, offline.
===========================================

* Displays L\ :sup:`A`\T\ :sub:`E`\X formatting
  as unicode plain text with console formatting
* Uses a hashing key-value store for fast look ups
* Computes search terms in advance with scoring for fast serching
* Stemming if `PyStemmer <http://pypi.python.org/pypi/PyStemmer/>`_
  is installed
* Handles undefined compound words beautifully: *lobybau* resolves to
  *jbobau* and *cizbau* to *cizra* and *bangu*
* Script friendly: define your own shell aliases and functions


jbo define
----------

.. image:: http://github.com/dag/jbo/raw/master/screenshots/define.png


jbo filter
----------

.. image:: http://github.com/dag/jbo/raw/master/screenshots/filter.png



jbo index
---------

The jbovlaste export for ``$JBO_LANGUAGE`` (defaultnig to ``en``)
is automatically downloaded and indexed if needed. This command lets you
keep the index up to date or index an export from a non-standard place
(file path or URL).

.. image:: http://github.com/dag/jbo/raw/master/screenshots/index.png

.. note::
    Progress bars require a recent version of the
    `progressbar <http://code.google.com/p/python-progressbar/>`_ library,
    which can be installed with:

        sudo easy_install progressbar

    This however is optional. The `python-lxml <http://codespeak.net/lxml/>`_
    library is similarly preferred if installed and might be faster,
    otherwise a standard library is used for processing XML.


jbo bashrc
----------

.. image:: http://github.com/dag/jbo/raw/master/screenshots/bashrc.png

.. image:: http://github.com/dag/jbo/raw/master/screenshots/fd.png
