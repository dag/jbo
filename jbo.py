#!/usr/bin/env python
#-*- coding:utf-8 -*-

from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import print_function

from os import environ, path, isatty, makedirs
from contextlib import closing
import shelve
import pickle
from anydbm import error as DBError
from locale import getpreferredencoding
from urllib2 import urlopen
import re
from contextlib import contextmanager
from textwrap import TextWrapper, dedent
import sys

try:
    from lxml import etree
except ImportError:
    try:
        import xml.etree.cElementTree as etree
    except ImportError:
        import xml.etree.ElementTree as etree

try:
    from progressbar import ProgressBar, Percentage, Bar
    if not callable(ProgressBar()):
        raise ImportError
except ImportError:
    Percentage = Bar = object
    class ProgressBar(object):
        def __init__(self, *args, **kwargs):
            pass
        def __call__(self, iterable):
            return iterable


DEBUG = 'JBO_DEBUG' in environ  # Show full tracebacks for errors
COLUMNS = int(environ.get('COLUMNS', 80))
DATADIR = environ.get('JBO_DATADIR', path.expanduser('~/.jbo'))
ESCAPES = isatty(1) if environ.get('JBO_ESCAPES', 'tty') == 'tty' \
                    else environ.get('JBO_ESCAPES') == 'always'

LANGUAGE = environ.get('JBO_LANGUAGE', 'en')
LANGUAGES = set(['id', 'ch', 'cy', 'da', 'de', 'et', 'en', 'es', 'eo', 'eu',
                 'ga', 'gl', 'ia', 'it', 'kw', 'la', 'lt', 'hu', 'mg', 'nl',
                 'pl', 'pt', 'ro', 'sl', 'sk', 'so', 'sr', 'fi', 'sv', 'test',
                 'vi', 'tr', 'wa', 'br', 'ca', 'fr', 'hr', 'art-loglan',
                 'jbo', 'no', 'sq', 'tpi', 'cs', 'el', 'be', 'bg', 'ru', 'uk',
                 'he', 'ne', 'sa', 'hi', 'gu', 'ta', 'ka', 'am', 'ar', 'fa',
                 'zh', 'ja', 'ko', 'tlh'])

COMMANDS = {}


def expose(name=None, options=False):
    """Expose a function as a command."""
    def decorator(handler):
        command = name
        if command is None:
            command = handler.__name__
        COMMANDS[command] = handler
        return handler
    return decorator


def dbopen(db, flag='r', writeback=False):
    """Self-closing context-manager for
    opening a database from the right place.

    """
    return closing(shelve.open(path.join(DATADIR, LANGUAGE, db),
                               flag, pickle.HIGHEST_PROTOCOL, writeback))


def dbopenbuild(db, flag='r'):
    """Same as :func:`dbopen` but builds the database if needed."""
    try:
        return dbopen(db, flag)
    except DBError as error:
        if isinstance(error, IOError) and error.errno == 32:
            raise
        build_database()
        return dbopen(db, flag)


def u(object):
    """Adapt and coerce object to unicode, assuming UTF-8."""
    if isinstance(object, unicode):
        return object
    return str(object).decode('utf-8')


def b(object):
    """Adapt and coerce object to a bytestring,
    encoded as mandated by the locale if possible, otherwise UTF-8.

    """
    try:
        return u(object).encode(getpreferredencoding())
    except UnicodeError:
        return u(object).encode('utf-8')


def print(*args, **kwargs):
    """Enforce the encoding of printed output via :func:`b`;
    removes the need for some boilerplate when printing to a non-tty.

    """
    __builtins__.print(*map(b, args), **kwargs)


def ansi_encode(text, start, end=0):
    """Encode text with ANSI control codes."""
    return '\x1b[{0}m{1}\x1b[{2}m'.format(start, text, end)


def bold(text):
    """Make text bold if printed to a terminal."""
    return ansi_encode(text, 1, 22)


def underline(text):
    """Underline text if printed to a terminal."""
    return ansi_encode(text, 4, 24)


def latex_to_text(latex):
    """Turns LaTeX used in jbovlaste into a more text-friendly format,
    returns a tuple of an ANSI encoded version and a raw version.

    """
    def math(m):
        t = []
        for x in m.group(1).split('='):
            x = x.replace('{', '').replace('}', '')
            x = x.replace('*', '×')
            for digit, sub in zip('0123456789', '₀₁₂₃₄₅₆₇₈₉'):
                x = x.replace('_' + digit, sub)
            for digit, sup in zip('0123456789', '⁰¹²³⁴⁵⁶⁷⁸⁹'):  # FIXME
                x = x.replace('^' + digit, sup)
            t.append(x)
        return '='.join(t)

    def typography(m):
        if m.group(1) == 'emph':
            return underline(m.group(2))
        elif m.group(1) == 'textbf':
            return bold(m.group(2))

    latex = re.sub(r'\s{3,}', '\n', latex)
    text_math = re.sub(r'\$(.+?)\$', math, latex)
    text_ansi = re.sub(r'\\(emph|textbf)\{(.+?)\}', typography, text_math)
    text_raw = re.sub(r'\\(?:emph|textbf)\{(.+?)\}', r'\1', text_math)

    return text_ansi, text_raw


def underline_references(text):
    return re.sub(r'\{(.+?)\}', lambda m: underline(m.group(1)), text)


@contextmanager
def exit_on_eof():
    try:
        yield
    except EOFError:
        raise SystemExit


class Entry(object):
    """Data type representing an entry in jbovlaste."""

    def __init__(self, word, type=None):
        self.word, self.type = word, type
        self.definition = self.raw_definition = None
        self.notes = self.raw_notes = None

    def __str__(self):
        return self.word

    def __repr__(self):
        return '<Entry {0!r}>'.format(self.definition)


@expose('index')
def build_database(url=None):
    """Builds an index from jbovlaste.

    Usage: jbo index [url or path]

    The optional argument is a URL pointing to an XML export of jbovlaste,
    or a filesystem path, absolute or relative, to such an export.

    With no argument, the current $JBO_LANGUAGE export is downloaded from
    jbovlaste. This is slow because jbovlaste is slow to generate exports.
    """
    if LANGUAGE not in LANGUAGES:
        raise SystemExit('error: {0!r} is not an available language'
                        .format(LANGUAGE))

    if url is None:
        print('Exporting data from jbovlaste, might take a minute…')
        url = 'http://jbovlaste.lojban.org/export/xml-export.html?lang=' \
            + LANGUAGE
    if path.isfile(url):
        url = 'file://' + path.abspath(url)

    try:
        makedirs(path.join(DATADIR, LANGUAGE))
    except OSError:
        pass

    def score_tokens(tokens, word, source, score=1):
        for token in re.finditer(r"[\w']+", source, re.UNICODE):
            token = b(token.group(0).lower())
            tokens.setdefault(token, {})
            tokens[token].setdefault(word, 0)
            tokens[token][word] += score

    def process_entries(entries, tokens, element):
        entry = Entry(u(element.get('word')), u(element.get('type')))
        for subelement in element:
            case = lambda tag: subelement.tag == tag

            if case('definition'):
                entry.definition, entry.raw_definition = \
                    latex_to_text(u(subelement.text))
                score_tokens(tokens, entry.word, entry.raw_definition, 2)

            elif case('notes'):
                entry.notes, entry.raw_notes = \
                    latex_to_text(u(subelement.text))
                entry.notes = underline_references(entry.notes)
                score_tokens(tokens, entry.word, entry.raw_notes)

        entries[element.get('word')] = entry

    with closing(urlopen(url)) as xml:
        root = etree.parse(xml)
        with dbopen('tokens', 'n', writeback=True) as tokens:

            with dbopen('entries', 'n') as entries:
                progress = ProgressBar(
                    widgets=['Entries: ', Percentage(), Bar()],
                    maxval=len(root.findall('//valsi')))
                for element in progress(root.getiterator('valsi')):
                    process_entries(entries, tokens, element)

            progress = ProgressBar(
                widgets=['Glosses: ', Percentage(), Bar()],
                maxval=len(root.findall('//nlword')))
            for element in progress(root.getiterator('nlword')):
                score_tokens(tokens,
                             u(element.get('valsi')),
                             u(element.get('word')),
                             score=4)
                if 'sense' in element.attrib:
                    score_tokens(tokens,
                                 u(element.get('valsi')),
                                 u(element.get('sense')),
                                 score=2)


@expose('filter')
def filter_entries(*terms):
    """Filter all entries and return matches."""
    entry_scores = {}

    with dbopenbuild('tokens') as tokens:
        terms = map(str.lower, terms)

        # Get the hits for the first term.
        if terms:
            entry_scores = tokens.get(terms.pop(0).lower(), {})

        # Remaining hits will have to be in previous hits.
        for word in entry_scores.keys():  # iterkeys() does not let us del
            for term in terms:
                more_entry_scores = tokens.get(term, {})
                if word not in more_entry_scores:
                    del entry_scores[word]
                else:
                    entry_scores[word] += more_entry_scores[word]

    items = sorted(entry_scores.iteritems(),
                   key=lambda (key, value): value,
                   reverse=True)
    for word, score in items:
        print(word)


@expose()
def define(*args):
    """Show data for entries.

    Usage: jbo define [entry…]

    Reads standard input if no entries are supplied as arguments.
    Useful when piped with an entries-outputting command such as jbo filter:

    jbo filter <term> [term…] | jbo define
    """
    wrapper = TextWrapper(width=COLUMNS - 4)
    wrapper.initial_indent = wrapper.subsequent_indent = '    '

    def show(entry, entries):
        if entry not in entries:
            print('error: {0!r} is not defined'.format(entry), file=sys.stderr)
            return
        entry = entries[entry]
        if ESCAPES:
            print(bold(entry))
            for line in entry.definition.splitlines():
                print(wrapper.fill(line))
            if entry.notes is not None:
                print()
                for line in entry.notes.splitlines():
                    print(wrapper.fill(line))
        else:
            print(entry)
            for line in entry.raw_definition.splitlines():
                print(wrapper.fill(line))
            if entry.raw_notes is not None:
                print()
                for line in entry.raw_notes.splitlines():
                    print(wrapper.fill(line))
        print()

    if args:
        with dbopenbuild('entries') as entries:
            for arg in args:
                for entry in arg.splitlines():
                    show(entry, entries)
    else:
        # Need to hold off opening the database until we get an entry,
        # for when filter is piped to define and no database is built before.
        with exit_on_eof():
            entry = raw_input().strip()
        with dbopenbuild('entries') as entries:
            while True:
                show(entry, entries)
                with exit_on_eof():
                    entry = raw_input().strip()


@expose()
def bashrc():
    """Useful stuff to put in ~/.bashrc.

    Installation: jbo bashrc >>~/.bashrc && source ~/.bashrc

    Usage:

    def <entry> [entry…]
        Look up definitions for entries.

    fd <term> [term…]
        Find entries by search terms and display in a pager.

    """
    stuff = '''
        alias def='COLUMNS=$COLUMNS jbo define'
        function fd() {
            jbo filter "$@" | JBO_ESCAPES=always def | less -R
        }
        '''
    print(dedent(stuff))


@expose()
def help(command='help'):
    """Learn how to use a command.

    Usage: jbo help <command>

    """
    if command in COMMANDS:
        doc = COMMANDS[command].__doc__.splitlines()
        print(doc[0])
        print(dedent('\n'.join(doc[1:])).rstrip('\n'))
    else:
        raise SystemExit('{0}: command not found'.format(command))



def main(argv):
    """Use jbovlaste on the command line, offline."""
    if len(argv) == 1:
        print(main.__doc__)
        print()
        pad = max(map(len, COMMANDS))
        for cmd, handler in COMMANDS.iteritems():
            doc = (handler.__doc__ or '\n').splitlines()[0].rstrip('.')
            print('  {0:<{pad}}  {1}' .format(cmd, doc, pad=pad))
        raise SystemExit

    cmd, args = argv[1], argv[2:]
    if cmd not in COMMANDS:
        raise SystemExit('{0}: command not found'.format(cmd))

    def show_error(error):
        if DEBUG:
            raise
        raise SystemExit('error: {0}'.format(error))
    try:
        COMMANDS[cmd](*args)
    except IOError as error:
        if error.errno == 32:
            raise SystemExit('aborted')
        show_error(error)
    except KeyboardInterrupt:
        raise SystemExit('\naborted')
    except Exception as error:
        show_error(error)


if __name__ == '__main__':
    main(sys.argv)
