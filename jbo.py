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


LANGUAGE = environ.get('JBO_LANGUAGE', 'en')
LANGUAGES = set(['id', 'ch', 'cy', 'da', 'de', 'et', 'en', 'es', 'eo', 'eu',
                 'ga', 'gl', 'ia', 'it', 'kw', 'la', 'lt', 'hu', 'mg', 'nl',
                 'pl', 'pt', 'ro', 'sl', 'sk', 'so', 'sr', 'fi', 'sv', 'test',
                 'vi', 'tr', 'wa', 'br', 'ca', 'fr', 'hr', 'art-loglan',
                 'jbo', 'no', 'sq', 'tpi', 'cs', 'el', 'be', 'bg', 'ru', 'uk',
                 'he', 'ne', 'sa', 'hi', 'gu', 'ta', 'ka', 'am', 'ar', 'fa',
                 'zh', 'ja', 'ko', 'tlh'])
LANGUAGE_NAMES = {'da': 'danish', 'nl': 'dutch', 'en': 'english',
                  'fi': 'finnish', 'fr': 'french', 'de': 'german',
                  'hu': 'hungarian', 'it': 'italian', 'no': 'norwegian',
                  'pt': 'portuguese', 'ro': 'romanian', 'ru': 'russian',
                  'es': 'spanish', 'sv': 'swedish', 'tr': 'turkish'}

try:
    import Stemmer
    if LANGUAGE not in LANGUAGE_NAMES:
        raise ImportError
except ImportError:
    stemmer = None
    def stem(word):
        return word
else:
    stemmer = Stemmer.Stemmer(LANGUAGE_NAMES[LANGUAGE])
    stem = stemmer.stemWord


DEBUG = 'JBO_DEBUG' in environ  # Show full tracebacks for errors
COLUMNS = int(environ.get('COLUMNS', 80))
DATADIR = environ.get('JBO_DATADIR', path.expanduser('~/.jbo'))
ESCAPES = isatty(1) if environ.get('JBO_ESCAPES', 'tty') == 'tty' \
                    else environ.get('JBO_ESCAPES') == 'always'
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


def ansi_encode(text, start, end=0, condition=ESCAPES):
    """Encode text with ANSI control codes."""
    if condition:
        return '\x1b[{0}m{1}\x1b[{2}m'.format(start, text, end)
    return text


def bold(text, condition=ESCAPES):
    """Make text bold if printed to a terminal."""
    return ansi_encode(text, 1, 22, condition)


def underline(text, condition=ESCAPES):
    """Underline text if printed to a terminal."""
    return ansi_encode(text, 4, 24, condition)


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
            return underline(m.group(2), True)
        elif m.group(1) == 'textbf':
            return bold(m.group(2), True)

    latex = re.sub(r'\s{3,}', '\n', latex)
    text_math = re.sub(r'\$(.+?)\$', math, latex)
    text_ansi = re.sub(r'\\(emph|textbf)\{(.+?)\}', typography, text_math)
    text_raw = re.sub(r'\\(?:emph|textbf)\{(.+?)\}', r'\1', text_math)

    return text_ansi, text_raw


def underline_references(text):
    return re.sub(r'\{(.+?)\}', lambda m: underline(m.group(1), True), text)


def compound_to_affixes(compound):
    """Split a compound word into affixes."""
    c = r'[bcdfgjklmnprstvxz]'
    v = r'[aeiou]'
    cc = r'''(?:bl|br|
                cf|ck|cl|cm|cn|cp|cr|ct|
                dj|dr|dz|fl|fr|gl|gr|
                jb|jd|jg|jm|jv|kl|kr|
                ml|mr|pl|pr|
                sf|sk|sl|sm|sn|sp|sr|st|
                tc|tr|ts|vl|vr|xl|xr|
                zb|zd|zg|zm|zv)'''
    vv = r'(?:ai|ei|oi|au)'
    rafsi3v = r"(?:{cc}{v}|{c}{vv}|{c}{v}'{v})".format(**locals())
    rafsi3 = r'(?:{rafsi3v}|{c}{v}{c})'.format(**locals())
    rafsi4 = r'(?:{c}{v}{c}{c}|{cc}{v}{c})'.format(**locals())
    rafsi5 = r'{rafsi4}{v}'.format(**locals())

    for i in xrange(1, len(compound)/3+1):
        reg = r'(?:({rafsi3})[nry]??|({rafsi4})y)'.format(**locals()) * i
        reg2 = r'^{reg}({rafsi3v}|{rafsi5})$$'.format(**locals())
        matches = re.findall(reg2, compound, re.VERBOSE)
        if matches:
            return tuple(r for r in matches[0] if r)

    return tuple()


def compound_to_metaphor(compound, affixes):
    try:
        return ' '.join(affixes[b(affix)].word
                        for affix in compound_to_affixes(compound))
    except KeyError:
        return ''


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
        self._definition = self.raw_definition = None
        self._notes = self.raw_notes = None
        self.class_ = self.metaphor = None
        self.affixes = []

    @property
    def definition(self):
        if ESCAPES:
            return self._definition
        return self.raw_definition

    @property
    def notes(self):
        if ESCAPES:
            return self._notes
        return self.raw_notes

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

    def score_tokens(word, source, score=1):
        for token in re.finditer(r"[\w']+", source, re.UNICODE):
            token = b(stem(token.group(0).lower()))
            tokens.setdefault(token, {})
            tokens[token].setdefault(word, 0)
            tokens[token][word] += score

    type_order = ('cmene', 'experimental cmavo', 'experimental gismu',
                  "fu'ivla", 'lujvo', 'cmavo cluster', 'cmavo', 'gismu')

    def process_entries(element):
        entry = Entry(u(element.get('word')), u(element.get('type')))
        if entry.type in ('gismu', 'experimental gismu'):
            affixes[b(entry.word[:4])] = affixes[b(entry.word)] = entry

        for subelement in element:
            case = lambda tag: subelement.tag == tag

            if case('definition'):
                entry._definition, entry.raw_definition = \
                    latex_to_text(u(subelement.text))
                score_tokens(entry.word, entry.raw_definition,
                             2 * type_order.index(entry.type))

            elif case('notes'):
                entry._notes, entry.raw_notes = \
                    latex_to_text(u(subelement.text))
                entry._notes = underline_references(entry._notes)
                score_tokens(entry.word, entry.raw_notes)

            elif case('selmaho'):
                entry.class_ = u(subelement.text)
                classes.setdefault(subelement.text, [])
                classes[subelement.text].append(entry.word)

            elif case('rafsi'):
                entry.affixes.append(u(subelement.text))
                affixes[subelement.text] = entry

        entries[element.get('word')] = entry

    with closing(urlopen(url)) as xml:
        root = etree.parse(xml)
        with dbopen('tokens', 'n', writeback=True) as tokens:

            with dbopen('entries', 'n', writeback=True) as entries:
                with dbopen('affixes', 'n', writeback=True) as affixes:
                    with dbopen('classes', 'n', writeback=True) as classes:
                        progress = ProgressBar(
                            widgets=['Entries  : ', Percentage(), Bar()],
                            maxval=len(root.findall('//valsi')))
                        for element in progress(root.getiterator('valsi')):
                            process_entries(element)

                    with dbopen('metaphors', 'n') as metaphors:
                        progress = ProgressBar(
                            widgets=['Metaphors: ', Percentage(), Bar()],
                            maxval=len(entries))
                        for entry in progress(entries.itervalues()):
                            if entry.type == 'lujvo':
                                metaphor = \
                                    compound_to_metaphor(entry.word, affixes)
                                if metaphor:
                                    entry.metaphor = metaphor
                                    metaphors[b(entry.metaphor)] = entry

                progress = ProgressBar(
                    widgets=['Glosses  : ', Percentage(), Bar()],
                    maxval=len(root.findall('//nlword')))
                for element in progress(root.getiterator('nlword')):
                    type_score = \
                        type_order.index(entries[element.get('valsi')].type)

                    score_tokens(u(element.get('valsi')),
                                 u(element.get('word')),
                                 3 * type_score)

                    if 'sense' in element.attrib:
                        score_tokens(u(element.get('valsi')),
                                     u(element.get('sense')))


@expose('filter')
def filter_entries(*terms):
    """Filter all entries and return matches."""
    entry_scores = {}

    from optparse import OptionParser
    parser = OptionParser()
    parser.add_option('-c', '--class', action='append')
    options, terms = parser.parse_args(list(terms))

    with dbopenbuild('classes') as classes:
        for class_ in getattr(options, 'class') or []:
            class_ = b(class_.upper().replace('H', 'h'))
            if class_ in classes:
                for entry in classes[class_]:
                    print(entry)
            else:
                print('error: unknown class {0!r}'.format(class_),
                      file=sys.stderr)

    with dbopenbuild('tokens') as tokens:
        terms = [stem(term).lower() for term in terms]

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

    def reversed_tuple(iterable):
        return tuple(reversed(iterable))

    for word, score in sorted(entry_scores.iteritems(),
                              key=reversed_tuple, reverse=True):
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

    from optparse import OptionParser
    parser = OptionParser()
    parser.add_option('-a', '--affix', action='append')
    options, args = parser.parse_args(list(args))

    def show(entry):
        if not isinstance(entry, Entry):
            entry = b(entry.replace('h', "'"))
            if entry not in entries:
                metaphor = b(compound_to_metaphor(entry, affixes))
                if metaphor in metaphors:
                    print('warning: {0!r} is defined as {1!r}'
                         .format(entry, b(metaphors[metaphor].word)),
                         file=sys.stderr)
                    entry = metaphors[metaphor]
                elif metaphor:
                    print('warning: {0!r} is not defined'.format(entry),
                          file=sys.stderr)
                    for entry in metaphor.split():
                        show(entry)
                    return
                else:
                    print('error: {0!r} is not defined'.format(entry),
                          file=sys.stderr)
                    return
            else:
                entry = entries[entry]

        header = [bold(entry)]
        if entry.affixes:
            header.append('-' + '-'.join(entry.affixes) + '-')
        if entry.class_:
            header.append(entry.class_)
        print(' '.join(header))

        for line in entry.definition.splitlines():
            print(wrapper.fill(line))

        if entry.notes is not None:
            print()
            for line in entry.notes.splitlines():
                print(wrapper.fill(line))

        print()

    if options.affix:
        with dbopenbuild('affixes') as affixes:
            for affix in options.affix:
                affix = b(affix.replace('h', "'"))
                if affix not in affixes:
                    print('error: unknown affix {0!r}'.format(affix),
                          file=sys.stderr)
                    continue
                show(affixes[affix])
    if args:
        with dbopenbuild('entries') as entries:
            with dbopen('metaphors') as metaphors:
                with dbopen('affixes') as affixes:
                    for arg in args:
                        for entry in arg.splitlines():
                            show(entry)
    elif not options.affix:
        # Need to hold off opening the database until we get an entry,
        # for when filter is piped to define and no database is built before.
        with exit_on_eof():
            entry = raw_input().strip()
        with dbopenbuild('entries') as entries:
            with dbopen('metaphors') as metaphors:
                with dbopen('affixes') as affixes:
                    while True:
                        show(entry)
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


@expose()
def shell():
    """Interactive shell with databases loaded."""
    import code
    banner = ('jbo 0.1\n'
              'Database instances: '
              'entries, tokens, classes, affixes, metaphors')
    with dbopenbuild('entries') as entries:
        with dbopen('tokens') as tokens:
            with dbopen('classes') as classes:
                with dbopen('affixes') as affixes:
                    with dbopen('metaphors') as metaphors:
                        context = dict(entries=entries, tokens=tokens,
                                       classes=classes, affixes=affixes,
                                       metaphors=metaphors)
                        code.interact(banner, local=context)


def main(argv):
    """Use jbovlaste on the command line, offline."""
    if len(argv) == 1:
        print(main.__doc__)
        print()
        pad = max(map(len, COMMANDS))
        for cmd, handler in sorted(COMMANDS.iteritems()):
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
