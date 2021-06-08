# -*- coding: utf-8 -*-
#
# Copyright 2019-2021 by Vinay Sajip. All Rights Reserved.
#
from __future__ import print_function, division, absolute_import, unicode_literals

import datetime
import io
import logging
import os
import random
import re
import sys
import unittest

from config import (Config, ConfigError, _parse_path, _path_iterator, _to_source,
                    timezone, is_identifier)
from config.parser import (Parser, ParserError, MappingBody, ListBody,
                           ODict)

from config.tokens import *

# These definitions are for resolving in the __main__ module.

LOGFILE = 'test_config.log'
ERRORLOGFILE = 'test_config_errors.log'
DEBUGLOGFILE = 'test_config_debug.log'


logger = logging.getLogger(__name__)


SEPARATOR_PATTERN = re.compile(r'^-- ([A-Z]\d+) -+')

NARROW_BUILD = (sys.platform in ('win32', 'darwin')) and (sys.version_info[0] < 3)

NARROW_BUILD_ERROR = re.compile('Invalid escape sequence at index 0: \\\\U')

def load_data(fn):
    """
    Load test data from a text file, which contains named text fragments.
    :param fn: The name of the file to load from.
    :return: A mapping of names to text fragments.
    """
    result = {}
    with io.open(fn, encoding='utf-8') as f:
        contents = f.read().splitlines()
    key = None
    value = []

    for line in contents:
        m = SEPARATOR_PATTERN.match(line)
        if not m:
            value.append(line)
        else:
            if key and value:
                result[key] = '\n'.join(value)
            key = m.groups()[0]
            value = []
    if key and value:
        result[key] = '\n'.join(value)
    return result


#
# Below is code to recursively scan an AST, converting from lists of mapping
# bodies to dicts for ease of comparison with expected data expressed as dicts
#

def convert(o):
    if not isinstance(o, (list, tuple, dict)):
        result = o
    elif isinstance(o, MappingBody):
        result = {}
        for k, v in o:
            result[k.value] = convert(v)
    elif isinstance(o, (list, tuple)):
        lv = []
        for item in o:
            lv.append(convert(item))
        if isinstance(o, tuple):  # pragma: no cover
            lv = tuple(lv)
        result = lv
    else:
        result = {}
        for k, v in o.items():
            result[k] = convert(v)
    return result


class BaseTestCase(unittest.TestCase):
    def setUp(self):
        self.parser = Parser()
        s = ' %s.%s ' % (self.__class__.__name__, self._testMethodName)
        n = len(s)
        logger.debug('--%s%s' % (s, '-' * (77 - n)))


W = lambda s: Token(WORD, s, s)
N = lambda s: Token(INTEGER, repr(s), s)
F = lambda s: Token(FLOAT, repr(s), s)
S = lambda s, r=None: Token(STRING, r or repr(s), s)
B = lambda s: Token(BACKTICK, '`%s`' % s, s)
C = lambda s: Token(COMPLEX, repr(s), s)


class MiscTestCase(BaseTestCase):
    def test_printable(self):
        if NARROW_BUILD:  # pragma: no cover
            raise unittest.SkipTest('Test not supported on narrow builds')
        self.assertTrue(is_printable(unichr(0xe01ef)))
        self.assertFalse(is_printable(unichr(0xe01f0)))

    def test_token_equality(self):
        t = W('foo')
        self.assertEqual(W('foo'), t)
        self.assertNotEqual(W('foo'), 'foo')


class FragmentTestCase(BaseTestCase):

    def test_fragments(self):
        self.maxDiff = None
        parser = self.parser
        t = parser.parse('foo', 'expr')
        self.assertEqual(t, W('foo'))
        t = parser.parse('.5', 'expr')
        self.assertEqual(t, F(0.5))
        t = parser.parse("'foo'" '"bar"', 'expr')
        self.assertEqual(t, S('foobar'))
        t = parser.parse('`sys.stderr`', 'expr')
        self.assertEqual(t, B('sys.stderr'))
        t = parser.parse('a.b.c.d', 'expr')
        self.assertEqual(t, {
            'op': DOT,
            'lhs': {
                'op': DOT,
                'lhs': {
                    'op': DOT,
                    'lhs': W('a'),
                    'rhs': W('b'),
                },
                'rhs': W('c'),
            },
            'rhs': W('d'),
        })

    def test_integers(self):
        cases = (
            ('0', N(0)),
            ('0x123a', N(0x123a)),
            ('0123', N(83)),
            ('0o123', N(83)),
            ('1234', N(1234)),
            ('1234_5678', N(12345678)),
            ('123_456_789', N(123456789)),
        )
        parser = self.parser
        for text, expected in cases:
            actual = parser.parse(text, 'value')
            self.assertEqual(actual, expected)

    def test_factor(self):
        for op in ('-', '+', '~'):
            text = '%sfoo' % op
            t = self.parser.parse(text, 'expr')
            self.assertTrue(isinstance(t, dict))
            self.assertEqual(t['op'], op)
            self.assertEqual(t['operand'], W('foo'))

    def do_test_expression(self, ops, rule):
        parser = self.parser

        aliases = {
            '<>': '!=',
            '&&': 'and',
            '||': 'or'
        }

        for op in ops:
            text = 'foo %s bar' % op
            ast = parser.parse(text, rule)
            op = aliases.get(op, op)
            expected = {'op': op, 'lhs': W('foo'), 'rhs': W('bar')}
            # if ast != expected:
                # import pdb; pdb.set_trace()
            self.assertEqual(ast, expected)
        for i in range(10):
            op1 = random.choice(ops)
            op2 = random.choice(ops)
            text = 'foo %s bar %s baz' % (op1, op2)
            ast = parser.parse(text, rule)
            op1 = aliases.get(op1, op1)
            op2 = aliases.get(op2, op2)
            expected = {'op': op2, 'lhs': {'op': op1, 'lhs': W('foo'), 'rhs': W('bar')}, 'rhs': W('baz')}
            # if ast != expected:
                # import pdb; pdb.set_trace()
            self.assertEqual(ast, expected)

    def test_term(self):
        parser = self.parser
        for s in ('[1, 2, 3]', '[1, 2, 3,]'):
            ast = parser.parse(s, 'mul_expr')
            self.assertEqual(ast, ListBody((N(1), N(2), N(3))))
        ops = ('*', '%', '/', '//')
        self.do_test_expression(ops, 'mul_expr')

    def test_add_expr(self):
        ops = ('+', '-')
        self.do_test_expression(ops, 'add_expr')

    def test_shift_expr(self):
        ops = ('<<', '>>')
        self.do_test_expression(ops, 'shift_expr')

    def test_and_expr(self):
        ops = ('&',)
        self.do_test_expression(ops, 'bitand_expr')

    def test_xor_expr(self):
        ops = ('^',)
        self.do_test_expression(ops, 'bitxor_expr')

    def test_expr(self):
        ops = ('|',)
        self.do_test_expression(ops, 'bitor_expr')

    def test_comparison(self):
        ops = ('<=', '<>', '<', '>=', '>', '==', '!=',
               'in', 'not in', 'is not', 'is')
        self.do_test_expression(ops, 'comparison')

    def test_not_test(self):
        parser = self.parser
        t = parser.parse('not foo', 'not_expr')
        self.assertTrue(isinstance(t, dict))
        self.assertEqual(t['op'], 'not')
        self.assertEqual(t['operand'], W('foo'))
        t = parser.parse('not not foo', 'not_expr')
        self.assertTrue(isinstance(t, dict))
        self.assertEqual(t['op'], 'not')
        t = t['operand']
        self.assertEqual(t['op'], 'not')
        self.assertEqual(t['operand'], W('foo'))

    def test_and_test(self):
        ops = ('and', '&&')
        self.do_test_expression(ops, 'and_expr')

    def test_or_test(self):
        ops = ('or', '||')
        self.do_test_expression(ops, 'or_expr')

    def test_slices(self):
        parser = self.parser
        ast = parser.parse('foo[start:stop:step]', 'expr')
        self.assertEqual(ast, {
            'op': COLON,
            'lhs': W('foo'),
            'rhs': (W('start'), W('stop'), W('step'))
        })
        ast = parser.parse('foo[start:stop]', 'expr')
        self.assertEqual(ast, {
            'op': COLON,
            'lhs': W('foo'),
            'rhs': (W('start'), W('stop'), None)
        })
        ast = parser.parse('foo[start:stop:]', 'expr')
        self.assertEqual(ast, {
            'op': COLON,
            'lhs': W('foo'),
            'rhs': (W('start'), W('stop'), None)
        })
        ast = parser.parse('foo[start:]', 'expr')
        self.assertEqual(ast, {
            'op': COLON,
            'lhs': W('foo'),
            'rhs': (W('start'), None, None)
        })
        ast = parser.parse('foo[:stop]', 'expr')
        self.assertEqual(ast, {
            'op': COLON,
            'lhs': W('foo'),
            'rhs': (None, W('stop'), None)
        })
        ast = parser.parse('foo[:stop:]', 'expr')
        self.assertEqual(ast, {
            'op': COLON,
            'lhs': W('foo'),
            'rhs': (None, W('stop'), None)
        })
        ast = parser.parse('foo[::step]', 'expr')
        self.assertEqual(ast, {
            'op': COLON,
            'lhs': W('foo'),
            'rhs': (None, None, W('step'))
        })
        ast = parser.parse('foo[::]', 'expr')
        self.assertEqual(ast, {
            'op': COLON,
            'lhs': W('foo'),
            'rhs': (None, None, None)
        })
        ast = parser.parse('foo[:]', 'expr')
        self.assertEqual(ast, {
            'op': COLON,
            'lhs': W('foo'),
            'rhs': (None, None, None)
        })
        ast = parser.parse('foo[start::]', 'expr')
        self.assertEqual(ast, {
            'op': COLON,
            'lhs': W('foo'),
            'rhs': (W('start'), None, None)
        })
        ast = parser.parse('foo[start::step]', 'expr')
        self.assertEqual(ast, {
            'op': COLON,
            'lhs': W('foo'),
            'rhs': (W('start'), None, W('step'))
        })
        with self.assertRaises(ParserError) as ec:
            parser.parse('foo[start::step:]', 'expr')
        self.assertIn("Expected ']', got ':' at (1, 16)", str(ec.exception))

    def test_errors(self):
        parser = self.parser
        with self.assertRaises(ParserError) as ec:
            parser.parse('/4', 'expr')
        self.assertIn('Unexpected \'/\' at (1, 1)', str(ec.exception))
        with self.assertRaises(ParserError) as ec:
            parser.parse('/4', 'value')
        self.assertIn('Unexpected \'/\' when looking for value: ', str(ec.exception))
        v = parser.parse('4 abc def', 'expr')
        self.assertEqual(v, N(4))
        self.assertFalse(parser.at_end)
        self.assertEqual(parser.remaining, 'def')


def make_tokenizer(s):
    return Tokenizer(io.StringIO(s))


# noinspection PyMethodMayBeStatic
class TokenizerTestCase(BaseTestCase):
    def setUp(self):
        logger.debug('-------------------- %s' % self._testMethodName)

    def test_raw(self):
        maker = make_tokenizer
        cases = {
            'C01': '',
            'C02': '"""\nabc#\ndef#\n"""',
            'C03': '"abc"',
            'C04': "'def'",
            'C05': "'''\nghi\njkl\n'''",
            'C06': 'foo not in bar',
            'C07': 'foo is not bar',
            'C08': '`abc.def`',
            'C09': 'a <> b',
            'C10': '@"logging.conf"',
            'C11': '${foo.bar}',
            'C12': "''''''",
            'C13': '\n\r\r\n',
            'C14': '1 -1 1.0 -1.0 1.0e8 -1.0e8 1.0e-8 -1.0e-08 .5 -.5',
            'C15': 'true',
            'C16': 'false',
            'C17': 'null',
            'C18': '\r\n',
            'C19': '!x',
            'C20': '!!x',
            'C21': "' '",
        }
        expected = {
            'C01': [],
            'C02': [(STRING, '\nabc#\ndef#\n')],
            'C03': [(STRING, 'abc')],
            'C04': [(STRING, 'def')],
            'C05': [(STRING, '\nghi\njkl\n')],
            'C06': [
                (WORD, 'foo'),
                (NOT, None),
                (IN, None),
                (WORD, 'bar'),
            ],
            'C07': [
                (WORD, 'foo'),
                (IS, None),
                (NOT, None),
                (WORD, 'bar'),
            ],
            'C08': [
                (BACKTICK, 'abc.def'),
            ],
            'C09': [
                (WORD, 'a'),
                (NEQ, None),
                (WORD, 'b'),
            ],
            'C10': [(AT, None), (STRING, 'logging.conf')],
            'C11': [
                (DOLLAR, None),
                (LCURLY, None),
                (WORD, 'foo'),
                (DOT, None),
                (WORD, 'bar'),
                (RCURLY, None),
            ],
            'C12': [(STRING, '')],
            'C13': [(NEWLINE, None), (NEWLINE, None), (NEWLINE, None)],
            'C14': [
                (INTEGER, 1),
                (INTEGER, -1),
                (FLOAT, 1.0),
                (FLOAT, -1.0),
                (FLOAT, 1.0e8),
                (FLOAT, -1.0e8),
                (FLOAT, 1.0e-8),
                (FLOAT, -1.0e-08),
                (FLOAT, 0.5),
                (FLOAT, -0.5),
            ],
            'C15': [
                (TRUE, True)
            ],
            'C16': [
                (FALSE, False)
            ],
            'C17': [
                (NONE, None)
            ],
            'C18': [
                (NEWLINE, None)
            ],
            'C19': [
                (NOT, None),
                (WORD, 'x')
            ],
            'C20': [
                (NOT, None),
                (NOT, None),
                (WORD, 'x')
            ],
            'C21': [
                (STRING, ' '),
            ]
        }

        # build up the cases

        for i, ch in enumerate(PUNCT, 1 + len(cases)):
            k = 'C%02d' % i
            cases[k] = ch
            if ch == '!':
                ch = NOT
            expected[k] = [(ch, None)]

        for i, op in enumerate(KEYWORDS, 1 + len(cases)):
            k = 'C%02d' % i
            cases[k] = op
            expected[k] = [(op, KEYWORD_VALUES.get(op))]

        operators = ('+', '-', '*', '/', '//', '%', '*', '**',
                     '<<', '>>', '<', '<=', '>', '>=', '==', '!=',
                     '.')
        for i, op in enumerate(operators, 1 + len(cases)):
            k = 'C%02d' % i
            cases[k] = op
            expected[k] = [(op, None)]

        operators += ('is', 'in', 'and', 'or')
        for i, op in enumerate(operators, 1 + len(cases)):
            k = 'C%02d' % i
            cases[k] = 'foo %s bar' % op
            expected[k] = [
                (WORD, 'foo'),
                (op, None),
                (WORD, 'bar'),
            ]
        # now go through the cases
        for k, v in sorted(cases.items()):
            actual = [(t.kind, t.value) for t in maker(v)]
            # if expected[k] != actual: import pdb; pdb.set_trace()
            self.assertEqual(expected[k], actual, 'Failed for %s: %r' % (k, v))

    def test_data(self):
        # Just test loading with nothing in the last entry
        fn = os.path.join('test', 'dummydata.txt')
        data = load_data(fn)

        fn = os.path.join('test', 'testdata.txt')
        data = load_data(fn)
        expected = {
            'C25': [
                (WORD, 'unicode'),
                (ASSIGN, None),
                (STRING, 'Grüß Gott'),
                (NEWLINE, None),
                (WORD, 'more_unicode'),
                (COLON, None),
                (STRING, 'Øresund')
            ],
            'D01': [(NEWLINE, None), (INTEGER, 123)],
            'D02': [
                (NEWLINE, None),
                (LBRACK, None),
                (INTEGER, 123),
                (COMMA, None),
                (STRING, 'abc'),
                (RBRACK, None),
            ],
            'D03': [
                (NEWLINE, None),
                (LCURLY, None),
                (WORD, 'a'),
                (COLON, None),
                (INTEGER, 7),
                (COMMA, None),
                (WORD, 'b'),
                (COLON, None),
                (FLOAT, 1.3),
                (COMMA, None),
                (WORD, 'c'),
                (COLON, None),
                (STRING, 'test'),
                (RCURLY, None),
            ],
        }
        maker = make_tokenizer
        for k, v in sorted(data.items()):
            tizer = maker(v)
            # if k == 'D01': import pdb; pdb.set_trace()
            t = [(t.kind, t.value) for t in tizer]
            if k in expected:
                self.assertEqual(t, expected[k], 'Failed for %s' % k)

    def test_positions(self):
        expected = []
        fn = os.path.join('test', 'pos.forms.cfg.txt')
        with io.open(fn, encoding='utf-8') as f:
            for line in f:
                nums = [int(s) for s in line.split()]
                assert len(nums) == 4, 'Unexpected line in test data file'
                expected.append(((nums[0], nums[1]), (nums[2], nums[3])))

        i = 0
        fn = os.path.join('test', 'forms.cfg')
        with io.open(fn, encoding='utf-8') as f:
            tokenizer = Tokenizer(f)
            while True:
                # if i == 0: import pdb; pdb.set_trace()
                t = tokenizer.get_token()
                e = expected[i]
                i += 1
                msg = 'failed at line %s' % i
                self.assertEqual(t.start, e[0], msg)
                self.assertEqual(t.end, e[1], msg)
                if t.kind == EOF:
                    break

        fn = os.path.join('test', 'forms.cfg')
        with io.open(fn, encoding='utf-8') as f:
            p = Parser(f)
            p.advance()
            p.mapping_body()
            self.assertTrue(p.at_end)

    def test_errors(self):
        cases = {
            'D01': ('0.5.7', 'Invalid character in number: .', (1, 4)),
            'D02': ('   ?', 'Unexpected character: \'?\'', (1, 4)),
            'D03': (' 0.4e-z', 'Invalid character in number: z', (1, 7)),
            'D04': (' 0.4e-8.3', 'Invalid character in number: .', (1, 8)),
            'D05': (' 089z', 'Invalid character in number: z', (1, 5)),
            'D06': ('0o89z', 'Invalid character in number: 8', (1, 3)),
            'D07': ('0X89g', 'Invalid character in number: g', (1, 5)),
            'D08': ('10z', 'Invalid character in number: z', (1, 3)),
            'D09': (' 0.4e-8Z', 'Invalid character in number: Z', (1, 8)),
            'D10': ('079', 'Badly-formed number: \'079\'', (1, 1)),
            'D11': ('`abc', 'Unterminated `-string: \'`abc\'', (1, 1)),
            'D12': ('`abc\n', 'Invalid char \n in `-string: \'`abc\'', (1, 5)),
            'D13': ('123_', 'Invalid \'_\' at end of number: 123_', (1, 4)),
            'D14': ('1__23', 'Invalid \'_\' in number: 1__', (1, 3)),
            'D15': ('1_2__3', 'Invalid \'_\' in number: 1_2__', (1, 5)),
            'D16': (' 0.4e-8_', 'Invalid \'_\' at end of number: 0.4e-8_', (1, 8)),
            'D17': (' 0.4_e-8', 'Invalid \'_\' at end of number: 0.4_', (1, 5)),
            'D18': (' 0._4e-8', 'Invalid \'_\' in number: 0._', (1, 4)),
            'D19': (r'\ ', 'Unexpected character: \\', (1, 2)),
            'D20': ('"abc', 'Unterminated quoted string: ', (1, 1)),
            'D21': ('"abc\\\ndef', 'Unterminated quoted string: ', (1, 1)),
        }

        for k, t in cases.items():
            s, msg, loc = t
            fmsg = 'Failed for %s' % k
            with self.assertRaises(TokenizerError, msg=fmsg) as ectx:
                t = make_tokenizer(s)
                # if k == 'D14': import pdb; pdb.set_trace()
                t.get_token()
            e = ectx.exception
            self.assertIn(msg, str(e), '%s: %s' % (fmsg, e))
            self.assertEqual(loc, e.location, fmsg)

    def test_empties(self):
        cases = (
            ("''", 1, 2),
            ('""', 1, 2),
            ("''''''", 1, 6),
            ('""""""', 1, 6),
        )
        for s, line, col in cases:
            tokenizer = make_tokenizer(s)
            t = tokenizer.get_token()
            self.assertEqual(t.kind, STRING)
            self.assertEqual(t.text, s)
            self.assertEqual(t.value, "")
            self.assertEqual(t.end, (line, col))

    def test_escapes(self):
        cases = (
            (r"'\a'", '\a'),
            (r"'\b'", '\b'),
            (r"'\n'", '\n'),
            (r"'\f'", '\f'),
            (r"'\r'", '\r'),
            (r"'\t'", '\t'),
            (r"'\v'", '\v'),
            (r"'\\'", '\\'),
            (r"'\''", '\''),
            (r"'\"'", '\"'),
            (r"'\xAB'", '\xAB'),
            (r"'\u2803'", '\u2803'),
            (r"'\u28A0abc\u28A0'", '\u28A0abc\u28A0'),
            (r"'\u28A0abc'", '\u28A0abc'),
            (r"'\ue000'", '\ue000'),
            (r"'\U0010FFFF'", '\U0010ffff'),
        )
        for s, e in cases:
            tokenizer = make_tokenizer(s)
            t = tokenizer.get_token()
            self.assertEqual(t.kind, STRING)
            self.assertEqual(t.text, s)
            self.assertEqual(t.value, e)

        cases = (
            r'"\z"',
            r'"\x"',
            r'"\xa"',
            r'"\xaz"',
            '"\\u"',
            '"\\u0"',
            '"\\u01"',
            '"\\u012"',
            '"\\u012z"',
            '"\\u012zA"',
            '"\\ud800"',
            '"\\udfff"',
            '"\\U00110000"',
        )

        for s in cases:
            tokenizer = make_tokenizer(s)
            self.assertRaises(TokenizerError, tokenizer.get_token)


# noinspection PyMethodMayBeStatic
class ParserTestCase(BaseTestCase):
    def test_data(self):
        fn = os.path.join('test', 'testdata.txt')
        data = load_data(fn)
        parser = self.parser

        self.maxDiff = None

        expected = {
            'C01': {'message': S('Hello, world!')},
            'C02': {'message': S('Hello, world!'), 'ident': N(42)},
            'C03': {'message': S('Hello, world!'), 'ident': N(43)},
            'C04': {'numbers': ListBody((N(0), N(0x12), N(11), N(1014)))},
            'C05': {'complex': ListBody((C(0j), C(1j), C(0.4j), C(0.7j)))},
            'C06': {'nested': {'a': W('b'), 'c': W('d'), 'e f': S('g')}},
            'C07': {
                'foo': ListBody((N(1), N(2), N(3))),
                'bar': ListBody(({'op': PLUS, 'lhs': N(4), 'rhs': W('x')}, N(5), N(6))),
                'baz': {
                    'foo': ListBody((N(1), N(2), N(3))),
                    'bar': {'op': PLUS, 'lhs': S('baz'), 'rhs': N(3)}
                },
            },
            'C08': {
                'total_period': N(100),
                'header_time': {'rhs': W('total_period'), 'lhs': F(0.3), 'op': STAR},
                'steady_time': {'rhs': W('total_period'), 'lhs': F(0.5), 'op': STAR},
                'trailer_time': {'rhs': W('total_period'), 'lhs': F(0.2), 'op': STAR},
                'base_prefix': S('/my/app/'),
                'log_file': {'rhs': S('test.log'), 'lhs': W('base_prefix'), 'op': PLUS}
            },
            'C09': {
                'message': S('Hello, world!'),
                'stream': B('sys.stderr')
            },
            'C10': {
                'messages': ListBody((
                    {
                        'stream': B('sys.stderr'),
                        'message': S('Welcome')
                    },
                    {
                        'stream': B('sys.stdout'),
                        'message': S('Welkom')
                    },
                    {
                        'stream': B('sys.stderr'),
                        'message': S('Bienvenue')
                    }
                ))
            },
            'C11': {
                'messages': ListBody((
                    {
                        'stream': B('sys.stderr'),
                        'message': W('Welcome'),
                        'name': S('Harry')
                    },
                    {
                        'stream': B('sys.stdout'),
                        'message': W('Welkom'),
                        'name': S('Ruud')
                    },
                    {
                        'stream': B('sys.stderr'),
                        'message': W('Bienvenue'),
                        'name': W('Yves')
                    }
                ))
            },
            'C12': {
                'messages': ListBody((
                    {
                        'stream': B('sys.stderr'),
                        'message': S('Welcome'),
                        'name': S('Harry')
                    },
                    {
                        'stream': B('sys.stdout'),
                        'message': S('Welkom'),
                        'name': S('Ruud')
                    },
                    {
                        'stream': {
                            'op': DOLLAR,
                            'operand': {
                                'op': DOT,
                                'lhs': {'op': LBRACK, 'lhs': W('messages'), 'rhs': N(0)},
                                'rhs': W('stream')
                            }
                        },
                        'message': S('Bienvenue'),
                        'name': W('Yves')
                    }
                ))
            },
            'C13': {
                'logging': {'op': AT, 'operand': S('logging.cfg')},
                'test': {
                    'op': DOLLAR,
                    'operand': {
                        'op': DOT,
                        'lhs': {
                            'op': DOT,
                            'lhs': {'op': DOT, 'lhs': W('logging'),
                                    'rhs': W('handler')},
                            'rhs': W('email')
                        },
                        'rhs': W('from')
                    }
                }
            },
            'C14': {
                'root': {
                    'level': W('DEBUG'),
                    'handlers': ListBody((
                        {
                            'op': DOLLAR,
                            'operand': {'op': DOT, 'lhs': W('handlers'),
                                        'rhs': W('console')}
                        },
                        {
                            'op': DOLLAR,
                            'operand': {'op': DOT, 'lhs': W('handlers'),
                                        'rhs': W('file')}
                        },
                        {
                            'op': DOLLAR,
                            'operand': {'op': DOT, 'lhs': W('handlers'),
                                        'rhs': W('email')}
                        }
                    ))
                },
                'handlers': {
                    'console': ListBody((
                        W('StreamHandler'),
                        {
                            'level': W('WARNING'),
                            'stream': B('sys.stderr')
                        }
                    )),
                    'file': ListBody((
                        W('FileHandler'),
                        {
                            'filename': {
                                'op': PLUS,
                                'lhs': {
                                    'op': PLUS,
                                    'lhs': {
                                        'op': DOLLAR,
                                        'operand': {
                                            'op': DOT,
                                            'lhs': W('app'),
                                            'rhs': W('base')
                                        }
                                    },
                                    'rhs': {
                                        'op': DOLLAR,
                                        'operand': {
                                            'op': DOT,
                                            'lhs': W('app'),
                                            'rhs': W('name')
                                        }
                                    }
                                },
                                'rhs': S('.log')
                            },
                            'mode': S('a')
                        }
                    )),
                    'socket': ListBody((
                        B('handlers.SocketHandler'),
                        {
                            'host': W('localhost'),
                            'port': B('handlers.DEFAULT_TCP_LOGGING_PORT')
                        }
                    )),
                    'nt_eventlog': ListBody((
                        B('handlers.NTEventLogHandler'),
                        {
                            'appname': {
                                'op': DOLLAR,
                                'operand': {'op': DOT, 'lhs': W('app'),
                                            'rhs': W('name')}
                            },
                            'logtype': W('Application')
                        }
                    )),
                    'email': ListBody((
                        B('handlers.SMTPHandler'),
                        {
                            'level': W('CRITICAL'),
                            'host': W('localhost'),
                            'port': N(25),
                            'from': {
                                'op': PLUS,
                                'lhs': {
                                    'op': DOLLAR,
                                    'operand': {'op': DOT, 'lhs': W('app'),
                                                'rhs': W('name')}
                                },
                                'rhs': {
                                    'op': DOLLAR,
                                    'operand': {
                                        'op': DOT,
                                        'lhs': W('app'),
                                        'rhs': W('mail_domain')
                                    }
                                }
                            },
                            'to': ListBody((
                                {
                                    'op': PLUS,
                                    'lhs': {
                                        'op': DOLLAR,
                                        'operand': {
                                            'op': DOT,
                                            'lhs': W('app'),
                                            'rhs': W('support_team')
                                        }
                                    },
                                    'rhs': {
                                        'op': DOLLAR,
                                        'operand': {
                                            'op': DOT,
                                            'lhs': W('app'),
                                            'rhs': W('mail_domain')
                                        }
                                    }
                                },
                                {
                                    'op': PLUS,
                                    'lhs': S('QA'),
                                    'rhs': {
                                        'op': DOLLAR,
                                        'operand': {
                                            'op': DOT,
                                            'lhs': W('app'),
                                            'rhs': W('mail_domain')
                                        }
                                    }
                                },
                                {
                                    'op': PLUS,
                                    'lhs': S('product_manager'),
                                    'rhs': {
                                        'op': DOLLAR,
                                        'operand': {
                                            'op': DOT,
                                            'lhs': W('app'),
                                            'rhs': W('mail_domain')
                                        }
                                    }
                                }
                            )),
                            'subject': S('Take cover')
                        }
                    ))
                },
                'loggers': {
                    'input': {
                        'handlers': ListBody((
                            {
                                'op': DOLLAR,
                                'operand': {'op': DOT, 'lhs': W('handlers'),
                                            'rhs': W('socket')}
                            },
                        ))
                    },
                    'input.xls': {
                        'handlers': ListBody((
                            {
                                'op': DOLLAR,
                                'operand': {
                                    'op': DOT,
                                    'lhs': W('handlers'),
                                    'rhs': W('nt_eventlog')
                                }
                            },
                        ))
                    }
                }
            },
            'C15': {
                'a': {
                    'op': DOT,
                    'lhs': {
                        'op': DOLLAR,
                        'operand': {'op': DOT, 'lhs': W('foo'), 'rhs': W('bar')}
                    },
                    'rhs': W('baz')
                },
                'b': {
                    'op': DOT,
                    'lhs': B('bish.bash'),
                    'rhs': W('bosh')
                }
            },
            'C16': {'test': Token(FALSE, '', False), 'another_test': Token(TRUE, '', True)},
            'C17': {'test': Token(NONE, '', None)},
            'C18': {
                'root': N(1),
                'stream': F(1.7),
                'neg': N(-1),
                'negfloat': F(-2.0),
                'posexponent': F(2.0999999e-08),
                'negexponent': F(-2.0999999e-08),
                'exponent': F(209999990.0)
            },
            'C19': {
                'mixed': ListBody((
                    S('VALIGN'),
                    ListBody((N(0), N(0))),
                    ListBody((N(-1), N(-1))),
                    S('TOP')
                )),
                'simple': ListBody((N(1), N(2))),
                'nested': ListBody((N(1), ListBody((N(2), N(3))), ListBody((N(4), ListBody((N(5), N(6)))))))
            },
            'C20': {
                'value1': N(10),
                'value2': N(5),
                'value3': S('abc'),
                'value4': S('\'ghi\'"jkl"', "'ghi'" '"jkl"'),
                'value5': N(0),
                'value6': {
                    'a': {'op': DOLLAR, 'operand': W('value1')},
                    'b': {'op': DOLLAR, 'operand': W('value2')}
                },
                'derived1': {
                    'rhs': {'op': DOLLAR, 'operand': W('value2')},
                    'lhs': {'op': DOLLAR, 'operand': W('value1')},
                    'op': PLUS
                },
                'derived2': {
                    'rhs': {'op': DOLLAR, 'operand': W('value2')},
                    'lhs': {'op': DOLLAR, 'operand': W('value1')},
                    'op': MINUS
                },
                'derived3': {
                    'rhs': {'op': DOLLAR, 'operand': W('value2')},
                    'lhs': {'op': DOLLAR, 'operand': W('value1')},
                    'op': STAR
                },
                'derived4': {
                    'rhs': {'op': DOLLAR, 'operand': W('value5')},
                    'lhs': {
                        'rhs': {'op': DOLLAR, 'operand': W('value2')},
                        'lhs': {'op': DOLLAR, 'operand': W('value1')},
                        'op': SLASH
                    },
                    'op': PLUS
                },
                'derived5': {
                    'rhs': {'op': DOLLAR, 'operand': W('value2')},
                    'lhs': {'op': DOLLAR, 'operand': W('value1')},
                    'op': MODULO
                },
                'derived6': {
                    'rhs': {'op': DOLLAR, 'operand': W('value4')},
                    'lhs': {'op': DOLLAR, 'operand': W('value3')},
                    'op': PLUS
                },
                'derived7': {
                    'rhs': {'op': DOLLAR, 'operand': W('value4')},
                    'lhs': {
                        'rhs': S('def'),
                        'lhs': {'op': DOLLAR, 'operand': W('value3')},
                        'op': PLUS
                    },
                    'op': PLUS
                },
                'derived8': {
                    'rhs': {'op': DOLLAR, 'operand': W('value4')},
                    'lhs': {'op': DOLLAR, 'operand': W('value3')},
                    'op': MINUS
                },
                'derived9': {
                    'rhs': {'op': DOLLAR, 'operand': W('value5')},
                    'lhs': {'op': DOLLAR, 'operand': W('value1')},
                    'op': SLASHSLASH
                },
                'derived10': {
                    'rhs': {'op': DOLLAR, 'operand': W('value5')},
                    'lhs': {'op': DOLLAR, 'operand': W('value1')},
                    'op': MODULO
                },
                'derived11': {'op': DOLLAR, 'operand': W('value17')},
                'derived12': {
                    'rhs': {
                        'op': DOT,
                        'lhs': {'op': DOLLAR, 'operand': W('value6')},
                        'rhs': W('b')
                    },
                    'lhs': {
                        'op': DOT,
                        'lhs': {'op': DOLLAR, 'operand': W('value6')},
                        'rhs': W('a')
                    },
                    'op': PLUS
                }
            },
            'C21': {
                'stderr': B('sys.stderr'),
                'stdout': B('sys.stdout'),
                'stdin': B('sys.stdin'),
                'debug': B('debug'),
                'DEBUG': B('DEBUG'),
                'derived': {
                    'rhs': N(10),
                    'lhs': {'op': DOLLAR, 'operand': W('DEBUG')},
                    'op': STAR
                }
            },
            'C22': {
                'messages': ListBody((
                    {
                        'stream': B('sys.stderr'),
                        'message': S('Welcome'),
                        'name': S('Harry')
                    },
                    {
                        'stream': B('sys.stdout'),
                        'message': S('Welkom'),
                        'name': S('Ruud')
                    },
                    {
                        'rhs': {'message': W('Bienvenue'), 'name': W('Yves')},
                        'lhs': {
                            'op': DOLLAR,
                            'operand': {'op': LBRACK, 'lhs': W('messages'), 'rhs': N(0)}
                        },
                        'op': PLUS
                    }
                ))
            },
            'C23': {'foo': ListBody((W('bar'), W('baz'), W('bozz')))},
            'C24': {
                'foo': ListBody((
                    S('bar'),
                    {
                        'op': PLUS,
                        'lhs': {
                            'op': MINUS,
                            'lhs': {'op': PLUS, 'lhs': W('a'), 'rhs': W('b')},
                            'rhs': W('c')
                        },
                        'rhs': W('d')
                    }
                ))
            },
            'C25': {'unicode': S('Grüß Gott'), 'more_unicode': S('Øresund')},
            'C26': {
                'foo': ListBody((
                    S('bar'),
                    {
                        'op': PLUS,
                        'lhs': {
                            'op': MINUS,
                            'lhs': {'op': PLUS, 'lhs': W('a'), 'rhs': W('b')},
                            'rhs': W('c')
                        },
                        'rhs': W('d')
                    }
                ))
            },
            'C27': {
                'foo': {
                    'op': BITXOR,
                    'lhs': {'op': BITAND, 'lhs': W('a'), 'rhs': W('b')},
                    'rhs': {'op': BITOR, 'lhs': W('c'), 'rhs': W('d')}
                }
            },
            'C28': {
                'foo': S('\na multi-line string with internal\n\'\' and "" sequences\n'),
                'bar': S('\nanother multi-line string with internal\n\'\' and "" sequences\n')
            },
            'C29': {'empty_dict': {}},
            'C30': {'empty_list': ListBody()},
            'C31': {
                'stuff_with_spaces': {
                    'classes': S('date'),
                    'kind': S('field'),
                    'label': S('A date field'),
                    'label_i18n': S(' '),
                    'name': S('foo2'),
                    'ph_i18n': S(' '),
                    'placeholder': S('Enter a date'),
                    'type': S('input'),
                }
            },
        }
        for k, v in sorted(data.items()):
            if k < 'D01':
                # if k == 'C12': import pdb; pdb.set_trace()
                try:
                    ast = parser.parse(v.strip())
                except ParserError:  # pragma: no cover
                    print('Failed for %s:' % k)
                    raise
                self.assertTrue(parser.at_end)
                ast = convert(ast)
                # if expected[k] != ast:
                    # print('Failed: %s' % k)
                self.assertEqual(expected[k], ast, 'Failed for %s' % k)
            else:
                # if k == 'D01': import pdb; pdb.set_trace()
                try:
                    with self.assertRaises(ParserError):
                        parser.parse(v)
                except AssertionError:  # pragma: no cover
                    print('Failed for %s:' % k)
                    raise
                if k in ('D02', 'D03'):
                    ast = parser.parse(v, rule='container')
                    if k == 'D02':
                        self.assertTrue(isinstance(ast, ListBody))
                    else:
                        self.assertTrue(isinstance(ast, MappingBody))

    def test_json(self):
        fn = os.path.join('test', 'forms.conf')
        with io.open(fn, encoding='utf-8') as f:
            p = Parser(f)
            p.advance()
            d = convert(p.mapping())
        self.assertTrue('refs' in d)
        self.assertTrue('fieldsets' in d)
        self.assertTrue('forms' in d)
        self.assertTrue('modals' in d)
        self.assertTrue('pages' in d)

    def test_misc_files(self):
        d = os.path.join('test', 'derived')
        for fn in sorted(os.listdir(d)):
            try:
                p = os.path.join(d, fn)
                with io.open(p, encoding='utf-8') as f:
                    parser = Parser(f)
                    parser.container()
            except TokenizerError as e:  # pragma: no cover
                m = NARROW_BUILD_ERROR.search(str(e))
                if not m or not NARROW_BUILD:
                    raise

    def test_bad_slices(self):
        parser = self.parser
        cases = (
            ('foo[1, 2:]', (1, 5), 2),
            ('foo[1:2, 3]', (1, 7), 2),
            ('foo[1:2:3, 4]', (1, 9), 2),
            ('foo[:3, 4]', (1, 6), 2),
            ('foo[::3, 4]', (1, 7), 2),
            ('foo[::::]', (1, 7), 0),
        )
        msg = 'Invalid index at %s: expected 1 expression, found %s'
        for c, pos, num in cases:
            with self.assertRaises(ParserError) as ec:
                parser.parse(c, 'expr')
            s = msg % (pos, num)
            self.assertEqual(s, str(ec.exception))

    def load(self, fn):
        with io.open(fn, encoding='utf-8') as f:
            parser = Parser(f)
            result = parser.container()
        return result

    def test_ast_positions(self):
        p = os.path.join('test', 'derived', 'empty.cfg')
        ast = self.load(p)
        self.assertEqual(ast.start, (2, 1))
        self.assertEqual(ast.end, (2, 1))
        p = os.path.join('test', 'derived', 'tools.cfg')
        ast = self.load(p)
        self.assertEqual(ast.start, (1, 1))
        self.assertEqual(ast.end, (3, 1))


# noinspection PyMethodMayBeStatic
class ConfigTestCase(BaseTestCase):
    def load(self, fn, **kwargs):
        return Config(fn, **kwargs)

    def test_path_iteration(self):
        p = _parse_path('foo[bar].baz.bozz[3].fizz')
        parts = list(_path_iterator(p))
        expected = [W('foo'), ('[', W('bar')), ('.', 'baz'), ('.', 'bozz'),
                    ('[', N(3)), ('.', 'fizz')]
        self.assertEqual(parts, expected)
        p = _parse_path('foo[start:stop:step]')
        parts = list(_path_iterator(p))
        expected = [W('foo'), (':', (W('start'), W('stop'), W('step')))]
        self.assertEqual(parts, expected)

    def test_path_across_includes(self):
        p = os.path.join('test', 'base', 'main.cfg')
        config = self.load(p)
        self.assertEqual(config['logging.handlers.file.class'],
                         logging.FileHandler)
        self.assertEqual(config['logging.handlers.file.filename'],
                         'run/server.log')
        self.assertEqual(config['logging.handlers.file.mode'], 'a')
        self.assertEqual(config['logging.handlers.error.class'],
                         logging.FileHandler)
        self.assertEqual(config['logging.handlers.error.filename'],
                         'run/server-errors.log')
        self.assertEqual(config['logging.handlers.error.mode'], 'w')
        self.assertEqual(config['redirects.freeotp.url'],
                         'https://freeotp.github.io/')
        self.assertEqual(config['redirects.freeotp.permanent'], False)

    def test_identifiers(self):
        cases = (
            ('foo', True),
            ('\u0935\u092e\u0938', True),
            ('\u73b0\u4ee3\u6c49\u8bed\u5e38\u7528\u5b57\u8868', True),
            ('foo ', False),
            ('foo[', False),
            ('foo [', False),
            ('foo.', False),
            ('foo .', False),
            ('\u0935\u092e\u0938.', False),
            ('\u73b0\u4ee3\u6c49\u8bed\u5e38\u7528\u5b57\u8868.', False),
            ('9', False),
            ('9foo', False),
            ('hyphenated-key', False),
        )
        for i, t in enumerate(cases):
            s, r = t
            msg = 'failed at %s for %r' % (i, s)
            self.assertEqual(r, is_identifier(s), msg=msg)

    def test_bad_paths(self):
        cases = (
            'foo[1, 2]',
            'foo[1] bar',
            'foo.123',
            'foo.',
            'foo[]',
            '4',
            'foo[1a] bar',
        )
        for case in cases:
            with self.assertRaises(ConfigError) as ec:
                p = _parse_path(case)
            self.assertIn('Invalid path: %s' % case, str(ec.exception))

    def test_to_source(self):
        cases = (
            'foo[::2]',
            'foo[:]',
            'foo[:2]',
            'foo[2:]',
            'foo[::1]',
            'foo[::-1]',
        )
        for c in cases:
            p = _parse_path(c)
            # import pdb; pdb.set_trace()
            s = _to_source(p)
            self.assertEqual(c, s)

    # def test_name(self):
        # p = os.path.join('test', 'derived', 'logging.cfg')
        # config = self.load(p)
        # self.assertEqual(config.name, 'foo')

    def test_close(self):
        p = os.path.join('test', 'derived', 'logging.cfg')
        f = io.open(p, encoding='utf-8')
        config = Config(f, path='foo')
        self.assertEqual(config.path, 'foo')
        self.assertFalse(f.closed)
        config.close()
        self.assertFalse(f.closed)
        config._can_close = True
        config.close()
        self.assertTrue(f.closed)

    def test_main(self):
        p = os.path.join('test', 'derived', 'main.cfg')
        ip = [os.path.join('test', 'base')]
        config = self.load(p, include_path=ip)
        self.assertRaises(ConfigError, config.get, 'nosuch')
        log_conf = config.get('logging')
        self.assertEqual(len(log_conf), 6)
        d = log_conf.as_dict()
        self.assertEqual(sorted(d), [
            'disable_existing_loggers', 'formatters', 'handlers', 'loggers',
            'root', 'version'])
        self.assertEqual(log_conf['version'], 1)
        self.assertRaises(ConfigError, log_conf.get, 'handlers.file/filename')
        self.assertRaises(ConfigError, log_conf.get, '"handlers.file/filename')
        self.assertEqual(log_conf.get('foo', 'bar'), 'bar')
        self.assertEqual(log_conf.get('foo.bar', 'baz'), 'baz')
        self.assertEqual(log_conf.get('handlers.debug.levl', 'bozz'), 'bozz')
        self.assertEqual(log_conf['handlers.file.filename'], LOGFILE)
        self.assertEqual(log_conf['handlers.debug.filename'], DEBUGLOGFILE)
        self.assertEqual(log_conf['root.handlers'], ['file', 'error', 'debug'])
        self.assertEqual(log_conf['root.handlers[:2]'], ['file', 'error'])
        self.assertEqual(log_conf['root.handlers[::2]'], ['file', 'debug'])
        self.assertEqual(log_conf.get('version'), 1)
        self.assertEqual(log_conf.get('disable_existing_loggers'), False)
        test = config.get('test')
        self.assertEqual(test['float'], 1.0e-7)
        self.assertEqual(test['float2'], 0.3)
        self.assertEqual(test['float3'], 3.0)
        self.assertEqual(test['list[1]'], 2)
        self.assertEqual(test['dict.a'], 'b')
        self.assertEqual(test['dict["a"]'], 'b')
        self.assertEqual(test['dict.c'], 'd')
        self.assertEqual(test['date'], datetime.date(2019, 3, 28))

        tzinfo = timezone(datetime.timedelta(hours=5, minutes=30))
        expected = datetime.datetime(2019, 3, 28, 23, 27, 4, 314159, tzinfo=tzinfo)
        self.assertEqual(test['date_time'], expected)

        if sys.version_info[:2] >= (3, 7):
            tzinfo = timezone(datetime.timedelta(hours=5, minutes=30, seconds=25,
                                                 microseconds=123456))
            expected = datetime.datetime(2019, 3, 28, 23, 27, 4, 314159,
                                         tzinfo=tzinfo)
            self.assertEqual(test['offset_time'], expected)

        expected = datetime.datetime(2019, 3, 28, 23, 27, 4, 271828)
        self.assertEqual(test['alt_date_time'], expected)
        expected = datetime.datetime(2019, 3, 28, 23, 27, 4, 0)
        self.assertEqual(test['no_ms_time'], expected)
        self.assertEqual(test['computed'], 3.3)
        self.assertEqual(test['computed2'], 2.7)
        self.assertAlmostEqual(test['computed3'], 0.9)
        self.assertAlmostEqual(test['computed4'], 10.0)
        base = config['base']
        #  print(len(config['routes']))
        self.assertEqual(config['combined_list'],
                         ['derived_foo', 'derived_bar', 'derived_baz',
                          'test_foo', 'test_bar', 'test_baz',
                          'base_foo', 'base_bar', 'base_baz'])
        self.assertGreater(len(base['routes']), 0)
        self.assertEqual(config['combined_map_1'], {
            'foo_key': 'base_foo',
            'bar_key': 'base_bar',
            'baz_key': 'base_baz',
            'base_foo_key': 'base_foo',
            'base_bar_key': 'base_bar',
            'base_baz_key': 'base_baz',
            'derived_foo_key': 'derived_foo',
            'derived_bar_key': 'derived_bar',
            'derived_baz_key': 'derived_baz',
            'test_foo_key': 'test_foo',
            'test_bar_key': 'test_bar',
            'test_baz_key': 'test_baz',
        })
        self.assertEqual(config['combined_map_2'], {
            'derived_bar_key': 'derived_bar',
            'derived_baz_key': 'derived_baz',
            'derived_foo_key': 'derived_foo'
        })
        self.assertEqual(config['number_3'],
                         config['number_1'] & config['number_2'])
        self.assertEqual(config['number_4'],
                         config['number_1'] ^ config['number_2'])

        v = config['logging.version']
        self.assertEqual(v, 1)
        cases = (
            ('logging[4]', 'string required, but found 4'),
            ('logging[:4]', 'slices can only operate on lists'),
        )
        for s, m in cases:
            with self.assertRaises(ConfigError) as ec:
                config[s]
            self.assertIn(m, str(ec.exception))

    def test_cache(self):
        p = os.path.join('test', 'derived', 'main.cfg')
        ip = [os.path.join('test', 'base')]
        config = self.load(p, include_path=ip, cache=True)
        log_conf = config.get('logging')
        self.assertEqual(len(log_conf), 6)
        d = log_conf.as_dict()
        self.assertEqual(sorted(d), [
            'disable_existing_loggers', 'formatters', 'handlers', 'loggers',
            'root', 'version'])
        self.assertEqual(log_conf['version'], 1)
        self.assertEqual(log_conf.get('foo', 'bar'), 'bar')
        self.assertEqual(log_conf.get('foo.bar', 'baz'), 'baz')
        self.assertEqual(log_conf.get('handlers.debug.levl', 'bozz'), 'bozz')
        self.assertEqual(log_conf['handlers.file.filename'], LOGFILE)
        self.assertEqual(log_conf['handlers.debug.filename'], DEBUGLOGFILE)
        self.assertEqual(log_conf['root.handlers'], ['file', 'error', 'debug'])
        self.assertEqual(log_conf['root.handlers[:2]'], ['file', 'error'])
        self.assertEqual(log_conf.get('version'), 1)
        self.assertEqual(log_conf.get('disable_existing_loggers'), False)
        expected = {'logging'}
        self.assertEqual(expected, set(config._cache))
        expected = {'version', 'foo', 'foo.bar', 'handlers.debug.levl',
                    'handlers.file.filename', 'handlers.debug.filename',
                    'root.handlers', 'root.handlers[:2]',
                    'disable_existing_loggers'}
        self.assertEqual(expected, set(log_conf._cache))
        # check fetch from cache returns the same value
        self.assertEqual(log_conf.get('disable_existing_loggers'), False)
        self.assertEqual(log_conf['disable_existing_loggers'], False)

    def test_datetimes(self):
        config = Config(None)
        cases = (
            ('2019-03-30', datetime.date(2019, 3, 30)),
            ('2019-03-30 11:11:11.123456',
             datetime.datetime(2019, 3, 30, 11, 11, 11, 123456)),
            ('2019-03-30T11:11:11.123456',
             datetime.datetime(2019, 3, 30, 11, 11, 11, 123456)),
            ('2019-03-30 11:11:11.123456-05:00',
             datetime.datetime(2019, 3, 30, 11, 11, 11, 123456,
                               tzinfo=timezone(-datetime.timedelta(minutes=300)))),
            ('2019-03-30T11:11:11.123456-05:00',
             datetime.datetime(2019, 3, 30, 11, 11, 11, 123456,
                               tzinfo=timezone(-datetime.timedelta(minutes=300)))),
        )
        for s, v in cases:
            self.assertEqual(config.convert_string(s), v)

    def test_bad_conversions(self):
        config = Config(None)
        self.assertTrue(config.strict_conversions)
        cases = (
            'foo',
        )
        for case in cases:
            with self.assertRaises(ConfigError) as e:
                config.convert_string(case)
            self.assertEqual('Unable to convert string \'%s\'' % case,
                             str(e.exception))
        config.strict_conversions = False
        for case in cases:
            self.assertEqual(config.convert_string(case), case)

    def test_custom_conversion(self):
        def custom_converter(s, config):
            return (s, s)

        config = Config(None)
        config._string_converter = custom_converter
        for s in ('foo', 'bar', 'baz'):
            r = config.convert_string(s)
            self.assertEqual(r, (s, s))

    def test_forms(self):
        p = os.path.join('test', 'derived', 'forms.cfg')
        config = self.load(p)
        d = config.get('modals.deletion.contents[0]')
        self.assertEqual(d['id'], 'frm-deletion')
        self.assertEqual(d['contents'][-1]['kind'], 'row')
        self.assertEqual(d['contents'][-1]['id'], 'deletion-contents')
        cases = (
            ('refs.delivery_address_field', {
                'kind': 'field', 'type': 'textarea', 'name': 'postal_address',
                'label': 'Postal address', 'label_i18n': 'postal-address',
                'short_name': 'address',
                'placeholder': 'We need this for delivering to you',
                'ph_i18n': 'your-postal-address', 'message': ' ', 'required': True,
                'attrs': {'minlength': 10}, 'grpclass': 'col-md-6'}),
            ('refs.delivery_instructions_field', {
                'kind': 'field', 'type': 'textarea',
                'name': 'delivery_instructions',
                'label': 'Delivery Instructions', 'short_name': 'notes',
                'placeholder': 'Any special delivery instructions?',
                'message': ' ', 'label_i18n': 'delivery-instructions',
                'ph_i18n': 'any-special-delivery-instructions',
                'grpclass': 'col-md-6'}),
            ('refs.verify_field', {
                'kind': 'field', 'type': 'input', 'name': 'verification_code',
                'label': 'Verification code', 'label_i18n': 'verification-code',
                'short_name': 'verification code',
                'placeholder': 'Your verification code (NOT a backup code)',
                'ph_i18n': 'verification-not-backup-code',
                'attrs': {'minlength': 6, 'maxlength': 6, 'autofocus': True},
                'append': {'label': 'Verify', 'type': 'submit',
                           'classes': 'btn-primary'},
                'message': ' ', 'required': True}),
            ('refs.signup_password_field', {
                'kind': 'field', 'type': 'password', 'label': 'Password',
                'label_i18n': 'password', 'message': ' ', 'name': 'password',
                'ph_i18n': 'password-wanted-on-site',
                'placeholder': 'The password you want to use on this site',
                'required': True, 'toggle': True}),
            ('refs.signup_password_conf_field', {
                'kind': 'field', 'type': 'password', 'name': 'password_conf',
                'label': 'Password confirmation',
                'label_i18n': 'password-confirmation',
                'placeholder': 'The same password, again, ' +
                               'to guard against mistyping',
                'ph_i18n': 'same-password-again', 'message': ' ',
                'toggle': True, 'required': True}),
            ('fieldsets.signup_ident[0].contents[0]', {
                'kind': 'field', 'type': 'input', 'name': 'display_name',
                'label': 'Your name', 'label_i18n': 'your-name',
                'placeholder': 'Your full name', 'ph_i18n': 'your-full-name',
                'message': ' ', 'data_source': 'user.display_name',
                'required': True, 'attrs': {'autofocus': True},
                'grpclass': 'col-md-6'}),
            ('fieldsets.signup_ident[0].contents[1]', {
                'kind': 'field', 'type': 'input', 'name': 'familiar_name',
                'label': 'Familiar name', 'label_i18n': 'familiar-name',
                'placeholder': 'If not just the first word in your full name',
                'ph_i18n': 'if-not-first-word',
                'data_source': 'user.familiar_name', 'message': ' ',
                'grpclass': 'col-md-6'}),
            ('fieldsets.signup_ident[1].contents[0]', {
                'kind': 'field', 'type': 'email', 'name': 'email',
                'label': 'Email address (used to sign in)',
                'label_i18n': 'email-address', 'short_name': 'email address',
                'placeholder': 'Your email address',
                'ph_i18n': 'your-email-address', 'message': ' ',
                'required': True, 'data_source': 'user.email',
                'grpclass': 'col-md-6'}),
            ('fieldsets.signup_ident[1].contents[1]', {
                'kind': 'field', 'type': 'input', 'name': 'mobile_phone',
                'label': 'Phone number', 'label_i18n': 'phone-number',
                'short_name': 'phone number', 'placeholder': 'Your phone number',
                'ph_i18n': 'your-phone-number', 'classes': 'numeric',
                'message': ' ', 'prepend': {'icon': 'phone'},
                'attrs': {'maxlength': 10},
                'required': True, 'data_source': 'customer.mobile_phone',
                'grpclass': 'col-md-6'}),
        )
        for k, v in cases:
            d = config.get(k)
            # if d != v:
                # print(d)
            msg = 'Failed for %s' % k
            self.assertEqual(d, v, msg)

    def test_example(self):
        p = os.path.join('test', 'derived', 'example.cfg')
        try:
            config = self.load(p)
        except TokenizerError as e:  # pragma: no cover
            m = NARROW_BUILD_ERROR.search(str(e))
            if not m or not NARROW_BUILD:
                raise
            raise unittest.SkipTest('Test not supported on narrow builds')

        # strings
        self.assertEqual(config['snowman_escaped'], config['snowman_unescaped'])
        self.assertEqual(config['snowman_escaped'], '\u2603')
        self.assertEqual(config['face_with_tears_of_joy'], '\U0001F602')
        self.assertEqual(config['unescaped_face_with_tears_of_joy'], '\U0001F602')
        strings = config['strings']
        self.assertEqual(strings[0], "Oscar Fingal O'Flahertie Wills Wilde")
        self.assertEqual(strings[1], 'size: 5"')
        self.assertEqual(strings[2], """Triple quoted form\ncan span\n'multiple' lines""")
        self.assertEqual(strings[3], '''with "either"\nkind of 'quote' embedded within''')
        # special strings
        self.assertIs(config['special_value_1'], sys.stderr)
        if os.name != 'nt':
            self.assertEqual(config['special_value_2'], os.path.expanduser('~'))
        else:
            self.assertEqual(config['special_value_2a'], os.path.expanduser('~'))
        tzinfo = timezone(datetime.timedelta(hours=5, minutes=30))
        dtv = datetime.datetime(2019,3,28,23,27,4, 314159, tzinfo=tzinfo)
        self.assertEqual(config['special_value_3'], dtv)
        self.assertEqual(config['special_value_4'], 'bar')
        # integers
        self.assertEqual(config['decimal_integer'], 123)
        self.assertEqual(config['hexadecimal_integer'], 0x123)
        self.assertEqual(config['octal_integer'], 0o123)
        self.assertEqual(config['binary_integer'], 0b000100100011)
        # floats
        self.assertAlmostEqual(config['common_or_garden'], 123.456)
        self.assertAlmostEqual(config['leading_zero_not_needed'], 0.123)
        self.assertAlmostEqual(config['trailing_zero_not_needed'], 123.0)
        self.assertAlmostEqual(config['scientific_large'], 1.0e6)
        self.assertAlmostEqual(config['scientific_small'], 1.0e-7)
        self.assertAlmostEqual(config['expression_1'], 3.14159)
        self.assertEqual(config['expression_2'], 3 + 2j)
        # complex
        self.assertEqual(config['list_value[4]'], 1 + 3j)
        # boolean
        self.assertIs(config['boolean_value'], True)
        self.assertIs(config['opposite_boolean_value'], False)
        self.assertIs(config['computed_boolean_1'], True)
        self.assertIs(config['computed_boolean_2'], False)
        # list
        self.assertEqual(config['incl_list'], ['a', 'b', 'c'])
        # mapping
        self.assertEqual(config['incl_mapping'].as_dict(), {
            'bar': 'baz',
            'foo': 'bar'
        })
        self.assertEqual(config['incl_mapping_body'].as_dict(), {
            'baz': 'bozz',
            'fizz': 'buzz'
        })

    def test_circular(self):
        p = os.path.join('test', 'derived', 'test.cfg')
        config = self.load(p)
        # circular references
        with self.assertRaises(ConfigError) as e:
            _ = config['circ_list[1]']
        self.assertEqual('Circular reference: circ_list[1] (46, 7)',
                         str(e.exception))
        with self.assertRaises(ConfigError) as e:
            _ = config['circ_list[]']
        self.assertEqual('Invalid path: circ_list[]: Invalid index at (1, 11): '
                         'expected 1 expression, found 0', str(e.exception))
        with self.assertRaises(ConfigError) as e:
            _ = config['circ_list [1, 2]']
        self.assertEqual('Invalid path: circ_list [1, 2]: Invalid index at '
                         '(1, 12): expected 1 expression, found 2',
                         str(e.exception))
        with self.assertRaises(ConfigError) as e:
            _ = config['circ_map.a']
        self.assertEqual('Circular reference: circ_map.a (53, 10), '
                         'circ_map.b (51, 10), '
                         'circ_map.c (52, 10)', str(e.exception))

    def test_dupes(self):
        p = os.path.join('test', 'derived', 'dupes.cfg')
        with self.assertRaises(ConfigError) as e:
            self.load(p)
        self.assertIn('Duplicate key', str(e.exception))
        config = self.load(p, no_duplicates=False)
        self.assertEqual(config['foo'], 'not again!')

    def test_context(self):
        p = os.path.join('test', 'derived', 'context.cfg')
        context = {'bozz': 'bozz-bozz'}
        config = self.load(p, context=context)
        self.assertEqual(config['baz'], 'bozz-bozz')
        with self.assertRaises(ConfigError) as ec:
            _ = config['bad']
        self.assertIn('Unknown variable ', str(ec.exception))

    def test_expressions(self):
        p = os.path.join('test', 'derived', 'test.cfg')
        config = self.load(p)
        d = config['dicts_added']
        self.assertEqual(d, {'a': 'b', 'c': 'd'})
        d = config['nested_dicts_added']
        self.assertEqual(d, {'a': {'b': 'c', 'w': 'x'},
                             'd': {'e': 'f', 'y': 'z'}})
        v = config['lists_added']
        self.assertEqual(v, ['a', 1, 'b', 2])
        v = config['list[:2]']
        self.assertEqual(v, [1, 2])
        d = config['dicts_subtracted']
        self.assertEqual(d, {'a': 'b'})
        d = config['nested_dicts_subtracted']
        self.assertEqual(d, {})
        d = config['dict_with_nested_stuff']
        expected = {
            'a_list': [1, 2, {'a': 3}],
            'a_map': {
                'k1': ['b', 'c', {'d': 'e'}]
            }
        }
        self.assertEqual(d, expected)
        lv = config['dict_with_nested_stuff.a_list[:2]']
        self.assertEqual(lv, [1, 2])
        self.assertEqual(config['unary'], -4)
        self.assertEqual(config['abcdefghijkl'], 'mno')
        self.assertEqual(config['power'], 8)
        self.assertEqual(config["c3"], 3 + 1j)
        self.assertEqual(config["c4"], 5 + 5j)
        self.assertEqual(config["computed8"], 2)
        self.assertEqual(config["computed9"], 160)
        self.assertEqual(config["computed10"], 62)

        cases = (
            ('bad_include', '@ operand must be a string'),
            ('computed7', 'not found in configuration'),
            ('dict[4]', 'string required, but found 4'),
            ('dict[4:]', 'slices can only operate on lists'),
            ('list[\'foo\']', 'integer or slice required, but found \'foo\''),
            ('bad_interp', 'Unable to convert string '),
        )
        for s, m in cases:
            with self.assertRaises(ConfigError) as ec:
                config[s]
            self.assertIn(m, str(ec.exception))

    def test_slices_and_indices(self):
        p = os.path.join('test', 'derived', 'test.cfg')
        config = self.load(p)
        the_list = ['a', 'b', 'c', 'd', 'e', 'f', 'g']

        # slices

        cases = (
            ('test_list[:]', the_list),
            ('test_list[::]', the_list),
            ('test_list[:20]', the_list),
            ('test_list[-20:4]', the_list[-20:4]),
            ('test_list[-20:20]', the_list[-20:20]),
            ('test_list[2:]', the_list[2:]),
            ('test_list[-3:]', the_list[-3:]),
            ('test_list[-2:2:-1]', the_list[-2:2:-1]),
            ('test_list[::-1]', the_list[::-1]),
            ('test_list[2:-2:2]', the_list[2:-2:2]),
            ('test_list[::2]', the_list[::2]),
            ('test_list[::3]', the_list[::3]),
            ('test_list[::2][::3]', the_list[::2][::3]),
            # ('', ),
        )
        for k, v in cases:
            self.assertEqual(v, config[k])

        # indices

        for i, v in enumerate(the_list):
            key = 'test_list[%d]' % i
            self.assertEqual(v, config[key])

        # negative indices

        n = len(the_list)
        for i in range(n, 0, -1):
            key = 'test_list[-%d]' % i
            self.assertEqual(the_list[n - i], config[key])

        # invalid indices

        for i in (n, n + 1, -(n + 1), -(n + 2)):
            key = 'test_list[%d]' % i
            with self.assertRaises(ConfigError) as ec:
                config[key]
            self.assertIn('index out of range', str(ec.exception))

    def test_bad_rule(self):
        parser = self.parser
        self.assertRaises(ValueError, parser.parse, 'foo', rule='bar')

    def test_absolute_include_path(self):
        p = os.path.abspath(os.path.join('test', 'derived', 'test.cfg'))
        # replace backslashes for Windows - avoids having to escape them
        s = 'test: @"%s"' % p.replace('\\', '/')
        cfg = Config(io.StringIO(s))
        self.assertEqual(2, cfg['test.computed6'])

    def test_interpolation(self):
        p = os.path.join('test', 'derived', 'test.cfg')
        config = self.load(p)
        self.assertEqual(config['interp'], "A-4 a test_foo True 10 1e-07 1 b "
                                           "[a, c, e, g]Z")
        self.assertEqual(config['interp2'], '{a: b}')

    def test_nested_include_path(self):
        p = os.path.join('test', 'base', 'top.cfg')
        ip = [os.path.join('test', 'derived'), os.path.join('test', 'another')]
        config = self.load(p, include_path=ip)
        self.assertEqual(config['level1.level2.final'], 42)

#
# Compatibility tests with older code base
#

STREAMS = {
    "simple_1":
        """
message: 'Hello, world!'
""",
    "malformed_1":
        """
123
""",
    "malformed_2":
        """
[ 123, 'abc' ]
""",
    "malformed_3":
        """
{ a : 7, b : 1.3, c : 'test' }
""",
    "malformed_4":
        """
test: $a [7] # note space before bracket
""",
    "malformed_5":
        """
test: 'abc'
test: 'def'
""",
    "wellformed_1":
        """
test: $a[7] # note no space before bracket
""",
    "boolean_1":
        """
test : false
another_test: true
""",
    "boolean_2":
        """
test : 'false'
another_test: 'true'
""",
    "none_1":
        """
test : null
""",
    "none_2":
        """
test : 'none'
""",
    "number_1":
        """
root: 1
stream: 1.7
neg: -1
negfloat: -2.0
posexponent: 2.0999999e-08
negexponent: -2.0999999e-08
exponent: 2.0999999e08
""",
    "sequence_1":
        """
mixed: [ "VALIGN", [ 0, 0 ], [ -1, -1 ], "TOP" ]
simple: [1, 2]
nested: [1, [2, 3], [4, [5, 6]]]
""",
    "include_1":
        """
included: @'include_2'
""",
    "include_2":
        """
test: 123
another_test: 'abc'
""",
    "expr_1":
        """
value1 : 10
value2 : 5
value3 : 'abc'
value4 : 'ghi'
value5 : 0
value6 : { 'a' : ${value1}, 'b': ${value2} }
derived1 : ${value1} + ${value2}
derived2 : ${value1} - ${value2}
derived3 : ${value1} * ${value2}
derived4 : ${value1} / ${value2}
derived5 : ${value1} % ${value2}
derived6 : ${value3} + ${value4}
derived7 : ${value3} + 'def' + ${value4}
derived8 : ${value3} - ${value4} # meaningless
derived9 : ${value1} / ${value5}    # div by zero
derived10 : ${value1} % ${value5}   # div by zero
derived11 : ${value17}    # doesn't exist
derived12 : ${value6.a} + ${value6.b}
""",
    "eval_1":
        """
stderr : `sys.stderr`
stdout : `sys.stdout`
stdin : `sys.stdin`
debug : `debug`
DEBUG : `DEBUG`
derived: $DEBUG * 10
""",
    "merge_1":
        """
value1: True
value3: [1, 2, 3]
value5: [ 7 ]
value6: { 'a' : 1, 'c' : 3 }
""",
    "merge_2":
        """
value2: False
value4: [4, 5, 6]
value5: ['abc']
value6: { 'b' : 2, 'd' : 4 }
""",
    "merge_3":
        """
value1: True
value2: 3
value3: [1, 3, 5]
value4: [1, 3, 5]
""",
    "merge_4":
        """
value1: False
value2: 4
value3: [2, 4, 6]
value4: [2, 4, 6]
""",
    "list_1":
        """
verbosity : 1
""",
    "list_2":
        """
verbosity : 2
program_value: 4
""",
    "list_3":
        """
verbosity : 3
suite_value: 5
""",
    "get_1":
        """
value1 : 123
value2 : 'abcd'
value3 : True
value4 : None
value5:
{
    value1 : 123
    value2 : 'abcd'
    value3 : True
    value4 : None
}
""",
    "multiline_1":
        """
value1: '''Value One
Value Two
'''
value2: \"\"\"Value Three
Value Four\"\"\"
"""
}


def make_stream(name):
    s = io.StringIO(STREAMS[name])
    s.name = name
    return s


class CompatibilityTestCase(BaseTestCase):
    def setUp(self):
        super(CompatibilityTestCase, self).setUp()
        self.cfg = Config(None)

    def test_creation(self):
        self.assertEqual(0, len(self.cfg))  # should be empty

    def test_simple(self):
        self.cfg.load(make_stream('simple_1'))
        self.assertTrue('message' in self.cfg)
        self.assertFalse('root' in self.cfg)
        self.assertFalse('stream' in self.cfg)
        self.assertFalse('load' in self.cfg)
        self.assertFalse('save' in self.cfg)

    def test_value_only(self):
        self.assertRaises(ParserError, self.cfg.load,
                          make_stream('malformed_1'))
        self.assertRaises(ConfigError, self.cfg.load,
                          make_stream('malformed_2'))

    def test_duplicate(self):
        self.assertRaises(ConfigError, self.cfg.load,
                          make_stream('malformed_5'))

    def test_boolean(self):
        self.cfg.load(make_stream('boolean_1'))
        self.assertIs(True, self.cfg['another_test'])
        self.assertIs(False, self.cfg['test'])

    def test_not_boolean(self):
        self.cfg.load(make_stream('boolean_2'))
        self.assertEqual('true', self.cfg['another_test'])
        self.assertEqual('false', self.cfg['test'])

    def test_none(self):
        self.cfg.load(make_stream('none_1'))
        self.assertIsNone(self.cfg['test'])

    def test_not_none(self):
        self.cfg.load(make_stream('none_2'))
        self.assertEqual('none', self.cfg['test'])

    def test_number(self):
        self.cfg.load(make_stream('number_1'))
        self.assertEqual(1, self.cfg['root'])
        self.assertEqual(1.7, self.cfg['stream'])
        self.assertEqual(-1, self.cfg['neg'])
        self.assertEqual(-2.0, self.cfg['negfloat'])
        self.assertAlmostEqual(-2.0999999e-08, self.cfg['negexponent'])
        self.assertAlmostEqual(2.0999999e-08, self.cfg['posexponent'])
        self.assertAlmostEqual(2.0999999e08, self.cfg['exponent'])

    def test_expression(self):
        import warnings

        self.cfg.load(make_stream('expr_1'))
        self.assertEqual(15, self.cfg['derived1'])
        self.assertEqual(5, self.cfg['derived2'])
        self.assertEqual(50, self.cfg['derived3'])
        self.assertEqual(2, self.cfg['derived4'])
        self.assertEqual(0, self.cfg['derived5'])
        self.assertEqual('abcghi', self.cfg['derived6'])
        self.assertEqual('abcdefghi', self.cfg['derived7'])
        self.assertRaises(ConfigError, lambda x: x['derived8'], self.cfg)
        self.assertRaises(ZeroDivisionError, lambda x: x['derived9'], self.cfg)
        self.assertRaises(ZeroDivisionError, lambda x: x['derived10'], self.cfg)
        self.assertRaises(ConfigError, lambda x: x['derived11'], self.cfg)
        with warnings.catch_warnings(record=True) as w:
            if sys.version_info[0] < 3:
                warnings.simplefilter('always')
            self.assertEqual(15, self.cfg.derived12)
        self.assertEqual(len(w), 1)
        self.assertIs(w[0].category, DeprecationWarning)
        self.assertIn('Attribute access is deprecated (derived12); '
                      'use indexed access instead.', str(w[0].message))

    def test_multiline(self):
        cfg = self.cfg
        cfg.load(make_stream('multiline_1'))
        self.assertEqual('Value One\nValue Two\n', cfg.get('value1'))
        self.assertEqual('Value Three\nValue Four', cfg.get('value2'))

    def test_sequence(self):
        cfg = self.cfg
        strm = make_stream('sequence_1')
        cfg.load(strm)
        self.assertEqual(cfg['simple'], [1, 2])
        self.assertEqual(cfg['nested'],
                         [1, [2, 3], [4, [5, 6]]])
        self.assertEqual(cfg['mixed'],
                         ['VALIGN', [0, 0], [-1, -1], 'TOP'])


if __name__ == '__main__':  # pragma: no branch
    logging.basicConfig(filename='test_config.log', filemode='w',
                        level=logging.DEBUG, format='%(message)s')
    unittest.main()
