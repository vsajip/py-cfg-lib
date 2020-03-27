# Copyright 2004-2020 by Vinay Sajip. All Rights Reserved.
#
# Permission to use, copy, modify, and distribute this software and its
# documentation for any purpose and without fee is hereby granted,
# provided that the above copyright notice appear in all copies and that
# both that copyright notice and this permission notice appear in
# supporting documentation, and that the name of Vinay Sajip
# not be used in advertising or publicity pertaining to distribution
# of the software without specific, written prior permission.
# VINAY SAJIP DISCLAIMS ALL WARRANTIES WITH REGARD TO THIS SOFTWARE, INCLUDING
# ALL IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL
# VINAY SAJIP BE LIABLE FOR ANY SPECIAL, INDIRECT OR CONSEQUENTIAL DAMAGES OR
# ANY DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER
# IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT
# OF OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.
from setuptools import setup

from config import __version__

setup(
    name='config',
    description=('A hierarchical, easy-to-use, powerful configuration '
                 'module for Python'),
    long_description=('This module allows a hierarchical configuration '
                      'scheme with support for mappings and sequences, '
                      'cross-references between one part of the configuration '
                      'and another, the ability to flexibly access real Python '
                      'objects without full-blown eval(), an include facility, '
                      'simple expression evaluation and the ability to change, '
                      'save, cascade and merge configurations. Interfaces '
                      'easily with environment variables and command-line '
                      'options.'),
    license=('Copyright (C) 2004-2020 by Vinay Sajip. All Rights Reserved. See '
             'LICENSE for license.'),
    version=__version__,
    author='Vinay Sajip',
    author_email='vinay_sajip@red-dove.com',
    maintainer='Vinay Sajip',
    maintainer_email='vinay_sajip@red-dove.com',
    url='http://docs.red-dove.com/cfg/python.html',
    packages=['config'],
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: BSD License',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Topic :: Software Development',
    ],
    platforms='any',
)
