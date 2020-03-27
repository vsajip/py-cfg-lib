The CFG configuration format is a text format for configuration files which is similar to, and a superset of, the JSON format. It dates from [2008](https://wiki.python.org/moin/HierConfig) and has the following aims:

* Allow a hierarchical configuration scheme with support for key-value mappings and lists.
* Support cross-references between one part of the configuration and another.
* Provide the ability to compose configurations (using include and merge facilities).
* Provide the ability to access real application objects safely.

It overcomes a number of drawbacks of JSON when used as a configuration format:

* JSON is more verbose than necessary.
* JSON doesn’t allow comments.
* JSON doesn’t allow trailing commas in lists and mappings.

Backwards-incompatible Changes
==============================
This library is a reimplementation of [an earlier version](https://bitbucket.org/vinay.sajip/config) written in 2008. The latest version of that implementation is [0.4.2](https://pypi.org/project/config/0.4.2/), released in May 2019. A new implementation was started in 2018 (with some changes to the format) and this differs from the earlier implementation in a number of ways:

Format
------

* The format now uses `true`, `false` and `null` rather than `True`, `False` and `None`. This is for JSON compatibility.
* The format now uses `${A.B.C}` for references rather than `$A.B.C`. This is to allow better expressivity in paths.
* Multiple strings following one another are concatenated into a single string.

Code
----

* There is no support for writing configurations through the API, only for reading them.
* `config` is now a package rather than a module.
* The classes `ConfigInputStream`, `ConfigOutputStream`, `ConfigList`, `ConfigMerger`, `ConfigReader`, `Container`, `Expression`, `Mapping`, `Namespace`, `Reference`, `SeqIter` and `Sequence` are not in the new implementation.
* The `ConfigResolutionError` exception is not in the new implementation.
* The `Config` class in the new implementation is completely different.
* The functions `defaultMergeResolve`, `defaultStreamOpener`, `isWord`, `makePath` and `overwriteMergeResolve` are not in the new implementation.

If your code relies on specific features of the old implementation, be sure to specify `config<0.5` in your dependencies.

Modules
-------
This Python implementation is divided into three modules:

* `config` contains the high-level API which you will normally interact with.
* `config.tokens` contains code pertaining to tokens and lexical scanning of CFG.
* `config.parser` contains code pertaining to parsing CFG and returning Abstract Syntax Trees (ASTs).


Installation
============
You can use this package using ``pip install config>= 0.5.0`` and then importing ``config`` in your code. You should install into a virtual environment.

Getting Started with CFG in Python
==================================
A configuration is represented by an instance of the `Config` class. The constructor for this class can be passed a filename or a stream which contains the text for the configuration. The text is read in, parsed and converted to an object that you can then query. A simple example:

```
a: 'Hello, '
b: 'world!'
c: {
  d: 'e'
}
'f.g': 'h'
christmas_morning: `2019-12-25 08:39:49`
home: `$HOME`
foo: `$FOO|bar`
```

Loading a configuration
=======================
The configuration above can be loaded as shown below. In an interactive shell:

```
>>> import io, os, sys, config
>>> cfg = config.Config('test0.cfg')
```

Access elements with keys
=========================
Accessing elements of the configuration with a simple key is just like using a dictionary:

```
>>> cfg['a']
'Hello, '
>>> cfg['b']
'world!'
```

Access elements with paths
==========================
As well as simple keys, elements  can also be accessed using `path` strings:
```
>>> cfg['c.d']
'e'
```
Here, the desired value is obtained in a single step, by (under the hood) walking the path `c.d` – first getting the mapping at key `c`, and then the value at `d` in the resulting mapping.

Note that you can have simple keys which look like paths:
```
>>> cfg['f.g']
'h'
```
If a key is given that exists in the configuration, it is used as such, and if it is not present in the configuration, an attempt is made to interpret it as a path. Thus, `f.g` is present and accessed via key, whereas `c.d` is not an existing key, so is interpreted as a path.

Access to `datetime` objects
============================
You can also get native Python `datetime` objects from a configuration, by using an ISO date/time pattern in a `backtick-string`:
```
>>> cfg['christmas_morning']  #  using `2019-12-25 08:39:49`
datetime.datetime(2019, 12, 25, 8, 39, 49)
```

Access to other Python objects
==============================
Access to other Python objects is also possible using the `backtick-string` syntax, provided that they are either environment values or objects contained within importable modules:
```
>>> cfg['error']  # using `sys.stderr`
<_io.TextIOWrapper name='<stderr>' mode='w' encoding='UTF-8'>
>>> cfg['error'] is sys.stderr  # Is it the exact same object?
True
>>> cfg['output']  # using `sys:stdout`
<_io.TextIOWrapper name='<stdout>' mode='w' encoding='UTF-8'>
>>> cfg['output'] is sys.stdout  # Is it the exact same object?
True
```

Access to environment variables
===============================
To access an environment variable, use a `backtick-string` of the form `$VARNAME`:
```
>>> cfg['home'] == os.path.expanduser('~')  # using `$HOME`
True
```
You can specify a default value to be used if an environment variable isn’t present using the `$VARNAME|default-value` form. Whatever string follows the pipe character (including the empty string) is returned if `VARNAME` is not a variable in the environment.
```
>>> cfg['foo']
'bar'
```
For more information, see [the CFG documentation](https://docs.red-dove.com/cfg/index.html).
