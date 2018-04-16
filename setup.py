# -*- coding: utf-8 -*-
#
#   Copyright (C) 2016 Mateusz Krzysztof Łącki and Michał Startek.
#
#   This file is part of MassTodon.
#
#   MassTodon is free software: you can redistribute it and/or modify
#   it under the terms of the GNU AFFERO GENERAL PUBLIC LICENSE
#   Version 3.
#
#   MassTodon is distributed in the hope that it will be useful,
#   but WITHOUT ANY WARRANTY; without even the implied warranty of
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
#
#   You should have received a copy of the GNU AFFERO GENERAL PUBLIC LICENSE
#   Version 3 along with MassTodon.  If not, see
#   <https://www.gnu.org/licenses/agpl-3.0.en.html>.

from setuptools import setup, find_packages

setup(
    name='rta',
    packages=find_packages(),
    version='0.0.1',
    description='Retention Time Alignment for LC-IM-MS.',
    author=u'Mateusz Krzysztof Łącki',
    author_email='matteo.lacki@gmail.com',
    url='https://github.com/MatteoLacki/rta',
    # download_url='https://github.com/MatteoLacki/MassTodonPy/tree/GutenTag',
    keywords=[
        'Analitical Chemistry',
        'Mass Spectrometry',
        'Retention Time Alignment'],
    classifiers=[
        'Development Status :: 1 - Planning',
        'License :: OSI Approved :: BSD License',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Chemistry',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3.6']
    # install_requires=[],
    # scripts=[
    #     'bin/rta']
)
