from setuptools import setup, find_packages

setup(
    name='rta',
    packages=find_packages(),
    version='0.0.1',
    description='Retention Time Alignment for LC-IM-MS.',
    long_description='The project implements advanced algorithms for the alignement of the retention times in the LC-IM-MS experiments.',
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
        'Programming Language :: Python :: 3.6'],
    install_requires=[ #TODO: trim the unused dependencies.
        'numpy',
        'pandas',
        'scipy',
        'matplotlib',
        'pymysql',
        'sqlalchemy'
    ],
    extras_require={
        'dev': ['ipython', 'sklearn', 'statsmodels']
    }
    # include_package_data=True,
    # package_data={
    #     'data': 'data/annotated_data.csv'
    # }
    # scripts=[
    #     'bin/rta']
)
