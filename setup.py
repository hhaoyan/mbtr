#!/usr/bin/env python

from setuptools import setup, find_packages, Extension

mbtr_calc = Extension(
    'mbtr.mbtr_imp',
    language='c++',
    sources=['src/mbtr_imp.cpp'],
    extra_compile_args=['-std=c++11']
)

if __name__ == "__main__":
    setup(
        name='mbtr',
        version='0.0.1',
        description='A code for calculating MBTR molecule/crystal structure representation. (arXiv:1704.06439)',
        url='https://github.com/hhaoyan/mbtr',
        author='Haoyan Huo',
        author_email='hhaoyann@gmail.com',
        license='MIT',
        packages=find_packages(),
        ext_modules=[mbtr_calc],
        zip_safe=False,
        classifiers=[
            'Programming Language :: Python :: 2.7',
            'Programming Language :: Python :: 3.6',
            'Development Status :: 4 - Beta',
            'Intended Audience :: Science/Research',
            'Intended Audience :: Information Technology',
            'Operating System :: OS Independent',
            'Topic :: Other/Nonlisted Topic',
            'Topic :: Scientific/Engineering'
        ]
    )
