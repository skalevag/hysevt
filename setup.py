from setuptools import setup,find_packages

setup(name='hysevt',
      version='0.3',
      description='Analysis of water and sediment pulses in alpine environments.',
      url='https://gitup.uni-potsdam.de/skalevag2/hysevt.git',
      author='Amalie Skålevåg',
      author_email='skalevag2@uni-potsdam.de',
      license='GNU GPLv3',
      packages=find_packages(),
      install_requires=[         
        'pandas',         
        'numpy',
        'matplotlib',
        'scipy',
        'scikit-learn',
        'cycler'
      ],
      zip_safe=False)
