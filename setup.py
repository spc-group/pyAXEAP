from setuptools import setup

setup(
    name='axeap',
    version='0.0.1',
    description='Argonne X-Ray Emmision Analysis Package',
    url='',
    author='Vikram Kashyap',
    author_email='vkashyap@anl.gov',
    license='MIT',
    packages=['axeap',
              'axeap.core',
              'axeap.utils',
              'axeap.monitor',
              'axeap.bluesky'],
    install_requires=['numpy',
                      'pandas',
                      'pillow',
                      'scipy',
                      'scikit-learn',
                      'opencv-contrib-python-headless',
                      'matplotlib',
                      'watchdog'
                      ],

    classifiers=[
        'Development Status :: 1 - Planning',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.8'
    ],
)
