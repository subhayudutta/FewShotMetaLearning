from setuptools import setup, find_packages

setup(
    name='meta_visionary',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'tensorflow',
        'numpy',
        'scikit-learn'
    ],
    entry_points={
        'console_scripts': [
            'meta_visionary=meta_visionary.few_shot_meta.py:main'
        ]
    },
    author='Subhayu Dutta',
    description='Few Shot Learning Package',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/subhayudutta/FewShotLearning',
)
