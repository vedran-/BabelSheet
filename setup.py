from setuptools import setup, find_packages

setup(
    name="babelsheet",
    version="0.1.0",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        'Click',
        'pandas',
        'google-auth',
        'google-auth-oauthlib',
        'google-auth-httplib2',
        'google-api-python-client',
        'pyyaml',
        'openai',
        'litellm',
        'rich>=13.9.0',
        'PyQt6>=6.6.1',
        'spacy>=3.0.0',
    ],
    entry_points={
        'console_scripts': [
            'babelsheet=babelsheet.src.cli.main:cli',
        ],
    },
) 