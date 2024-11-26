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
    ],
    entry_points={
        'console_scripts': [
            'babelsheet=babelsheet.src.cli.main:cli',
        ],
    },
) 