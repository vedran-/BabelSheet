from setuptools import setup, find_packages

setup(
    name="babelsheet",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        'google-auth-oauthlib>=1.0.0',
        'google-api-python-client>=2.0.0',
        'pandas>=2.0.0',
        'pyyaml>=6.0.0',
        'openai>=1.0.0',
        'google-cloud-translate>=3.0.0',
        'click>=8.0.0',
        'aiohttp>=3.8.0'
    ]
) 