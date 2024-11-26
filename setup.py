from setuptools import setup, find_packages

setup(
    name="babelsheet",
    version="0.1.0",
    packages=find_packages(),
    python_requires=">=3.7",
    entry_points={
        'console_scripts': [
            'babelsheet=babelsheet.__main__:main',
        ],
    },
    install_requires=[
        "google-api-python-client>=2.0.0",
        "google-auth-oauthlib>=0.4.0",
        "pyyaml>=5.1",
    ],
) 