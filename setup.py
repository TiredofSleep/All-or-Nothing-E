from setuptools import setup, find_packages

setup(
    name="coherence_router",
    version="0.1.0",
    author="Brayden — 7Site LLC",
    author_email="brayden@7sitellc.com",
    description="Universal signal dynamics classifier. Feed any time series, get physics back.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/7sitellc/coherence-router",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[],  # Zero dependencies. stdlib only.
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering",
        "Topic :: System :: Monitoring",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
    ],
    keywords="dynamics chaos lyapunov entropy coherence monitoring time-series",
)
