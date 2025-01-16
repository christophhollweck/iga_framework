from setuptools import setup, find_packages

setup(
    name="iga_framework",
    version="0.1.0",
    description="A Python framework for Isogeometric Analysis (IGA).",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Christoph Hollweck",
    author_email="hollweck.christoph@gmail.com",
    url="https://github.com/christophhollweck/iga_framework",
    license="MIT",
    packages=find_packages(include=["iga_framework", "iga_framework.*"]),  # Hier explizit das Paket angeben
    install_requires=[
        "numpy",
        "matplotlib",
        "splinepy",
        "shapely",
        "scipy",
        "pyvista",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
)
