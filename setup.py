from distutils.core import setup
import os

PACKAGE = "spinkyDemo"
NAME = "spinkyDemo"
DESCRIPTION = "Package containing useful function for demonstrating the functionnality of the Spinky MATLAB toobox under Python."
AUTHOR = "Christian O'Reilly"
AUTHOR_EMAIL = "christian.oreilly@epfl.ch"
VERSION = "0.1.0"

def is_package(path):
    return (
        os.path.isdir(path) and
        os.path.isfile(os.path.join(path, '__init__.py'))
        )

def find_packages(path, base="" ):
    """ Find all packages in path """
    packages = {}
    for item in os.listdir(path):
        dir = os.path.realpath(os.path.join(path, item))
        if is_package( dir ):
            if base:
                module_name = "%(base)s.%(item)s" % vars()
            else:
                module_name = item
            packages[module_name] = dir
            packages.update(find_packages(dir, module_name))
    return packages

packages=find_packages(os.path.join(os.path.dirname(os.path.realpath(__file__)), "."))

setup(
    name=NAME,
    packages=packages.keys(),
    package_dir=packages,
    version=VERSION,
    description=DESCRIPTION,
    long_description=DESCRIPTION, #open("README.txt").read(),
    author=AUTHOR,
    author_email=AUTHOR_EMAIL,
    install_requires=["pandas", "numpy", "pymatbridge"],
    maintainer=AUTHOR,
    maintainer_email=AUTHOR_EMAIL,  
    license='LICENSE.txt',
    requires=[],
    classifiers=["Development Status :: 3 - Alpha",
			"Environment :: MacOS X", #"Environment :: Win32 (MS Windows)",
			"Environment :: X11 Applications",
			"Intended Audience :: Developers",
			"Intended Audience :: Science/Research",
			"License :: Free for non-commercial use",
			"License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
			"Natural Language :: English",
			"Programming Language :: Python :: 3.4",
			"Topic :: Scientific/Engineering"])
