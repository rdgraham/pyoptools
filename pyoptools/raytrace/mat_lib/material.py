"""
Material class definition, and helper functions used to load the
constants of the dispersion formula to be used in the calculation of
the refraction index.

It uses the database from  https://refractiveindex.info
"""

from pkg_resources import resource_filename
from .mat_eq import from_yml, ModelNotImplemented

import sys
import json
from pathlib import Path

class MaterialLibrary:

    def __init__(self, prefix = None):

        self.prefix = prefix
        self._cache = {}

        dp = Path(resource_filename("pyoptools.raytrace.mat_lib", "data"))

        if self.prefix is None:
            self.glass_path = dp/'glass'
            self.compound_path = dp/'compounds'
        else:
            self.glass_path = dp/'glass'/prefix

        with (dp/'aliases.json').open() as af:
            self.ailises = json.load(af)

    def _material_factory(self, name, mat_path):
        "Builds and caches a material given path to yml file"
        mat = from_yml(mat_path)
        self._cache[name] = mat
        return mat

    def __getitem__(self, name: str):

        if name in self._cache:
            return self._cache[name]

        # check in ailises, handling special case of a compond
        if name in self.ailises:
            ailas = self.ailises[name]
            if 'compound' in ailas:
                ap = (self.compound_path /
                      ailas['compound'] /
                      f"{ailas['reference']}.yml")
            else:
                ap = self.glass_path/ailas['library']/f"{ailas['material']}.yml"
            return self._material_factory(name, ap)

        # find in glasses
        matches = list(self.glass_path.glob(f"**/{name}.yml"))
        if len(matches) > 1:
            warning = (f"Multiple matches for glass type {name}. "
                       f"Use one of: ")
            for m in matches:
                warning += f"material.{m.parts[-2]}['{name}'] or "
            raise KeyError(warning[:-4])

        if matches:
            return self._material_factory(name, matches[0])

        # find in compounds with reference as a suffix
        if name.startswith('compound:'):
            _, compound, reference = name.split(':')
            if reference == '':
                # if inspecified, get the first reference data
                cp = list((self.compound_path/compound).glob('*.yml'))[0]
            else:
                cp = self.compound_path/compound/f"{reference}.yml"
            if cp.exists():
                return self._material_factory(name, cp)

        raise KeyError(f"Material {name} not found.")

    def get_from(self, name: str, libs: str):
        """Finds glass type with name located in a manufacturer library

        name : name of the glass to find
        libs : A name of a manufacturer library, e.g. 'schott', or a list
        of space-seperate manufacturer names, in which case they will be
        searched in the order given.

        Raise KeyError if no material found in the given libraries.
        """

        if self.prefix is not None:
            raise Exception('get_from only availabe on base library.')

        for libname in libs.split(' '):
            libname = libname.lower()
            #print(f"checking {libname}")
            if (self.glass_path/libname).is_dir():
                try:
                    return MaterialLibrary(prefix = libname)[name]
                except AttributeError:
                    #print('No library')
                    pass
                except KeyError:
                    #print('Not in library')
                    pass

        raise KeyError(f"Material {name} not found in any of {libs.split()}.")

    def __getattr__(self, name: str):
        # Guard for if instantiated as a sub-module
        if self.prefix is not None:
            raise AttributeError()

        if (self.glass_path/name).is_dir():
            return MaterialLibrary(prefix = name)
        else:
            raise AttributeError()

sys.modules[__name__] = MaterialLibrary()

