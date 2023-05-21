from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy
from setuptools.command.build_ext import build_ext


# Avoid a gcc warning below:
# cc1plus: warning: command line option ‘-Wstrict-prototypes’ is valid
# for C/ObjC but not for C++
class BuildExtDebug(build_ext):
    def build_extensions(self):
        if '-Wstrict-prototypes' in self.compiler.compiler_so:
            self.compiler.compiler_so.remove('-Wstrict-prototypes')
        super().build_extensions()


extension = cythonize(
    [
        Extension('pyjobgensim',
                  ['pyjobgensim.pyx',
                   'src/JobGenSim.cpp', 'src/dists/distribution.cpp', 'src/utils.cpp',
                   'src/dists/normal_dist.cpp', 'src/dists/pareto_distribution.cpp',
                   'src/dists/exponential_distribution.cpp', 'src/dists/static_dist.cpp', ],
                  language="c++",
                  extra_compile_args=["-std=c++14"]
                  ),
        Extension('pyjobgenfile',
                  ['pyjobgenfile.pyx',
                   'src/JobGenFile.cpp', 'src/utils.cpp'],
                  language="c++",
                  extra_compile_args=["-std=c++14"],
                  include_dirs=[numpy.get_include()]
                  ),
        Extension('pyenv',
                  ['pyenv.pyx',
                   'src/ActionSpace.cpp', 'src/Job.cpp', 'src/LoadBalanceEnv.cpp', 'src/Logger.cpp',
                   'src/ObservationSpace.cpp', 'src/Server.cpp', 'src/TimeLine.cpp', 'src/WallTime.cpp',
                   'src/JobGenSim.cpp', 'src/dists/distribution.cpp', 'src/utils.cpp',
                   'src/dists/normal_dist.cpp', 'src/dists/pareto_distribution.cpp',
                   'src/dists/exponential_distribution.cpp', 'src/dists/static_dist.cpp',
                   'src/pipes/AgentWindowStatsPipe.cpp', 'src/pipes/WindowStatsPipe.cpp',
                   'src/pipes/FilePipe.cpp', 'src/pipes/pipe.cpp',
                   'src/pipes/TimeBucketPipe.cpp', 'src/pipes/LoggerSortPipe.cpp', ],
                  language="c++",
                  extra_compile_args=["-std=c++14"],
                  include_dirs=[numpy.get_include()]
                  ),
        Extension('pylogreader',
                  ['pylogreader.pyx',
                   'src/LogSort.cpp'],
                  language="c++",
                  extra_compile_args=["-std=c++14"],
                  include_dirs=[numpy.get_include()]
                  )
    ],
    compiler_directives={'language_level': "3"}
)

setup(
    # Information
    name="pyloadbalance",
    ext_modules=extension,
    cmdclass={'build_ext': BuildExtDebug}
)
