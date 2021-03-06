# Copyright (C) 2008-today The SG++ project
# This file is part of the SG++ project. For conditions of distribution and
# use, please see the copyright notice provided with SG++ or at
# sgpp.sparsegrids.org

import distutils.sysconfig
import os
import subprocess
import re

import SCons.Script

import Helper

def doConfigure(env, moduleFolders, languageWrapperFolders):
  print
  print "Checking programs and libraries:"

  config = env.Configure(custom_tests={"CheckExec" : Helper.CheckExec,
                                       "CheckJNI" : Helper.CheckJNI,
                                       "CheckFlag" : Helper.CheckFlag,
                                       "CheckCompiler" : Helper.CheckCompiler})

  # now set up all further environment settings that should never fail
  # compiler setup should be always after checking headers and flags,
  # as they can make the checks invalid, e.g., by setting "-Werror"

  if env["OPT"] == True:
    config.env.Append(CPPFLAGS=["-O3", "-g"])

  else:
    config.env.Append(CPPFLAGS=["-g", "-O0"])

  # make settings case-insensitive
  config.env["COMPILER"] = config.env["COMPILER"].lower()
  config.env["ARCH"] = config.env["ARCH"].lower()

  if config.env["COMPILER"] in ("gnu", "openmpi", "mpich"):
    configureGNUCompiler(config)
  elif config.env["COMPILER"] == "clang":
    configureClangCompiler(config)
  elif config.env["COMPILER"] in ("intel", "intel.mpi"):
    configureIntelCompiler(config)
  else:
    Helper.printErrorAndExit("You must specify a valid value for Compiler.",
                             "Available configurations are:",
                             "gnu, clang, intel, openmpi, mpich, intel.mpi")

  if config.env["COMPILER"] in ("openmpi", "mpich", "intel.mpi"):
    config.env["CPPDEFINES"]["USE_MPI"] = "1"
    config.env["USE_MPI"] = True
  else:
    config.env["USE_MPI"] = False

  # special treatment for different platforms
  if config.env["PLATFORM"] == "darwin":
    # the "-undefined dynamic_lookup"-switch is required to actually build a shared library
    # in OSX. "-dynamiclib" alone results in static linking of
    # all further dependent shared libraries
    # beware: if symbols are missing that are actually required
    # (because the symbols don't reside in a shared library),
    # there will be no error during compilation
    # the python binding (pysgpp) requires lpython and a flat namespace
    # also for the python binding, the library must be suffixed with '*.so' even
    # though it is a dynamiclib and not a bundle (see SConscript in src/pysgpp)
    config.env.AppendUnique(LINKFLAGS=["-flat_namespace", "-undefined", "dynamic_lookup", "-lpython"])
    # The GNU assembler (GAS) is not supported in Mac OS X.
    # A solution that fixed this problem is by adding -Wa,-q to the compiler flags.
    # From the man pages for as (version 1.38):
    # -q Use the clang(1) integrated assembler instead of the GNU based system assembler.
    # Note that the CPPFLAG is exactly "-Wa,-q", where -Wa passes flags to the assembler and
    # -q is the relevant flag to make it use integrated assembler
    config.env.AppendUnique(CPPFLAGS=["-Wa,-q"])
    config.env.AppendUnique(CPPPATH="/usr/local/include")
    config.env.AppendUnique(LIBPATH="/usr/local/lib")
    config.env["SHLIBSUFFIX"] = ".dylib"
  elif config.env["PLATFORM"] == "cygwin":
    # required to find the static libraries compiled before the shared libraries
    # the static libraries are required as the linker on windows cannot ignore undefined symbols
    # (as is done on linux automatically and is done on OSX with the settings above)
    #env.Append(LIBPATH=[BUILD_DIR])
    pass

  # will lead to a warning on cygwin and win32
  # ("all code is position independent")
  if config.env["PLATFORM"] not in ["cygwin", "win32"]:
    config.env.AppendUnique(CPPFLAGS=["-fPIC"])

  # setup the include base folder
  # env.Append(CPPPATH=["#/src/sgpp"])
  for moduleFolder in moduleFolders:
    if (moduleFolder in languageWrapperFolders) or (not config.env["SG_" + moduleFolder.upper()]):
      continue
    config.env.AppendUnique(CPPPATH=["#/" + moduleFolder + "/src/"])

  # detour compiler output
  config.env["PRINT_CMD_LINE_FUNC"] = Helper.printCommand

  checkCpp11(config)
  if "doxygen" in SCons.Script.BUILD_TARGETS:
    checkDoxygen(config)
    checkDot(config)
  checkOpenCL(config)
  checkZlib(config)
  checkGSL(config)
  checkDAKOTA(config)
  checkBoostTests(config)
  checkSWIG(config)
  checkPython(config)
  checkJava(config)

  if config.env["USE_CUDA"] == True:
    config.env['CUDA_TOOLKIT_PATH'] = ''
    config.env['CUDA_SDK_PATH'] = ''
    config.env.Tool('cuda')
    # clean up the flags to forward
    # flagsToForward = [flag for flag in config.env["CPPFLAGS"] if flag not in ['-Wmissing-format-attribute', '']]
    # flagsToForward = " -Xcompiler " + (" -Xcompiler ".join(flagsToForward))
    # ensure same flags for host code
    config.env['NVCCFLAGS'] = "-ccbin " + config.env["CXX"] + " -std=c++11 -Xcompiler -fpic,-Wall "# + flagsToForward
    # config.env.AppendUnique(LIBPATH=['/usr/local.nfs/sw/cuda/cuda-7.5/'])

  if config.env["USE_HPX"]:
    hpxLibs = ["dl", "rt", "boost_chrono", "boost_date_time", "boost_filesystem", "boost_program_options", "boost_regex" ,
               "boost_system", "boost_thread", "boost_context", "boost_random", "boost_atomic", "tcmalloc_minimal", "hwloc"]
    if config.env["OPT"]:
      hpxLibs += ["hpx", "hpx_init", "hpx_iostreams"]
      if "HPX_RELEASE_LIBRARY_PATH" in config.env:
        config.env.AppendUnique(LIBPATH=config.env["HPX_RELEASE_LIBRARY_PATH"])
    else:
      if "HPX_DEBUG_LIBRARY_PATH" in env:
        config.env.AppendUnique(LIBPATH=config.env["HPX_DEBUG_LIBRARY_PATH"])
      hpxLibs += ["hpxd", "hpx_initd", "hpx_iostreamsd"]

    for lib in hpxLibs:
      if not config.CheckLib(lib, language="c++", autoadd=1):
        Helper.printErrorAndExit("lib" + lib + " not found, but required for HPX")
    config.env.AppendUnique(LIBS=hpxLibs)
    config.env["CPPDEFINES"]["USE_HPX"] = "1"


    if 'HPX_SHARED_INCLUDE_PATH' in config.env:
      config.env.AppendUnique(CPPPATH=config.env['HPX_SHARED_INCLUDE_PATH'])
    if config.env["OPT"]:
      config.env.AppendUnique(CPPDEFINES=["HPX_APPLICATION_EXPORTS", "HPX_ENABLE_ASSERT_HANDLER"]);
      if 'HPX_RELEASE_INCLUDE_PATH' in config.env:
        config.env.AppendUnique(CPPPATH=config.env['HPX_RELEASE_INCLUDE_PATH'])
    else:
      config.env.AppendUnique(CPPDEFINES=["HPX_DEBUG", "HPX_APPLICATION_EXPORTS", "HPX_ENABLE_ASSERT_HANDLER"]);
      if 'HPX_DEBUG_INCLUDE_PATH' in config.env:
        config.env.AppendUnique(CPPPATH=config.env['HPX_DEBUG_INCLUDE_PATH'])

    if not config.CheckCXXHeader("hpx/hpx_init.hpp"):
      Helper.printErrorAndExit("hpx/hpx_init.hpp not found, but required for HPX")
    if not config.CheckCXXHeader("hpx/include/actions.hpp"):
      Helper.printErrorAndExit("hpx/include/actions.hpp not found, but required for HPX")

  env = config.Finish()

  print "Configuration done."
  print

def checkCpp11(config):
  # check C++11 support
  if not config.env['USE_HPX']:
    if not config.CheckFlag("-std=c++11"):
      Helper.printErrorAndExit("The compiler doesn't seem to support the C++11 standard. Abort!")
      Exit(1)

    config.env.AppendUnique(CPPFLAGS="-std=c++11")
  else:
    if not config.CheckFlag("-std=c++14"):
      Helper.printErrorAndExit("HPX requires a compiler that supports the C++14 standard. Abort!")
      Exit(1)

    config.env.AppendUnique(CPPFLAGS="-std=c++14")

def checkDoxygen(config):
  # check whether Doxygen installed
  if not config.CheckExec("doxygen"):
    Helper.printWarning("Doxygen cannot be found.",
                        "You will not be able to generate the documentation.",
                        "Check the PATH environment variable!")
  else:
    Helper.printInfo("Using Doxygen " + re.findall(
       r"[0-9.]*[0-9]+", subprocess.check_output(["doxygen", "--version"]))[0] + ".")

def checkDot(config):
  # check whether dot installed
  if not config.CheckExec("dot"):
    Helper.printWarning("dot (Graphviz) cannot be found.",
                        "The documentation might lack diagrams.",
                        "Check the PATH environment variable!")
  else:
    Helper.printInfo("Using " +
        subprocess.check_output(["dot", "-V"], stderr=subprocess.STDOUT).strip() + ".")

def checkOpenCL(config):

  # OpenCL also need boost to build
  config.env.AppendUnique(CPPPATH=[config.env["BOOST_INCLUDE_PATH"]])
  config.env.AppendUnique(LIBPATH=[config.env["BOOST_LIBRARY_PATH"]])

  if config.env["USE_OCL"]:
    if "OCL_INCLUDE_PATH" in config.env["ENV"]:
      config.env.AppendUnique(CPPPATH=[config.env["ENV"]["OCL_INCLUDE_PATH"]])
    elif "OCL_INCLUDE_PATH" in config.env:
      config.env.AppendUnique(CPPPATH=[config.env["OCL_INCLUDE_PATH"]])
    else:
      Helper.printInfo("Trying to find the OpenCL header without the OCL_INCLUDE_PATH variable.")

    if not config.CheckCXXHeader("CL/cl.h"):
      Helper.printErrorAndExit("CL/cl.h not found, but required for OpenCL")

    if "OCL_LIBRARY_PATH" in config.env["ENV"]:
      config.env.AppendUnique(LIBPATH=[config.env["ENV"]["OCL_LIBRARY_PATH"]])
    elif "OCL_LIBRARY_PATH" in config.env:
      config.env.AppendUnique(LIBPATH=[config.env["OCL_LIBRARY_PATH"]])
    else:
      Helper.printInfo("Trying to find the libOpenCL library without the OCL_LIBRARY_PATH variable.")

    if not config.CheckLib("OpenCL", language="c++", autoadd=1):
      Helper.printErrorAndExit("libOpenCL not found, but required for OpenCL")

    if not config.CheckLib("boost_program_options", language="c++", autoadd=0):
      Helper.printErrorAndExit("libboost-program-options not found, but required for OpenCL",
                               "On debian-like system the package libboost-program-options-dev",
                               "can be installed to solve this issue.")

    if not config.CheckCXXHeader("zlib.h"):
      Helper.printErrorAndExit("zlib.h not found, but required for OpenCL",
                               "On debian-like system the package zlib1g-dev",
                               "can be installed to solve this issue.")

    if not config.CheckLib("libz", language="c++", autoadd=0):
      Helper.printErrorAndExit("libz not found, but required for OpenCL",
                               "On debian-like system the package zlib1g",
                               "can be installed to solve this issue.")

    config.env["CPPDEFINES"]["USE_OCL"] = "1"

def checkDAKOTA(config):
    if config.env["USE_DAKOTA"]:
        config.env.AppendUnique(CPPPATH=[config.env["DAKOTA_INCLUDE_PATH"]])

        config.env.AppendUnique(LIBPATH=[config.env["DAKOTA_LIBRARY_PATH"]])

        if not config.CheckCXXHeader("pecos_global_defs.hpp"):
            Helper.printErrorAndExit("pecos_global_defs.hpp not found, but required for PECOS")

def checkDAKOTA(config):
    if config.env["USE_DAKOTA"]:
        if not config.CheckCXXHeader("pecos_global_defs.hpp"):
            Helper.printErrorAndExit("pecos_global_defs.hpp not found, but required for PECOS. Consider setting the flag 'CPPPATH'.")

def checkGSL(config):
  if config.env["USE_GSL"]:
    config.env.AppendUnique(CPPPATH=[config.env["GSL_INCLUDE_PATH"]])
    if "GSL_LIBRARY_PATH" in config.env:
	  config.env.AppendUnique(LIBPATH=[config.env["GSL_LIBRARY_PATH"]])

    if not config.CheckCXXHeader("gsl/gsl_version.h"):
      Helper.printErrorAndExit("gsl/gsl_version.h not found, but required for GSL")
    if not config.CheckLib(["gsl", "gslcblas"], language="c++", autoadd=0):
      Helper.printErrorAndExit("libsgl/libgslcblas not found, but required for GSL")

    config.env["CPPDEFINES"]["USE_GSL"] = "1"

def checkZlib(config):
#zlib needed for datamining
    if(config.env["USE_ZLIB"]):
        if config.env["PLATFORM"] == "win32":
            Helper.printWarning("zlib is currently not supported on Windows. Continuing withouth zlib.")
        else:
            if not config.CheckLibWithHeader("z","zlib.h", language="C++",autoadd=0):
                Helper.printErrorAndExit("The flag USE_ZLIB was set, but the necessary header 'zlib.h' or library was not found.")

            config.env["CPPDEFINES"]["ZLIB"] = "1"

def checkBoostTests(config):
  # Check the availability of the boost unit test dependencies
  if config.env["COMPILE_BOOST_TESTS"]:
    config.env.AppendUnique(CPPPATH=[config.env["BOOST_INCLUDE_PATH"]])
    config.env.AppendUnique(LIBPATH=[config.env["BOOST_LIBRARY_PATH"]])

    if not config.CheckHeader(os.path.join("boost", "test", "unit_test.hpp"), language="c++"):
      config.env["COMPILE_BOOST_TESTS"] = False
      Helper.printWarning("****************************************************",
                          "No Boost Unit Test Headers found. Omitting Boost unit tests.",
                          "Please install the corresponding package, e.g., on Ubuntu",
                          "> sudo apt-get install libboost-test-dev",
                          "****************************************************")

    if not config.CheckLib("boost_unit_test_framework", autoadd=0, language="c++"):
      config.env["COMPILE_BOOST_TESTS"] = False
      Helper.printWarning("****************************************************",
                          "No Boost Unit Test library found. Omitting Boost unit tests.",
                          "Please install the corresponding package, e.g., on Ubuntu",
                          "> sudo apt-get install libboost-test-dev",
                          "****************************************************")

def checkSWIG(config):
  if config.env["SG_PYTHON"] or config.env["SG_JAVA"]:
    # check whether swig installed
    if not config.CheckExec("swig"):
      Helper.printErrorAndExit("SWIG cannot be found, but required for SG_PYTHON.",
                               "Check the PATH environment variable!")

    # make sure swig version is new enough
    swigVersion = re.findall(
        r"[0-9.]*[0-9]+", subprocess.check_output(["swig", "-version"]))[0]

    swigVersionTuple = config.env._get_major_minor_revision(swigVersion)
    if swigVersionTuple < (3, 0, 3):
      Helper.printErrorAndExit("SWIG version too old! At least 3.0.3 required.")

    Helper.printInfo("Using SWIG {}".format(swigVersion))

def checkPython(config):
  if config.env["SG_PYTHON"]:
    config.env.AppendUnique(CPPPATH=[distutils.sysconfig.get_python_inc()])
    Helper.printInfo("pythonpath = " + distutils.sysconfig.get_python_inc())

    if not config.CheckCXXHeader("Python.h"):
      Helper.printErrorAndExit("Python.h not found, but required for SG_PYTHON.",
                               "Check path to Python include files:",
                               distutils.sysconfig.get_python_inc(),
                               "Hint: You might have to install the package python-dev.")

    if not config.CheckCXXHeader("pyconfig.h"):
      Helper.printErrorAndExit("pyconfig.h not found, but required for SG_PYTHON.",
                               "Check path to Python include files:",
                               distutils.sysconfig.get_python_inc(),
                               "Hint: You might have to install the package python-dev.")

    try:
      import numpy
      numpy_path = os.path.join(os.path.split(numpy.__file__)[0], "core", "include")
      config.env.AppendUnique(CPPPATH=[numpy_path])
      if not config.CheckCXXHeader(["Python.h", "pyconfig.h", "numpy/arrayobject.h"]):
        Helper.printWarning("Cannot find NumPy header files in " + str(numpy_path) + ".")
        if config.env["RUN_PYTHON_TESTS"]:
          config.env["RUN_PYTHON_TESTS"] = False
          Helper.printWarning("Python unit tests were disabled due to missing numpy development headers.")
    except:
      Helper.printWarning("Warning: Numpy doesn't seem to be installed.")
      if config.env["RUN_PYTHON_TESTS"]:
        Helper.printWarning("Python unit tests were disabled because numpy is not available.")
        config.env["RUN_PYTHON_TESTS"] = False
  else:
    Helper.printInfo("Python extension (SG_PYTHON) not enabled.")

def checkJava(config):
  if config.env["SG_JAVA"]:
    # check for $JAVA_HOME; prepend to search path
    if os.environ.get("JAVA_HOME"):
      config.env.PrependENVPath("PATH", os.path.join(os.environ.get("JAVA_HOME"), "bin"))

    # check whether javac installed
    if not config.CheckExec("javac"):
      Helper.printErrorAndExit("javac cannot be found, but required by SG_JAVA.",
                               "Check the PATH environment variable!")

    # check whether java installed
    if not config.CheckExec("java"):
      Helper.printErrorAndExit("java cannot be found, but required by SG_JAVA.",
                               "Check the PATH environment variable!")

    # check for JNI headers
    if os.environ.get("JNI_CPPINCLUDE"):
      config.env.AppendUnique(CPPPATH=[os.environ.get("JNI_CPPINCLUDE")])
    if not config.CheckCXXHeader("jni.h"):
      # not found; try to find
      if not config.CheckJNI():
        Helper.printErrorAndExit(" jni.h not found."
                                 "Please set JAVA_HOME environment variable"
                                 "with $JAVA_HOME/bin/javac, $JAVA_HOME/include/jni.h"
                                 "or directly $JNI_CPPINCLUDE with $JNI_CPPINCLUDE/jni.h.")
  else:
    Helper.printInfo("Java support (SG_JAVA) not enabled.")

def configureGNUCompiler(config):

  if config.env["RUN_ON_HAZELHEN"]:
    config.env["CC"] = 'CC'
    config.env["CXX"] = 'CC'
    config.env.Append(CPPPATH = [os.environ['BOOST_ROOT'] + '/include'])
    config.env.Append(LIBPATH = [os.environ['BOOST_ROOT'] + '/lib'])
    config.env.Append(CPPFLAGS=["-dynamic"])
    config.env.Append(LINKFLAGS=["-dynamic"])
  if config.env["COMPILER"] == "openmpi":
    config.env["CC"] = ("mpicc")
    config.env["LINK"] = ("mpicxx")
    config.env["CXX"] = ("mpicxx")
    Helper.printInfo("Using openmpi.")
  elif config.env["COMPILER"] == "mpich":
    if config.env["CC"]:
      config.env.Append(CFLAGS=["-cc=" + config.env["CC"]])
    if config.env["CXX"]:
      config.env.Append(CPPFLAGS=["-cxx=" + config.env["CXX"]])
      config.env.Append(LINKFLAGS=["-cxx=" + config.env["CXX"]])
    config.env["CC"] = ("mpicc.mpich")
    config.env["LINK"] = ("mpicxx.mpich")
    config.env["CXX"] = ("mpicxx.mpich")
    Helper.printInfo("Using mpich.")

  versionString = subprocess.check_output([config.env["CXX"], "-dumpversion"]).strip()
  if "." not in versionString:
    versionString = subprocess.check_output([config.env["CXX"], "-dumpfullversion"]).strip()
  version = config.env._get_major_minor_revision(versionString)
  Helper.printInfo("Using {} {}".format(config.env["CXX"], versionString))

  if not config.CheckExec(config.env["CXX"]) or not config.CheckExec(config.env["CC"]) or \
      not config.CheckExec(config.env["LINK"]) :
    Helper.printErrorAndExit("Compiler executable not found!")

  if not config.CheckCompiler():
    Helper.printErrorAndExit("Compiler found, but it is not working! (Hint: check flags)")

  allWarnings = \
      "-Wall -Wextra \
      -Wcast-qual -Wconversion -Wformat=2 \
      -Wformat-nonliteral -Wformat-security -Winit-self  \
      -Wmissing-format-attribute \
      -Wmissing-include-dirs -Wpacked \
      -Wunreachable-code -Wunused \
      -Wno-unused-parameter".split(" ")

  if not config.env['USE_HPX']:
    allWarnings.append(['-Wswitch-enum', '-Wredundant-decls', '-pedantic', '-Wswitch-default'])
  else:
    allWarnings.append(['-Wno-conversion', '-Wno-format-nonliteral'])


  # -fno-strict-aliasing: http://www.swig.org/Doc1.3/Java.html or
  #     http://www.swig.org/Release/CHANGES, 03/02/2006
  #     "If you are going to use optimizations turned on with gcc > 4.0 (for example -O2),
  #     ensure you also compile with -fno-strict-aliasing"
  config.env.Append(CPPFLAGS=allWarnings + [
      "-fno-strict-aliasing",
      "-funroll-loops", "-mfpmath=sse"])
#   if not config.env["USE_HPX"]:
  config.env.Append(CPPFLAGS=["-fopenmp"])
  config.env.Append(LINKFLAGS=["-fopenmp"])

  #   # limit the number of errors display to something reasonable (useful for templated code)
  #   config.env.Append(CPPFLAGS=["-fmax-errors=5"])

  # required for profiling
  config.env.Append(CPPFLAGS=["-fno-omit-frame-pointer"])

  # GCC has support for colored output since 4.9
  if (version >= (4, 9, 0)) and Helper.terminalSupportsColors():
    config.env.Append(CPPFLAGS=["-fdiagnostics-color=always"])

  if config.env["BUILD_STATICLIB"]:
    config.env.Append(CPPFLAGS=["-D_BUILD_STATICLIB"])

  if config.env["ARCH"] == "sse3":
    config.env.AppendUnique(CPPFLAGS=["-msse3"])
  elif config.env["ARCH"] == "sse42":
    config.env.AppendUnique(CPPFLAGS=["-msse4.2"])
  elif config.env["ARCH"] == "avx":
    config.env.AppendUnique(CPPFLAGS=["-mavx"])
  elif config.env["ARCH"] == "fma4":
    config.env.AppendUnique(CPPFLAGS=["-mavx"])
    config.env.AppendUnique(CPPFLAGS=["-mfma4"])
  elif config.env["ARCH"] == "avx2":
    config.env.AppendUnique(CPPFLAGS=["-mavx2"])
    config.env.AppendUnique(CPPFLAGS=["-mfma"])
  elif config.env["ARCH"] == "avx512":
    config.env.AppendUnique(CPPFLAGS=["-mavx512f"])
    config.env.AppendUnique(CPPFLAGS=["-mavx512cd"])
    config.env.AppendUnique(CPPFLAGS=["-mfma"])
  else:
    Helper.printErrorAndExit("You must specify a valid ARCH value for gnu.",
                             "Available configurations are: sse3, sse42, avx, fma4, avx2, avx512")

  # check if using MinGW (g++ on win32)
  if config.env["PLATFORM"] == "win32":
    # disable warnings which occur when including Boost in the tests
    # note that definition of hypot is necessary for to the current version of
    # mingw (6.3) and the python interface (see http://stackoverflow.com/questions/10660524/error-building-boost-1-49-0-with-gcc-4-7-0)
    # -> could be removed in the future hopefully
    config.env.Append(CPPFLAGS=["-Wno-switch-enum", "-Wno-deprecated-declarations", "-D_hypot=hypot"])
    # also use "lib" prefix on MinGW for consistency with Linux (default is no prefix)
    config.env["SHLIBPREFIX"] = "lib"

def configureClangCompiler(config):
  config.env["CC"] = ("clang")
  config.env["LINK"] = ("clang++")
  config.env["CXX"] = ("clang++")

  versionString = subprocess.check_output([config.env["CXX"], "--version"])
  try:
    versionString = re.match(r"^.*version ([^ ]+)", versionString).group(1)
  except AttributeError:
    versionString = "(unknown version)"
  Helper.printInfo("Using {} {}".format(config.env["CXX"], versionString))

  if not config.CheckExec(config.env["CXX"]) or not config.CheckExec(config.env["CC"]) or \
      not config.CheckExec(config.env["LINK"]) :
    Helper.printErrorAndExit("Compiler not found!")

  if not config.CheckCompiler():
    Helper.printErrorAndExit("Compiler found, but it is not working! (Hint: check flags)")

  allWarnings = "-Wall -Wextra -Wno-unused-parameter".split(" ")

  # -fno-strict-aliasing: http://www.swig.org/Doc1.3/Java.html or
  #     http://www.swig.org/Release/CHANGES, 03/02/2006
  #    "If you are going to use optimisations turned on with gcc > 4.0 (for example -O2),
  #     ensure you also compile with -fno-strict-aliasing"
#   if not config.env["USE_HPX"]:
  config.env.Append(CPPFLAGS=["-fopenmp=libiomp5"])
  config.env.Append(LINKFLAGS=["-fopenmp=libiomp5"])

  if config.env["BUILD_STATICLIB"]:
    config.env.Append(CPPFLAGS=["-D_BUILD_STATICLIB"])

  if config.env["ARCH"] == "sse3":
    config.env.AppendUnique(CPPFLAGS=["-msse3"])
  elif config.env["ARCH"] == "sse42":
    config.env.AppendUnique(CPPFLAGS=["-msse4.2"])
  elif config.env["ARCH"] == "avx":
    config.env.AppendUnique(CPPFLAGS=["-mavx"])
  elif config.env["ARCH"] == "fma4":
    config.env.AppendUnique(CPPFLAGS=["-mavx"])
    config.env.AppendUnique(CPPFLAGS=["-mfma4"])
  elif config.env["ARCH"] == "avx2":
    config.env.AppendUnique(CPPFLAGS=["-mavx2"])
    config.env.AppendUnique(CPPFLAGS=["-mfma"])
  elif config.env["ARCH"] == "avx512":
    config.env.AppendUnique(CPPFLAGS=["-mavx512f"])
    config.env.AppendUnique(CPPFLAGS=["-mavx512cd"])
    config.env.AppendUnique(CPPFLAGS=["-mfma"])
  else:
    Helper.printErrorAndExit("You must specify a valid ARCH value for clang.",
                             "Available configurations are: sse3, sse4.2, avx, fma4, avx2, avx512")

def configureIntelCompiler(config):
  config.env.AppendUnique(CPPFLAGS=["-Wall", "-ansi", "-Wno-deprecated", "-wd1125",
                                    "-fno-strict-aliasing",
                                    "-ip", "-ipo", "-funroll-loops",
                                    "-ansi-alias", "-fp-speculation=safe",
                                    "-qno-offload"])
  if config.env["COMPILER"] == "intel.mpi":
    config.env["CC"] = ("mpiicc")
    config.env["LINK"] = ("mpiicpc")
    config.env["CXX"] = ("mpiicpc")
    config.env["CPPDEFINES"]["USE_MPI"] = "1"
  else:
    config.env["CC"] = ("icc")
    config.env["LINK"] = ("icpc")
    config.env["CXX"] = ("icpc")

  versionString = subprocess.check_output([config.env["CXX"], "-dumpversion"]).strip()
  Helper.printInfo("Using {} {}".format(config.env["CXX"], versionString))

  if not config.CheckExec(config.env["CXX"]) or not config.CheckExec(config.env["CC"]) or \
      not config.CheckExec(config.env["LINK"]) :
    Helper.printErrorAndExit("Compiler not found!")

  if not config.CheckCompiler():
    Helper.printErrorAndExit("Compiler found, but it is not working! (Hint: check flags)")

#   if not config.env["USE_HPX"]:
  config.env.AppendUnique(CPPFLAGS=["-qopenmp"])
  config.env.AppendUnique(LINKFLAGS=["-qopenmp"])

  if config.env["BUILD_STATICLIB"]:
    config.env.AppendUnique(CPPFLAGS=["-D_BUILD_STATICLIB"])

  if config.env["ARCH"] == "sse3":
    config.env.AppendUnique(CPPFLAGS=["-msse3"])
  elif config.env["ARCH"] == "sse42":
    config.env.AppendUnique(CPPFLAGS=["-msse4.2"])
  elif config.env["ARCH"] == "avx":
    config.env.AppendUnique(CPPFLAGS=["-mavx"])
  elif config.env["ARCH"] == "avx2":
    config.env.AppendUnique(CPPFLAGS=["-xCORE-AVX2"])
    config.env.AppendUnique(CPPFLAGS=["-fma"])
  elif config.env["ARCH"] == "avx512":
    config.env.AppendUnique(CPPFLAGS=["-xCOMMON-AVX512"])
    config.env.AppendUnique(CPPFLAGS=["-fma"])
  elif config.env["ARCH"] == "mic":
    config.env.AppendUnique(CPPFLAGS=["-mmic"])
    config.env.AppendUnique(LINKFLAGS=["-mmic"])
    config.env["CPPDEFINES"]["USEMIC"] = "1"
  else:
    Helper.printErrorAndExit("You must specify a valid ARCH value for intel.",
                             "Available configurations are: sse3, sse4.2, avx, avx2, avx512, mic")

  config.env.AppendUnique(CPPPATH=[distutils.sysconfig.get_python_inc()])
