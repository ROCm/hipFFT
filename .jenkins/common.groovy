// This file is for internal AMD use.
// If you are interested in running your own Jenkins, please raise a github issue for assistance.
import static groovy.io.FileType.FILES

def runCompileCommand(platform, project, jobName)
{
    project.paths.construct_build_prefix()

    def getDependenciesCommand = ""
    if (project.installLibraryDependenciesFromCI)
    {
        project.libraryDependencies.each
        { libraryName ->
            getDependenciesCommand += auxiliary.getLibrary(libraryName, platform.jenkinsLabel, 'develop')
        }
    }

    String cmake = platform.jenkinsLabel.contains('centos') ? "cmake3" : "cmake" 
    String hipClang = platform.jenkinsLabel.contains('hipClang') ? "HIP_COMPILER=clang" : ""
    def command = """#!/usr/bin/env bash
                set -x
                
                ls /fftw/lib
                export FFTW_ROOT=/fftw
                export FFTW_INCLUDE_PATH=\${FFTW_ROOT}/include
                export FFTW_LIB_PATH=\${FFTW_ROOT}/lib
                export LD_LIBRARY_PATH=\${FFTW_LIB_PATH}:/opt/rocm/lib:/opt/rocm/hip/lib
                export CPLUS_INCLUDE_PATH=\${FFTW_INCLUDE_PATH}:\${CPLUS_INCLUDE_PATH}
                export CMAKE_PREFIX_PATH=\${FFTW_LIB_PATH}/cmake/fftw3:\${CMAKE_PREFIX_PATH}
                export CMAKE_PREFIX_PATH=\${FFTW_LIB_PATH}/cmake/fftw3f:\${CMAKE_PREFIX_PATH}
                
                cd ${project.paths.project_build_prefix}
                mkdir -p build/release && cd build/release
                ${getDependenciesCommand}
                ${hipClang} ${cmake} ${project.paths.build_command}
            """

    platform.runCommand(this, command)
}

def runTestCommand (platform, project, gfilter)
{
    String sudo = auxiliary.sudo(platform.jenkinsLabel)
    def command = """#!/usr/bin/env bash
                    set -x
                    cd ${project.paths.project_build_prefix}/build/release/clients/staging
                    ${sudo} GTEST_LISTENER=NO_PASS_LINE_IN_LOG ./hipfft-test --gtest_output=xml --gtest_color=yes --gtest_filter=${gfilter}
                """

    platform.runCommand(this, command)
    junit "${project.paths.project_build_prefix}/build/release/clients/staging/*.xml"
}

def runPackageCommand(platform, project, jobName, label='')
{
    def command

    label = label != '' ? '-' + label.toLowerCase() : ''
    ext = platform.jenkinsLabel.contains('ubuntu') ? "deb" : "rpm"
    manager = platform.jenkinsLabel.contains('ubuntu') ? "dpkg -c" : "rpm -qlp"

    command = """
            set -x
            cd ${project.paths.project_build_prefix}/build/release
            make package
            mkdir -p package
            if [ ! -z "$label" ]
            then
                for f in hipfft*.${ext}
                do
                    echo f
                    mv "\$f" "hipfft-internal${label}-\${f#*-}"
                    ls
                done
            fi
            mv *.${ext} package/
            ${manager} package/*.${ext}
        """

    platform.runCommand(this, command)
    platform.archiveArtifacts(this, """${project.paths.project_build_prefix}/build/release/package/*.${ext}""")
}


return this
