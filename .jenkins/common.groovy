// This file is for internal AMD use.
// If you are interested in running your own Jenkins, please raise a github issue for assistance.
import static groovy.io.FileType.FILES

def runCompileCommand(platform, project, jobName, boolean sameOrg = false)
{
    project.paths.construct_build_prefix()

    def getDependenciesCommand = ""
    if (project.installLibraryDependenciesFromCI)
    {
        project.libraryDependencies.each
        { libraryName ->
            getDependenciesCommand += auxiliary.getLibrary(libraryName, platform.jenkinsLabel, 'develop', sameOrg)
        }
    }

    String cmake = platform.jenkinsLabel.contains('centos') ? "cmake3" : "cmake" 
    String hipClang = platform.jenkinsLabel.contains('hipClang') ? "HIP_COMPILER=clang" : ""
    String path = platform.jenkinsLabel.contains('centos7') ? "export PATH=/opt/rh/devtoolset-7/root/usr/bin:$PATH" : ":"
    String dir = jobName.contains('Debug') ? "debug" : "release"

    def command = """#!/usr/bin/env bash
                set -x
                
                ls /fftw/lib
                export FFTW_ROOT=/fftw
                export FFTW_INCLUDE_PATH=\${FFTW_ROOT}/include
                export FFTW_LIB_PATH=\${FFTW_ROOT}/lib
                export LD_LIBRARY_PATH=\${FFTW_LIB_PATH}:/opt/rocm/lib
                export CPLUS_INCLUDE_PATH=\${FFTW_INCLUDE_PATH}:\${CPLUS_INCLUDE_PATH}
                export CMAKE_PREFIX_PATH=\${FFTW_LIB_PATH}/cmake/fftw3:\${CMAKE_PREFIX_PATH}
                export CMAKE_PREFIX_PATH=\${FFTW_LIB_PATH}/cmake/fftw3f:\${CMAKE_PREFIX_PATH}
                
                cd ${project.paths.project_build_prefix}
                mkdir -p build/${dir} && cd build/${dir}
                ${getDependenciesCommand}
                ${path}
                ${hipClang} ${cmake} ${project.paths.build_command}
                make -j\$(nproc)
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
    String ext = platform.jenkinsLabel.contains('ubuntu') ? "deb" : "rpm"
    String dir = jobName.contains('Debug') ? "debug" : "release"

    command = """
            set -x
            cd ${project.paths.project_build_prefix}/build/${dir}
            make package
            mkdir -p package
            for f in hipfft*.$ext
            do 
                mv "\$f" "hipfft${label}-\${f#*-}"
            done
            mv *.${ext} package/
        """

    platform.runCommand(this, command)
    platform.archiveArtifacts(this, """${project.paths.project_build_prefix}/build/${dir}/package/*.${ext}""")
}


return this
