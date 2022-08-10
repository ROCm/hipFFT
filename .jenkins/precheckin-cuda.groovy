#!/usr/bin/env groovy
@Library('rocJenkins@pong') _

import com.amd.project.*
import com.amd.docker.*
import java.nio.file.Path

def runCI =
{
    nodeDetails, jobName, buildCommand, label, runTest ->

    def prj = new rocProject('hipFFT-internal', 'PreCheckin-Cuda')
    // customize for project
    prj.paths.build_command = buildCommand
    prj.timeout.test = 600

    // Define test architectures, optional rocm version argument is available
    def nodes = new dockerNodes(nodeDetails, jobName, prj)

    def commonGroovy

    boolean formatCheck = false

    def compileCommand =
    {
        platform, project->

        project.paths.construct_build_prefix()

        commonGroovy = load "${project.paths.project_src_prefix}/.jenkins/common.groovy"
        commonGroovy.runCompileCommand(platform, project,jobName)
    }

    def testCommand =
    {
        platform, project->

        def gfilter = '-*swap*'
        commonGroovy.runTestCommand(platform, project, gfilter)
    }

    def packageCommand =
    {
        platform, project->

        commonGroovy.runPackageCommand(platform, project, jobName, label)
    }

    buildProject(prj, formatCheck, nodes.dockerArray, compileCommand, runTest ? testCommand : null, packageCommand)
}

def setupCI(urlJobName, jobNameList, buildCommand, runCI, label, runTest)
{
    jobNameList = auxiliary.appendJobNameList(jobNameList)

    jobNameList.each
    {
        jobName, nodeDetails->
        if (urlJobName == jobName)
            stage(label + ' ' + jobName) {
                runCI(nodeDetails, jobName, buildCommand, label, runTest)
            }
    }

    // For url job names that are not listed by the jobNameList i.e. compute-rocm-dkms-no-npi-1901
    if(!jobNameList.keySet().contains(urlJobName))
    {
        properties(auxiliary.addCommonProperties([pipelineTriggers([cron('0 1 * * *')])]))
        stage(label + ' ' + urlJobName) {
            runCI(['ubuntu20-cuda11':['anycuda']], urlJobName, buildCommand, label, runTest)
        }
    }
}

ci: {
    String urlJobName = auxiliary.getTopJobName(env.BUILD_URL)

    def propertyList = []
    propertyList = auxiliary.appendPropertyList(propertyList)

    def jobNameList = [:]
    
    propertyList.each
    {
        jobName, property->
        if (urlJobName == jobName)
            properties(auxiliary.addCommonProperties(property))
    }

    String compilerVar = ' -DCMAKE_CXX_COMPILER='
    String gBuildCommand = ' -DCMAKE_BUILD_TYPE=RelWithDebInfo \
                            -DBUILD_WITH_LIB=CUDA -DHIP_INCLUDE_DIRS=/opt/rocm/hip/include \
                            -DCMAKE_MODULE_PATH="/opt/rocm/lib/cmake/hip;/opt/rocm/hip/cmake;/opt/rocm/share/rocm/cmake" \
                            -DBUILD_CLIENTS=ON -L ../..'
    String boostLibraryDir = ' -DBOOST_LIBRARYDIR=/usr/lib/x86_64-linux-gnu'

    // Run tests on normal g++ build
    setupCI(urlJobName, jobNameList, compilerVar + 'g++' + gBuildCommand, runCI, 'g++', true)
    // Also build with hipcc+CUDA backend, both shared and static lib.
    // Static build allows the hipFFT callback sample to be built.
    // Skip tests since the first build would have already run tests.
    setupCI(urlJobName, jobNameList, compilerVar + 'hipcc' + gBuildCommand + boostLibraryDir, runCI, 'hipcc', false)
    setupCI(urlJobName, jobNameList, compilerVar + 'hipcc' + gBuildCommand + boostLibraryDir + ' -DBUILD_SHARED_LIBS=OFF', runCI, 'hipcc-static', false)
}
