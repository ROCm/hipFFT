#!/usr/bin/env groovy
// This file uses an AMD internal Jenkins shared library. Please contact the maintainers if you have wish to enable Jenkins based continuous integration for this library.
@Library('rocJenkins@pong') _

// This is file for internal AMD use.
// If you are interested in running your own Jenkins, please raise a github issue for assistance.

import com.amd.project.*
import com.amd.docker.*
import java.nio.file.Path

def runCI =
{
    nodeDetails, jobName, buildCommand, label ->

    def prj = new rocProject('hipFFT-internal', 'PreCheckin-Cuda')
    // customize for project
    prj.paths.build_command = buildCommand

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

    buildProject(prj, formatCheck, nodes.dockerArray, compileCommand, testCommand, packageCommand)
}

def setupCI(urlJobName, jobNameList, buildCommand, runCI, label)
{
    jobNameList = auxiliary.appendJobNameList(jobNameList)

    jobNameList.each
    {
        jobName, nodeDetails->
        if (urlJobName == jobName)
            stage(label + ' ' + jobName) {
                runCI(nodeDetails, jobName, buildCommand, label)
            }
    }

    // For url job names that are not listed by the jobNameList i.e. compute-rocm-dkms-no-npi-1901
    if(!jobNameList.keySet().contains(urlJobName))
    {
        properties(auxiliary.addCommonProperties([pipelineTriggers([cron('0 1 * * *')])]))
        stage(label + ' ' + urlJobName) {
            runCI(['ubuntu20-cuda11':['anycuda']], urlJobName, buildCommand, label)
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
    
    String gBuildCommand = '-DCMAKE_CXX_COMPILER=g++ -DCMAKE_BUILD_TYPE=RelWithDebInfo \
                            -DBUILD_WITH_LIB=CUDA -DHIP_INCLUDE_DIRS=/opt/rocm/hip/include \
                            -DCMAKE_MODULE_PATH="/opt/rocm/hip/cmake;/opt/rocm/share/rocm/cmake" \
                            -DBUILD_CLIENTS_TESTS=ON -L ../..'

    setupCI(urlJobName, jobNameList, gBuildCommand, runCI, 'g++')
}
