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

    def prj = new rocProject('hipFFT-internal', 'PreCheckin')
    // customize for project
    prj.paths.build_command = buildCommand
    prj.libraryDependencies = ['rocFFT-internal']

    // Define test architectures, optional rocm version argument is available
    def nodes = new dockerNodes(nodeDetails, jobName, prj)

    boolean formatCheck = true
    boolean staticAnalysis = true

    buildProject(prj, formatCheck, nodes.dockerArray, null, null, null, staticAnalysis)
}

ci: { 
    String urlJobName = auxiliary.getTopJobName(env.BUILD_URL)

    properties(auxiliary.addCommonProperties([pipelineTriggers([cron('0 1 * * 7')])]))
    stage(urlJobName) {
        runCI([ubuntu18:['any']], urlJobName)
    }
}
