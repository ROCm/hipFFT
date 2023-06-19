#!/usr/bin/env groovy
@Library('rocJenkins@pong') _

import com.amd.project.*
import com.amd.docker.*
import java.nio.file.Path

def runCI =
{
    nodeDetails, jobName, buildCommand, label ->

    def prj = new rocProject('hipFFT-internal', 'PreCheckin')
    // customize for project
    prj.paths.build_command = buildCommand
    prj.libraryDependencies = ['rocRAND','rocFFT-internal']

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
        runCI([ubuntu20:['any']], urlJobName)
    }
}
