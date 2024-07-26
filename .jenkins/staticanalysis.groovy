#!/usr/bin/env groovy
@Library('rocJenkins@pong') _

import com.amd.project.*
import com.amd.docker.*
import java.nio.file.Path

def runCI =
{
    nodeDetails, jobName, buildCommand ->

    def prj = new rocProject('hipFFT-internal', 'PreCheckin')
    // customize for project
    prj.paths.build_command = buildCommand
    prj.libraryDependencies = ['rocRAND','rocFFT']

    // Define test architectures, optional rocm version argument is available
    def nodes = new dockerNodes(nodeDetails, jobName, prj)

    boolean formatCheck = true
    boolean staticAnalysis = true

    buildProject(prj, formatCheck, nodes.dockerArray, null, null, null, staticAnalysis)
}

ci: { 
    String urlJobName = auxiliary.getTopJobName(env.BUILD_URL)
    String hipClangBuildCommand = '-DCMAKE_CXX_COMPILER=/opt/rocm/bin/amdclang++ -DCMAKE_BUILD_TYPE=RelWithDebInfo -DBUILD_CLIENTS_TESTS=ON -DBUILD_CLIENTS_SAMPLES=ON -L ../..'

    properties(auxiliary.addCommonProperties([pipelineTriggers([cron('0 1 * * 7')])]))
    stage(urlJobName) {
        runCI([ubuntu22:['any']], urlJobName, hipClangBuildCommand)
    }
}
