# Documentation for Steve: Problem Overview
# Problem Overview


'''
here starts the try catch for the ai
Attempting to explicitly load PyTorch engine...
DJL cache directory set to: C:\Users\Ortwin\.djl.ai
SLF4J(W): No SLF4J providers were found.
SLF4J(W): Defaulting to no-operation (NOP) logger implementation
SLF4J(W): See https://www.slf4j.org/codes.html#noProviders for further details.
ai.djl.engine.EngineException: Failed to load PyTorch native library
	at ai.djl.pytorch.engine.PtEngine.newInstance(PtEngine.java:90)
	at ai.djl.pytorch.engine.PtEngineProvider.getEngine(PtEngineProvider.java:41)
	at org.openwaterfoundation.tstool.plugin.ai.commands.RunAiForecastModel_Command.runCommand(RunAiForecastModel_Command.java:294)
	at rti.tscommandprocessor.core.TSEngine.processCommands(TSEngine.java:2749)
	at rti.tscommandprocessor.core.TSCommandProcessor.runCommands(TSCommandProcessor.java:4609)
	at rti.tscommandprocessor.core.TSCommandProcessor.processRequest_RunCommands(TSCommandProcessor.java:3789)
	at rti.tscommandprocessor.core.TSCommandProcessor.processRequest(TSCommandProcessor.java:2537)
	at rti.tscommandprocessor.core.TSCommandProcessor.processRequest(TSCommandProcessor.java:1996)
	at rti.tscommandprocessor.core.TSCommandProcessorThreadRunner.run(TSCommandProcessorThreadRunner.java:92)
	at java.base/java.lang.Thread.run(Unknown Source)
Caused by: java.lang.AssertionError: No pytorch version found in property file.
	at ai.djl.util.Platform.detectPlatform(Platform.java:92)
	at ai.djl.pytorch.jni.LibUtils.findNativeLibrary(LibUtils.java:306)
	at ai.djl.pytorch.jni.LibUtils.getLibTorch(LibUtils.java:93)
	at ai.djl.pytorch.jni.LibUtils.loadLibrary(LibUtils.java:81)
	at ai.djl.pytorch.engine.PtEngine.newInstance(PtEngine.java:53)
	... 9 more
RTi.Util.IO.CommandException: PyTorch engine failed to initialize: Failed to load PyTorch native library
	at org.openwaterfoundation.tstool.plugin.ai.commands.RunAiForecastModel_Command.runCommand(RunAiForecastModel_Command.java:305)
	at rti.tscommandprocessor.core.TSEngine.processCommands(TSEngine.java:2749)
	at rti.tscommandprocessor.core.TSCommandProcessor.runCommands(TSCommandProcessor.java:4609)
	at rti.tscommandprocessor.core.TSCommandProcessor.processRequest_RunCommands(TSCommandProcessor.java:3789)
	at rti.tscommandprocessor.core.TSCommandProcessor.processRequest(TSCommandProcessor.java:2537)
	at rti.tscommandprocessor.core.TSCommandProcessor.processRequest(TSCommandProcessor.java:1996)
	at rti.tscommandprocessor.core.TSCommandProcessorThreadRunner.run(TSCommandProcessorThreadRunner.java:92)
	at java.base/java.lang.Thread.run(Unknown Source)
'''


It doesnt find the Pytorch native library or the Model engine.
I tried multiple things to fix this, but nothing worked.

In my test project it just worked with this.

'''
<project xmlns="http://maven.apache.org/POM/4.0.0" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
  xsi:schemaLocation="http://maven.apache.org/POM/4.0.0 http://maven.apache.org/maven-v4_0_0.xsd">
  <modelVersion>4.0.0</modelVersion>
  <groupId>com.example.djl</groupId>
  <artifactId>djl-test</artifactId>
  <packaging>jar</packaging>
  <version>1.0-SNAPSHOT</version>
  <name>djl-test</name>
  <url>http://maven.apache.org</url>
 <dependencies>
    <dependency>
        <groupId>ai.djl</groupId>
        <artifactId>api</artifactId>
        <version>0.33.0</version>
    </dependency>
    <dependency>
        <groupId>ai.djl.pytorch</groupId>
        <artifactId>pytorch-engine</artifactId>
        <version>0.33.0</version>
    </dependency>
</dependencies>
<build>
  <plugins>
    <plugin>
      <groupId>org.codehaus.mojo</groupId>
      <artifactId>exec-maven-plugin</artifactId>
      <version>3.5.1</version>
      <configuration>
        <mainClass>com.example.djl.DJLTest</mainClass>
      </configuration>
    </plugin>
  </plugins>
</build>
</project>

'''
'''

package com.example.djl;

import ai.djl.engine.Engine;

public class DJLTest {
    public static void main(String[] args) {
        System.out.println("Engine: " + Engine.getInstance().getEngineName());
    }
}

'''

'''

[INFO]
[INFO] --- exec:3.5.1:java (default-cli) @ djl-test ---
SLF4J(W): No SLF4J providers were found.
SLF4J(W): Defaulting to no-operation (NOP) logger implementation
SLF4J(W): See https://www.slf4j.org/codes.html#noProviders for further details.
Engine: PyTorch
[INFO] ------------------------------------------------------------------------
[INFO] BUILD SUCCESS
[INFO] ------------------------------------------------------------------------
[INFO] Total time:  7.013 s
[INFO] Finished at: 2025-07-24T13:15:27-06:00
[INFO] ----------------------------------------------

'''