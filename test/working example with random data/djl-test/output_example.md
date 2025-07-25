PS C:\Users\Ortwin\Downloads\testjava\djl-test> mvn clean compile exec:java
[INFO] Scanning for projects...
[INFO] 
[INFO] ----------------------< com.example.djl:djl-test >----------------------
[INFO] Building djl-test 1.0-SNAPSHOT
[INFO]   from pom.xml
[INFO] --------------------------------[ jar ]---------------------------------
[INFO] 
[INFO] --- clean:3.2.0:clean (default-clean) @ djl-test ---
[INFO] Deleting C:\Users\Ortwin\Downloads\testjava\djl-test\target
[INFO] 
[INFO] --- resources:3.3.1:resources (default-resources) @ djl-test ---
[WARNING] Using platform encoding (UTF-8 actually) to copy filtered resources, i.e. build is platform dependent!
[INFO] Copying 1 resource from src\main\resources to target\classes
[INFO] 
[INFO] --- compiler:3.13.0:compile (default-compile) @ djl-test ---
[INFO] Recompiling the module because of changed source code.
[WARNING] File encoding has not been set, using platform encoding UTF-8, i.e. build is platform dependent!
[INFO] Compiling 1 source file with javac [debug target 1.8] to target\classes
[WARNING] Bootstrap Classpath ist nicht zusammen mit -source 8 festgelegt
  Wenn Sie den Bootstrap Classpath nicht festlegen, kann dies zu Klassendateien führen, die auf JDK 8 nicht ausgeführt werden können
    --release 8 wird anstelle von -source 8 -target 1.8 empfohlen, weil dadurch der Bootstrap Classpath automatisch festgelegt wird
[WARNING] Quellwert 8 ist veraltet und wird in einem zukünftigen Release entfernt
[WARNING] Zielwert 8 ist veraltet und wird in einem zukünftigen Release entfernt
[WARNING] Verwenden Sie -Xlint:-options, um Warnungen zu veralteten Optionen zu unterdrücken.
[INFO] 
[INFO] --- exec:3.5.1:java (default-cli) @ djl-test ---
SLF4J(W): No SLF4J providers were found.
SLF4J(W): Defaulting to no-operation (NOP) logger implementation
SLF4J(W): See https://www.slf4j.org/codes.html#noProviders for further details.
ND: (1, 10) cpu() float32
[[-0.5347, -0.4647, -0.5382, -0.5735, -0.602 , -0.6078, -0.6079, -0.6056, -0.6068, -0.6007],
]

[INFO] ------------------------------------------------------------------------
[INFO] BUILD SUCCESS
[INFO] ------------------------------------------------------------------------
[INFO] Total time:  10.476 s
[INFO] Finished at: 2025-07-25T09:48:16-06:00
[INFO] ------------------------------------------------------------------------