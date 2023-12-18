
# Java DeepSTN

Java DeepSTN is an implementation of the DeepSTN plus model in Java, using the Deep Learning for Java (DL4J) library. This project aims to replicate the functionality of the original Python-based DeepSTN, which utilizes Keras, in a Java environment. It is designed to work within IntelliJ IDEA and is managed with Maven for easy dependency resolution and project building.The project is structured to provide a clear and modular approach to crowd flow prediction with deep learning (using CNN).


## Features

- Port of the DeepSTN model to the Java language using DL4J and Nd4j (could be a use case to test deeplearning java framewroks ).
- Maven integration for dependency management and build automation.
- Modular design for ease of maintenance and scalability.
- Configurable setup through external configuration file.


## Project Structure

The project is organized into several packages and resources:

- `src/main/java/com/deepstn`: Root package for the source code.
  - `dataset`: Contains utilities for managing datasets.
  - `main`: Houses the `MainApp` class which serves as the entry point for the application.
  - `model`: Includes the `DeepSTN` class (and or model classes)  that defines the model architecture.
  - `training`: Contains the `DSTNPlusTrainer` class responsible for model training and evaluation.
  - `utils`: Provides helper classes like `ConfigReader` for reading configuration files, `INDArraySlicer`, and `SliceInputPreProcessor` for data preprocessing.

- `src/main/resources`: Resources directory for configuration files and data.
  - `config.properties`: Configuration file containing model parameters and paths for the data files.
  - `flow_data.npy`:  data file (`poi_data.npy` file and/or other data files) .

- `target`: Contains the compiled and packaged JAR file after building the project.

- `pom.xml`: Maven configuration file with project settings and dependencies.

### Description of Key Files

- `MainApp.java`: The main class that initializes the application and starts the model training process.
- `DeepSTN.java`: Defines the architecture of the DeepSTN neural network model.
- `DSTNPlusTrainer.java`: Manages the training process, including the training and evaluation.
- `ConfigReader.java`: Utility class to read and parse the configuration settings from `config.properties`.
- `INDArraySlicer.java`: Helper class to slice NDArrays for batch processing.
- `DataManager.java`:  Manages data operations such as loading, splitting, and pre-processing.


## Installation

### Prerequisites

- Java JDK 1.7 or later
- Maven 3.9.5 or later
- IntelliJ IDEA 

### Steps

1. Ensure that Java JDK and Maven are properly installed on your system. You can verify the installation by running `java -version` and `mvn -version` in your terminal.
2. Clone the repository .
3. Import the project to IntelliJ or another IDE.
4. IntelliJ IDEA should automatically detect the Maven project and begin importing dependencies. If not, you can trigger the process by opening the Maven tab on the right-hand side and clicking 'Reload All Maven Projects'.
5. After all dependencies are resolved, the project should be ready to build.

## Usage

### Running Inside IntelliJ IDEA

1. Navigate to the `src/main/resources` directory in the Project view.
2. Place your data file (e.g., `flow_data.npy`) and configuration file (`config.properties`) inside the resources directory.
3. Run `MainApp.java` from the `src/main/java/com/deepstn/main` directory.

### Building and Running the Executable JAR

1. Open a terminal and navigate to the root directory of the project.
2. To build the project and create a standalone executable JAR file that includes all necessary dependencies, run the Maven command: mvn clean package. The Maven Shade Plugin configured in the pom.xml will be invoked during this process. Once the build is complete, you will find the shaded JAR file in the target directory. This JAR is self-contained and can be run on any system with a compatible Java Runtime Environment.
3. Move your data file and the config.properties file to the same directory where the shaded JAR file is located. This ensures that the application can access the resources it needs.
4. Execute the application by runing the command java -jar target/your-artifactId-version-shaded.jar, replacing your-artifactId-version-shaded.jar with the actual name of your shaded JAR file.


### Configuration

- Edit the `config.properties` file to change the model parameters and paths to the data file as needed.
- The parameters in this file control various aspects of the application, such as parameters and hyperparameters for the model and file paths.

