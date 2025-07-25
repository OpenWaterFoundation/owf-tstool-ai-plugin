// RunAiForecast_Command - this class initializes, checks, and runs the RunAiForecastModelt() command.

/* NoticeStart

Open Water Foundation AI Plugin
Copyright (C) 2025 Open Water Foundation

OWER TSTool AI Plugin is free software:  you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

OWER TSTool AI Plugin is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

You should have received a copy of the GNU General Public License
    along with OWF TSTool AI Plugin.  If not, see <https://www.gnu.org/licenses/>.

NoticeEnd */

// FIXME SAM 2008-06-25 Need to clean up exception handling and command status in runCommand().

package org.openwaterfoundation.tstool.plugin.ai.commands;

import java.io.File;


import ai.djl.Model;
import ai.djl.engine.Engine;
import ai.djl.inference.Predictor;
import ai.djl.ndarray.*;
import ai.djl.ndarray.types.Shape;
import ai.djl.translate.*;
import ai.djl.modality.Input;
import ai.djl.modality.Output;
import ai.djl.ndarray.types.DataType;
import ai.djl.pytorch.engine.PtEngineProvider;
import ai.djl.pytorch.engine.PtEngine;


import java.nio.file.Paths;




import java.util.ArrayList;
import java.util.List;

import javax.swing.JFrame;

import rti.tscommandprocessor.core.TSCommandProcessorUtil;
import RTi.Util.IO.AbstractCommand;
import RTi.Util.IO.Command;
import RTi.Util.IO.CommandException;
import RTi.Util.IO.CommandLogRecord;
import RTi.Util.IO.CommandPhaseType;
import RTi.Util.IO.CommandProcessor;
import RTi.Util.IO.CommandStatusType;
import RTi.Util.IO.CommandStatus;
import RTi.Util.IO.CommandWarningException;
import RTi.Util.IO.FileGenerator;
import RTi.Util.IO.InvalidCommandParameterException;
import RTi.Util.IO.IOUtil;
import RTi.Util.IO.PropList;
import RTi.Util.Message.Message;
import RTi.Util.Message.MessageUtil;

/**
This class initializes, checks, and runs the RunAiForecastModel() command.
*/
public class RunAiForecastModel_Command extends AbstractCommand implements Command, FileGenerator
{

/**
Output file that is created by this command.
*/
private File __OutputFile_File = null;

/**
Constructor.
*/
public RunAiForecastModel_Command () {
	super();
	setCommandName ( "RunAiForecastModel" );
}

/**
Check the command parameter for valid values, combination, etc.
@param parameters The parameters for the command.
@param command_tag an indicator to be used when printing messages, to allow a cross-reference to the original commands.
@param warning_level The warning level to use when printing parse warnings
(recommended is 2 for initialization, and 1 for interactive command editor dialogs).
*/
public void checkCommandParameters ( PropList parameters, String command_tag, int warning_level )
throws InvalidCommandParameterException {
	String InputFile = parameters.getValue ( "InputFile" );
	String OutputFile = parameters.getValue ( "OutputFile" );
	String warning = "";
	String message;

	CommandStatus status = getCommandStatus();
	status.clearLog(CommandPhaseType.INITIALIZATION);
	CommandProcessor processor = getCommandProcessor();

	// The existence of the file to remove is not checked during initialization
	// because files may be created dynamically at runtime.

	if ( (InputFile == null) || (InputFile.length() == 0) ) {
		message = "The input file must be specified.";
		warning += "\n" + message;
		status.addToLog(CommandPhaseType.INITIALIZATION,
			new CommandLogRecord(CommandStatusType.FAILURE,
				message, "Specify the input file."));
	}

	// LocalFile is not required given that output property can be specified.
    if ( (OutputFile != null) && !OutputFile.isEmpty() && (OutputFile.indexOf("${") < 0) ) {
        String working_dir = null;
        try {
            Object o = processor.getPropContents ( "WorkingDir" );
            if ( o != null ) {
                working_dir = (String)o;
            }
        }
        catch ( Exception e ) {
            message = "Error requesting WorkingDir from processor.";
            warning += "\n" + message;
            status.addToLog ( CommandPhaseType.INITIALIZATION,new CommandLogRecord(CommandStatusType.FAILURE,
                message, "Software error - report the problem to support." ) );
        }

        try {
            String adjusted_path = IOUtil.verifyPathForOS(IOUtil.adjustPath (working_dir,
                TSCommandProcessorUtil.expandParameterValue(processor,this,OutputFile)));
            File f = new File ( adjusted_path );
            File f2 = new File ( f.getParent() );
            if ( !f2.exists() ) {
                message = "The local file parent directory does not exist for: \"" + adjusted_path + "\".";
                warning += "\n" + message;
                status.addToLog ( CommandPhaseType.INITIALIZATION,new CommandLogRecord(CommandStatusType.FAILURE,
                    message, "Create the output directory." ) );
            }
            f = null;
            f2 = null;
        }
        catch ( Exception e ) {
            message = "The local file:\n" +
            "    \"" + OutputFile +
            "\"\ncannot be adjusted using the working directory:\n" +
            "    \"" + working_dir + "\".";
            warning += "\n" + message;
            status.addToLog ( CommandPhaseType.INITIALIZATION,new CommandLogRecord(CommandStatusType.FAILURE,
                message, "Verify that local file and working directory paths are compatible." ) );
        }
    }

	// Check for invalid parameters.
	List<String> validList = new ArrayList<>(2);
	validList.add ( "InputFile" );
	validList.add ( "OutputFile" );
	warning = TSCommandProcessorUtil.validateParameterNames ( validList, this, warning );

	if ( warning.length() > 0 ) {
		Message.printWarning ( warning_level,
		MessageUtil.formatMessageTag(command_tag,warning_level),warning );
		throw new InvalidCommandParameterException ( warning );
	}
	status.refreshPhaseSeverity(CommandPhaseType.INITIALIZATION,CommandStatusType.SUCCESS);
}

/**
Edit the command.
@param parent The parent JFrame to which the command dialog will belong.
@return true if the command was edited (e.g., "OK" was pressed), and false if not (e.g., "Cancel" was pressed.
*/
public boolean editCommand ( JFrame parent ) {
	// The command will be modified if changed.
	return (new RunAiForecastModel_JDialog ( parent, this )).ok();
}

/**
Return the list of files that were created by this command.
@return the list of files that were created by this command
*/
public List<File> getGeneratedFileList () {
    List<File> list = new ArrayList<>();
    if ( getOutputFile() != null ) {
        list.add ( getOutputFile() );
    }
    return list;
}

/**
Return the output file generated by this command.  This method is used internally.
@return the output file generated by this command
*/
private File getOutputFile () {
    return __OutputFile_File;
}

// Use base class parseCommand.

/**
Run the command.
@param command_line Command number in sequence.
@exception CommandWarningException Thrown if non-fatal warnings occur (the command could produce some results).
@exception CommandException Thrown if fatal warnings occur (the command could not produce output).
*/
public void runCommand ( int command_number )
throws InvalidCommandParameterException, CommandWarningException, CommandException {
    
    String routine = getClass().getSimpleName() + ".runCommand", message;
    int warning_level = 2;
    int log_level = 3; // Level for non-user messages for log file.
    String command_tag = "" + command_number;
    int warning_count = 0;
    CommandPhaseType commandPhase = CommandPhaseType.RUN;

    // Clear the output file.

    setOutputFile ( null );

	PropList parameters = getCommandParameters();

    CommandProcessor processor = getCommandProcessor();
	CommandStatus status = getCommandStatus();
    Boolean clearStatus = Boolean.TRUE; // Default.
    try {
    	Object o = processor.getPropContents("CommandsShouldClearRunStatus");
    	if ( o != null ) {
    		clearStatus = (Boolean)o;
    	}
    }
    catch ( Exception e ) {
    	// Should not happen.
    }
    if ( clearStatus ) {
		status.clearLog(CommandPhaseType.RUN);

	}
  

    String InputFile = parameters.getValue ( "InputFile" );
	InputFile = TSCommandProcessorUtil.expandParameterValue(processor,this,InputFile);
	String inputFile_full = InputFile;
	File inputFile = null;
	if ( (InputFile != null) && !InputFile.isEmpty() ) {
		inputFile_full = IOUtil.verifyPathForOS(
	        IOUtil.toAbsolutePath(TSCommandProcessorUtil.getWorkingDir(processor),InputFile) );
		inputFile = new File(inputFile_full);
	}

    String OutputFile = parameters.getValue ( "OutputFile" );
    boolean doOutputFile = false;
	if ( (OutputFile != null) && !OutputFile.isEmpty() ) {
		OutputFile = TSCommandProcessorUtil.expandParameterValue(processor,this,OutputFile);
		doOutputFile = true;
	}
	String OutputFile_full = OutputFile;
	if ( (OutputFile != null) && !OutputFile.isEmpty() ) {
		OutputFile_full = IOUtil.verifyPathForOS(
	        IOUtil.toAbsolutePath(TSCommandProcessorUtil.getWorkingDir(processor),OutputFile) );
	}

	if ( warning_count > 0 ) {
		message = "There were " + warning_count + " warnings about command parameters.";
		Message.printWarning ( warning_level,
		MessageUtil.formatMessageTag(command_tag, ++warning_count), routine, message );
		throw new InvalidCommandParameterException ( message );
	}

	// Run the forecast model.
    Message.printStatus(2, routine,"Here starts the try catch for the ai.");
    
    try {
        // Explicitly load the PtEngine class. Its static initializer will register
        // the engine, bypassing the ServiceLoader issues in the plugin environment.
        try {
        	Message.printStatus(2,routine,"Attempting to load the PyTorch engine...");

        	// Set the cache folder:
        	// - currently default to a standard location
       		// - may change this later to be a command parameter to allow customization
        	String cacheDir = null;
            if ( IOUtil.isUNIXMachine() ) {
            	// Use a temporary folder that is unique for TSTool and the user:
            	cacheDir = "/tmp/" + System.getProperty("user.name") + "/TSTool/djl.ai";
            }
            else {
            	// Windows.
            	cacheDir = System.getProperty("user.home") + File.separator
            		+ "AppData" + File.separatorChar
            		+ "Local" + File.separator
            		+ "TSTool" + File.separator
            		+ "djl.ai";
            }
            System.setProperty("ai.djl.repository.zoo.location", cacheDir);
            Message.printStatus(2, routine, "DJL cache directory set to: " + cacheDir);
            
            // Try to explicitly register the PyTorch engine.
            PtEngineProvider provider = new PtEngineProvider();
            Engine engine = provider.getEngine();
            
            if ( engine == null ) {
            	message = "Failed to get PyTorch engine from provider.";
   	   			Message.printWarning(log_level,
   	   				MessageUtil.formatMessageTag( command_tag, ++warning_count),
   	   				routine, message );
   	   			status.addToLog ( commandPhase,
   	   				new CommandLogRecord(CommandStatusType.FAILURE, 
   	   					message, "See the log file for information." ) );
                throw new RuntimeException(message);
            }
            
            Message.printStatus(2,routine,"Successfully loaded PyTorch engine: " + engine.getEngineName() + " (Version: " + engine.getVersion() + ")");
            
        } catch (Exception e) {
            message = "PyTorch engine failed to initialize: " + e.getMessage();
  			Message.printWarning(log_level,
  				MessageUtil.formatMessageTag( command_tag, ++warning_count),
 				routine, message );
   			status.addToLog ( commandPhase,
   				new CommandLogRecord(CommandStatusType.FAILURE, 
   					message, "See the log file for information." ) );
            Message.printWarning(3, routine, e);
            throw new CommandException(message);
        }

        int seq_len = 10;
        int weather_features = 12;

        // Load the model:
        // - the name is the name of the model file without the extension ".pt"
        // - TODO smalers 2025-07-25 this is pretty specific, need to evaluate if this can be generalized more
        //try (Model model = Model.newInstance("water_level_model")) {
        String modelName = inputFile.getName().replace(".pt","");
        try (Model model = Model.newInstance(modelName)) {
        	// Load the model file from its folder:
        	// = the path needs to be a Path object
        	// - this is the parent folder of the model file
            //model.load(Paths.get("C:\\Users\\Ortwin\\cdss-dev\\TSTool\\git-repos\\owf-tstool-ai-plugin\\owf-tstool-ai-plugin\\src\\main\\resources")); 
            model.load(Paths.get(inputFile.getParent())); 

            Translator<NDList, NDArray> translator = new Translator<NDList, NDArray>() {
                @Override
                public NDArray processOutput(TranslatorContext ctx, NDList list) {
                    Message.printStatus(2, routine, "Processing model output.");
                    return list.singletonOrThrow();
                }

                @Override
                public NDList processInput(TranslatorContext ctx, NDList input) {
                    return input;
                }

                @Override
                public Batchifier getBatchifier() {
                    return null;
                }
            };

            try (NDManager manager = NDManager.newBaseManager();
                 Predictor<NDList, NDArray> predictor = model.newPredictor(translator)) {

                NDArray historical = manager.randomNormal(0, 1, new Shape(1, seq_len, weather_features + 1), DataType.FLOAT32);
                NDArray futureWeather = manager.randomNormal(0, 1, new Shape(1, seq_len, weather_features), DataType.FLOAT32);

                NDList input = new NDList(historical, futureWeather);
                NDArray output = predictor.predict(input);

                //System.out.println("Model prediction output:");
                //System.out.println(output);
                Message.printStatus(2,routine,"Model prediction output:");
                Message.printStatus(2,routine,"" + output);
            }
        }
        
        Message.printStatus(2, routine, "AI Model run completed successfully.");
        
    }
    catch ( Exception e ) {
        message = "Unexpected error running the model (" + e.getMessage() + ").";
        Message.printWarning ( warning_level,
        MessageUtil.formatMessageTag(command_tag, ++warning_count),routine, message );
        Message.printWarning ( 3, routine, e );
        status.addToLog(CommandPhaseType.RUN,
            new CommandLogRecord(CommandStatusType.FAILURE,
                message, "See the log file for details. Make sure the output file is not open in other software."));
        throw new CommandException ( message ); // FIX: Use single-argument constructor
    }

    status.refreshPhaseSeverity(CommandPhaseType.RUN,CommandStatusType.SUCCESS);
}

/**
Set the output file that is created by this command.  This is only used internally.
@param file output file that is created by this command
*/
private void setOutputFile ( File file ) {
    __OutputFile_File = file;
}

/**
Return the string representation of the command.
@param parameters to include in the command
@return the string representation of the command
*/
public String toString ( PropList parameters ) {
	String [] parameterOrder = {
    	"InputFile",
    	"OutputFile"
	};
	return this.toString(parameters, parameterOrder);
}

}