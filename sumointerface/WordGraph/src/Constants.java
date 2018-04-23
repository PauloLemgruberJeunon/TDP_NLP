/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

import java.io.File;
import java.nio.file.Paths;

/**
 *
 * @author paulojeunon
 */
public class Constants {
    public static final String sep = File.separator;
    
    private static final String currLoc = Paths.get(".").toAbsolutePath().normalize().toString();
    private static final String baseLoc = currLoc + sep + ".." + sep + ".." + sep;
    
    public static final String txtFolder = baseLoc + "txtFiles" + sep;
    public static final String xlsxFolder = baseLoc +
            "xlsxFiles" + sep + "generatedXlsxFiles" + sep;
    
    public static final String hypernymGDFFolder =
            baseLoc + "gdfFiles" + sep + "interviewHypernymFiles" + sep;
    
    public static final String hypernymGDFFromDeptsAndStages = hypernymGDFFolder + "departamentStageSpecific" + sep;
    
    public static final String pathToHypernymPathMeasurements = txtFolder + "hypernymPathMeasurements" + sep;
    public static final String pathToHypernymImportanceMeasurements = txtFolder + "mostImportantHypernymsOutputs" + sep;
    public static final String pathToOtherGraphMeasurements = txtFolder + "otherGraphMeasurements" + sep;
}
