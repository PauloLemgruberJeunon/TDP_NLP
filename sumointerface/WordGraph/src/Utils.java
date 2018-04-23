
import java.util.*;
import java.io.*;

import edu.cmu.lti.ws4j.WS4J;

//import org.jfree.chart.ChartFactory;
//import org.jfree.chart.JFreeChart;
//import org.jfree.chart.plot.PlotOrientation;
//import org.jfree.data.category.DefaultCategoryDataset;
//import org.jfree.chart.ChartUtilities;

import org.apache.poi.ss.usermodel.*;
import org.apache.poi.xssf.usermodel.XSSFWorkbook;


public class Utils {


    public static double calcAvgWordSimilarity(String word1, String word2) {
            double wupValue = WS4J.runWUP(word1, word2);
            double jcnValue = WS4J.runJCN(word1, word2);
            double linValue = WS4J.runLIN(word1, word2);
            double lchValue = WS4J.runLCH(word1, word2);

            double averageValue = (wupValue + jcnValue + linValue + lchValue)/4.0;
            if(averageValue > 1.0) {
                averageValue = 1.0;
            } else if(averageValue < 0.01) {
                averageValue = 0.01;
            }

            return averageValue;
    }

    public static String wordCoder(String fullName, String synset) {

        String codedWord = "";
        for(int i = 0; i < fullName.length(); i++) {
            if(i > 3) {
                break;
            }
            codedWord += String.valueOf((int)fullName.charAt(i));
        }

        codedWord += synset;

        return codedWord;
    }

    public static String colorsToDepartment(String color) {
        if(color.equals("#000080")) {
            return "Chemical";
        } else if(color.equals("#ff0000")) {
            return "Civil";
        } else if(color.equals("#228b22")) {
            return "Computational";
        } else if(color.equals("#ffff00")) {
            return "Electrical";
        } else if(color.equals("#ff1493")) {
            return "Materials";
        } else if(color.equals("#8b4513")) {
            return "Mechanical";
        } else if(color.equals("#ffa500")) {
            return "Mining";
        } else if(color.equals("#778899")) {
            return "Petroleum";
        } else {
            return "unknow";
        }
    }

    public static ArrayList<String> getColorList() {
        ArrayList<String> colors = new ArrayList<>();
        colors.add("#000080");
        colors.add("#ff0000");
        colors.add("#228b22");
        colors.add("#ffff00");
        colors.add("#ff1493");
        colors.add("#8b4513");
        colors.add("#ffa500");
        colors.add("#778899");
        return colors;
    }

    public static void printGraph(HashMap<String, WordNode> graph) {
        for(String synset : graph.keySet()) {
            WordNode currNode = graph.get(synset);
            System.out.print("\n");
            System.out.println("full word = " + currNode.getFullWord());
            for(WordNode sonNode : currNode.getSonNodes()) {
                System.out.println("son full word = " + sonNode.getFullWord());
                System.out.println("son from interview = " + sonNode.getFromInterview());
            }
        }
    }

    public static void calculateAvgPathBetweenDeptsAndOthers(HashMap<String, WordNode> graph,
                                                             boolean enableSynset0, String fileToSave) {
        ArrayList<String> colors = getColorList();
        int numberOfDepts = colors.size();
        float[][] pathMatrix = new float[numberOfDepts][numberOfDepts];

        HashMap<String, ArrayList<String>> nodesByColor = new HashMap<String, ArrayList<String>>();
        for(String color : colors) {
            nodesByColor.put(color, new ArrayList<String>());
        }

        for(Map.Entry<String, WordNode> wnEntry : graph.entrySet()) {
            String wnKey = wnEntry.getKey();
            WordNode wn = wnEntry.getValue();
            if(enableSynset0 || wn.getSynset().equals("0") == false) {
                if(wn.getHexColor().equals("#000000") == false) {
                    nodesByColor.get(wn.getHexColor()).add(wnKey);
                }
            }
        }

        int i = 0;
        for(String iColor : colors) {
            ArrayList<String> currColorNodesList = nodesByColor.get(iColor);
            int j = 0;

            for(String jColor : colors) {
                if(j <= i) {
                    j++;
                    continue;
                }

                ArrayList<String> tempColorNodesList = nodesByColor.get(jColor);
                float deptToDeptCounter = 0.0f;

                for(String currWnKey : currColorNodesList) {
                    float singleNodeCounter = 0.0f;

                    for(String tempWnKey : tempColorNodesList) {
                        singleNodeCounter += calculateNodeToNodePathDist(graph, currWnKey, tempWnKey);
                    }
                    singleNodeCounter = singleNodeCounter/tempColorNodesList.size();
                    deptToDeptCounter += singleNodeCounter;
                }

                deptToDeptCounter = deptToDeptCounter/currColorNodesList.size();
                pathMatrix[i][j] = deptToDeptCounter;

                j++;
            }

            i++;
        }

        PrintWriter writter = null;
        try {
                writter = new PrintWriter(fileToSave, "UTF-8");

                for(int k = 0; k < numberOfDepts; k++) {
                        for (int l = k+1; l < numberOfDepts; l++) {
                                writter.println(Utils.colorsToDepartment(colors.get(k)) + " & " + Utils.colorsToDepartment(colors.get(l)) +
                                                " = " + pathMatrix[k][l]);
                        }
                }

                writter.close();
        } catch(Exception e) {
                System.out.println("Path not found ...");
        }
    }


    private static float calculateNodeToNodePathDist(HashMap<String, WordNode> graph, String node1Key, String node2Key) {
        ArrayList<String> node1Hypernyms = new ArrayList<>();
        createHypernymList(graph.get(node1Key), node1Hypernyms);

        ArrayList<String> node2Hypernyms = new ArrayList<>();
        createHypernymList(graph.get(node2Key), node2Hypernyms);

        boolean shouldBreak;
        float distance = 0.0f;
        for(int i = 0; i < node1Hypernyms.size(); i++) {
            shouldBreak = false;
            for(int j = 0; j < node2Hypernyms.size(); j++) {
                if(node1Hypernyms.get(i).equals(node2Hypernyms.get(j))) {
                    distance = i + j + 2;
                    shouldBreak = true;
                    break;
                }
            }
            
            if(shouldBreak) {
                break;
            }
        }

        return distance;
    }


    private static void createHypernymList(WordNode wn, ArrayList<String> nodeHypernyms) {
        if(wn.getMyHypernym() != null) {
            nodeHypernyms.add(Utils.wordCoder(wn.getMyHypernym().getReducedWord(), wn.getMyHypernym().getSynset()));
            createHypernymList(wn.getMyHypernym(), nodeHypernyms);
        }
    }

    public static void findAndSaveVerbNounFrequency(String pathToOutput,
                                                    WordNode root,
                                                    HashMap<String, WordNode> graph) {

        HashMap<String, ArrayList<String>> nodesRelatedToList = new HashMap<>();
        HashMap<String, String> nodeLevels = new HashMap<>();
        getRelatedVerbsForNodes(nodesRelatedToList, nodeLevels, root, 0);

        // Create a Workbook
        Workbook workbook = new XSSFWorkbook();

        // Create a Sheet
        Sheet sheet = workbook.createSheet("Associated VerbNounPairs");

        // Create font
        Font headerFont = workbook.createFont();
        headerFont.setBold(true);
        headerFont.setFontHeightInPoints((short)9);

        // Create cell style
        CellStyle headerCellStyle = workbook.createCellStyle();
        headerCellStyle.setFont(headerFont);
                
        CellStyle separatorCellStyle = workbook.createCellStyle();
        separatorCellStyle.setFillForegroundColor(IndexedColors.GREEN.getIndex());
        separatorCellStyle.setFillPattern(FillPatternType.SOLID_FOREGROUND);

        ArrayList<String> nouns = new ArrayList<>();
        ArrayList<String> verbs = new ArrayList<>();

        int rowTracker = -1;

        for(String wnUniqueCode : nodesRelatedToList.keySet()) {
            ArrayList<String> verbNounPairs = nodesRelatedToList.get(wnUniqueCode);
            String currHypernymName = graph.get(wnUniqueCode).getFullWord();
            String currHypernymVerb = graph.get(wnUniqueCode).getVerb();

            for(String tempWNUniqueCode : verbNounPairs) {
                WordNode tempWN = graph.get(tempWNUniqueCode);
                if(tempWN.getVerb().equals("-")) {
                    continue;
                }
                nouns.add(tempWN.getFullWord());
                verbs.add(tempWN.getVerb());
            }

            if(nouns.size() < 2) {
                nouns.clear();
                verbs.clear();
                continue;
            }

            rowTracker += 1;
            Row row = sheet.createRow(rowTracker);
            Cell header = row.createCell(0);
            header.setCellValue("Current Hypernym:");
            header.setCellStyle(headerCellStyle);
            row.createCell(1).setCellValue(currHypernymName);
            row.createCell(2).setCellValue(currHypernymVerb);

            rowTracker += 1;
            row = sheet.createRow(rowTracker);
            header = row.createCell(0);
            header.setCellValue("Level:");
            header.setCellStyle(headerCellStyle);
            row.createCell(1).setCellValue(nodeLevels.get(wnUniqueCode));
            
            rowTracker += 1;
            row = sheet.createRow(rowTracker);
            header = row.createCell(0);
            header.setCellValue("Qty verb_noun:");
            header.setCellStyle(headerCellStyle);
            Cell qtyOfNounVerb = row.createCell(1);
            qtyOfNounVerb.setCellValue(Integer.toString(nouns.size()));
            qtyOfNounVerb.setCellStyle(headerCellStyle);
            
            rowTracker += 1;
            row = sheet.createRow(rowTracker);
            header = row.createCell(0);
            header.setCellValue("Verbs Related:");
            header.setCellStyle(headerCellStyle);
            for(int i = 0; i < verbs.size(); i++) {
                row.createCell(i+1).setCellValue(verbs.get(i));
            }

            rowTracker += 1;
            row = sheet.createRow(rowTracker);
            header = row.createCell(0);
            header.setCellValue("Nouns Related:");
            header.setCellStyle(headerCellStyle);
            for(int i = 0; i < nouns.size(); i++) {
                row.createCell(i+1).setCellValue(nouns.get(i));
            }

            rowTracker += 2;
            row = sheet.createRow(rowTracker);
            
            for(int i = 0; i < 8; i++) {
                Cell sepCell = row.createCell(i);
                sepCell.setCellStyle(separatorCellStyle);
                sepCell.setCellValue("");
            }
            
            rowTracker += 1;

            nouns.clear();
            verbs.clear();
        }

        try{
            File yourFile = new File(pathToOutput);
            // if file already exists will do nothing
            yourFile.createNewFile(); 
            FileOutputStream fileOut = new FileOutputStream(pathToOutput, false);

            // Write the output to a file
            workbook.write(fileOut);
            fileOut.close();

            // Closing the workbook
            workbook.close();

        } catch(Exception e) {
            System.out.println("\n\nError while saving the frequency count of the verbs \n\n");
        }
    }

    private static void getRelatedVerbsForNodes(HashMap<String, ArrayList<String>> nodesRelatedToList,
                                                    HashMap<String,String> nodeLevels, WordNode root, int depth) {

        if(root.getSonNodes().isEmpty()) {
            return;
        }

        nodesRelatedToList.put(root.getUniqueCode(), new ArrayList<>());
        ArrayList<String> relatedNodesList = nodesRelatedToList.get(root.getUniqueCode());
        nodeLevels.put(root.getUniqueCode(), Integer.toString(depth));

        for(WordNode sonWN : root.getSonNodes()) {
            relatedNodesList.add(sonWN.getUniqueCode());
            getRelatedVerbsForNodes(nodesRelatedToList, nodeLevels, sonWN, depth+1);

            ArrayList<String> sonRelatedList = nodesRelatedToList.get(sonWN.getUniqueCode());
            if(sonRelatedList != null) {
                for(String currWNRelation : sonRelatedList) {
                    relatedNodesList.add(currWNRelation);
                }
            }
        }
    }
    
    public static void getAvgDegreePerAvgDepth(WordNode root, String pathToSave) {
        float[] counters = new float[4];
        for(int i = 0; i < 4; i++) {
            counters[i] = 0.0f;
        } 
        
        getGraphMeasurements(counters, root, 0.0f);
        
        float avgDegree = counters[2]/counters[1];
        float avgDepth = counters[3]/counters[0];
        
        try {
            PrintWriter writer = new PrintWriter(pathToSave, "UTF-8");
            
            writer.println("avgDegree = " + avgDegree);
            writer.println("avgDepth = " + avgDepth);
            writer.println("avgDegree per avgDepth = " + (avgDegree/avgDepth));
            
            writer.close();
            
        } catch(FileNotFoundException | UnsupportedEncodingException e) {
            System.out.println("[WARNING] FILE PATH NOT FOUND IN METHOD \"getAvgDegreePerAvgDepth\"");
        }
        
    }
    
    private static void getGraphMeasurements(float[] counters, WordNode root, float depth) {
        // counters[0] is for leaves cunting
        // counters[1] is for the number of nodes in total
        // counters[2] is for the total number of edges
        // counters[3] is for the sum of all the leaves depth
        
        counters[1] = counters[1] + 1.0f;
        
        if(root.getMyHypernym() != null) {
            counters[2] = counters[2] + 1.0f;
        }
        
        if(root.getSonNodes().isEmpty()) {
            counters[0] = counters[0] + 1.0f;
            counters[3] = counters[3] + depth;
            return;
        }
        
        for(WordNode sonWN : root.getSonNodes()) {
            counters[2] = counters[2] + 1.0f;
            getGraphMeasurements(counters, sonWN, depth+1.0f);
        }
        
    }
}