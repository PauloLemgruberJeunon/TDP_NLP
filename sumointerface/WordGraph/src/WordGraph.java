
import com.articulate.sigma.WordNet;
import com.articulate.sigma.AVPair;
//import com.articulate.sigma.WordNetUtilities;
//import py4j.GatewayServer;

import java.util.*;
import java.io.*;

public class WordGraph {

    public class Edge {

        private String _synset1 = "";
        private String _synset2 = "";

        public Edge(String synset1, String synset2) {
            _synset1 = synset1;
            _synset2 = synset2;
        }

        public String getSynset1() {
            return _synset1;
        } 

        public String getSynset2() {
            return _synset2;
        }

        public String getGDFString() {
            return (_synset1 + "," + _synset2);
        }
    }


    private HashMap<String, String> _pathDict = null;
    private HashMap<String, HashMap<String, WordContainer>> _wordsAndSynsets = null;

    private final String _entityFullWord = "entity";
    private final String _entitySynset = "100001740";
    private final String _entityKey = Utils.wordCoder(_entityFullWord, _entitySynset);

    private final String _abstractFullWord = "abstraction";
    private final String _abstractSynset = "100002137";
    private final String _abstractKey = Utils.wordCoder(_abstractFullWord, _abstractSynset);

    private final String _physicalFullWord = "physical_entity";
    private final String _physicalSynset = "100001930";
    private final String _physicalKey = Utils.wordCoder(_physicalFullWord, _physicalSynset);

    public WordGraph(HashMap<String, HashMap<String, WordContainer>> wordsAndSynsets, HashMap<String, String> pathDict) {

        setWordGraphNewInputs(wordsAndSynsets, pathDict);
        WordNet.initOnce();
    }

    public void wordGraphCreationHandler(boolean doForEachDepartment) {
        HashMap<String, WordNode> graph = new HashMap<>();
        ArrayList<Edge> edges = new ArrayList<>();
        HashMap<String, WordContainer> tempWordContainers = new HashMap<>();

        if(doForEachDepartment) {

            //System.out.println("\n\n\n *** STARTING THE CODE TO GET GRAPHS FOR EACH DEPARTMENT *** \n\n\n");

            ArrayList<String> colors = Utils.getColorList();
            for(String color : colors) {
                //System.out.println("*** COLOR *** = " + color + "\n\n");
                for(String stage : _wordsAndSynsets.keySet()) {
                    //System.out.println("* stage * = " + stage + "\n");
                    for(String key : _wordsAndSynsets.get(stage).keySet()) {
                        WordContainer currContainer = _wordsAndSynsets.get(stage).get(key);
                        if(color.equals(currContainer.getHexColor())) {
                            WordContainer tempContainer = new WordContainer(currContainer);
                            tempContainer.setStage(stage);
                            tempWordContainers.put(key, tempContainer);
                        }
                    }

                    String filePathNameAndDept = _pathDict.get("path_to_output_gdf_hypernym_departments") +
                                                 "hyperGraph_" + Utils.colorsToDepartment(color) + "_" + stage + ".gdf";
                    startWordGraph(graph, edges, tempWordContainers, filePathNameAndDept);
                    
                    graph.clear();
                    edges.clear();
                    tempWordContainers.clear();

                    //System.out.println("\n-----------------------------\n\n");
                }
            }
        }

        System.out.println("\n\n\n *** STARTING THE CODE TO GENERATE THE GRAPHS USING ALL NOUNS *** \n\n\n");

        for(String stage : _wordsAndSynsets.keySet()) {
            System.out.println("*** STAGE *** = " + stage + "\n");
            
            String filePathNameAndStage = _pathDict.get("path_to_output_gdf_hypernym") + "hyperGraph_" + stage + ".gdf";
            startWordGraph(graph, edges, _wordsAndSynsets.get(stage), filePathNameAndStage);
            
            String address = _pathDict.get("path_to_output_xlsx_hypernym") +
                          "verbFrequency_" + stage + ".xlsx";
            Utils.findAndSaveVerbNounFrequency(address, graph.get(_entityKey), graph);
            
            address = _pathDict.get("path_to_output_txt_hypernym_path_meas") + "hypernymDeptsPathsLength_" + stage + ".txt";
            Utils.calculateAvgPathBetweenDeptsAndOthers(graph, false, address);
            
            address = _pathDict.get("path_to_output_txt_hypernym_important_nodes") +
                      "mostImportantHypernyms_all_nodes_below_" +  stage + ".txt";
            findMostImportantHypernyms(graph, address);
            
            address = _pathDict.get("path_to_output_txt_hypernym_other_meas") + "avgDegreePerAvgDepth_" +
                      stage + ".txt";
            Utils.getAvgDegreePerAvgDepth(graph.get(_entityKey), address);
            
            graph.clear();
            edges.clear();
            System.out.println("\n-----------------------------\n\n");
        }
    }

    private void startWordGraph(HashMap<String, WordNode> graph,
                                ArrayList<Edge> edges, 
                                HashMap<String, WordContainer> wordsAndSynsets,
                                String filePathNameAndStage) {

        for(String fullWordAndSynset : wordsAndSynsets.keySet()) {
            graph.put(fullWordAndSynset, new WordNode(new WordContainer(wordsAndSynsets.get(fullWordAndSynset))));
        }		

        for(String fullWordAndSynset : wordsAndSynsets.keySet()) {
            findHypernyms(fullWordAndSynset, graph, edges);
        }

        if(graph.containsKey(_entityKey) == false) {
            WordContainer toNodeContainer = new WordContainer(_entityFullWord, _entityFullWord, _entitySynset, "-", "-");
            WordNode toBeAddedNode = new WordNode(toNodeContainer);

            graph.put(_entityKey, toBeAddedNode);						
        }

        ArrayList<String> notAddedAbstractKeys = new ArrayList<>();
        ArrayList<String> notAddedPhysicalKeys = new ArrayList<>();

        for(String fullWordAndSynset : graph.keySet()) {
            WordNode currNode = graph.get(fullWordAndSynset);
            if(currNode.getSynset().equals("0")) {

                WordNode currNodeHypernym = null;

                if(currNode.getNatureOfEntity().equals("a")) {
                    currNodeHypernym = graph.get(_abstractKey);

                    if(currNodeHypernym == null) {
                        notAddedAbstractKeys.add(fullWordAndSynset);
                        continue;
                    }
                    edges.add(new Edge(fullWordAndSynset, _abstractKey));
                } else {
                    currNodeHypernym = graph.get(_physicalKey);

                    if(currNodeHypernym == null) {
                        notAddedPhysicalKeys.add(fullWordAndSynset);
                        continue;
                    }
                    edges.add(new Edge(fullWordAndSynset, _physicalKey));
                }

                currNodeHypernym.addSonNode(currNode);
                currNode.setMyHypernym(currNodeHypernym);
            }
        }

        if(notAddedAbstractKeys.size() > 0) {
            WordContainer toNodeContainer = new WordContainer(_abstractFullWord, _abstractFullWord,
                                                                                      _abstractSynset, "-", "-");
            WordNode toBeAddedNode = new WordNode(toNodeContainer);
            graph.put(_abstractKey, toBeAddedNode);

            edges.add(new Edge(_abstractKey, _entityKey));

            for(String key : notAddedAbstractKeys) {
                edges.add(new Edge(key, _abstractKey));

                WordNode currNode = graph.get(key);

                toBeAddedNode.addSonNode(currNode);
                currNode.setMyHypernym(toBeAddedNode);
            }
        } 

        if(notAddedPhysicalKeys.size() > 0) {
            WordContainer toNodeContainer = new WordContainer(_physicalFullWord, _physicalFullWord,
                                                                                      _physicalSynset, "-", "-");
            WordNode toBeAddedNode = new WordNode(toNodeContainer);
            graph.put(_physicalKey, toBeAddedNode);

            edges.add(new Edge(_physicalKey, _entityKey));

            for(String key : notAddedPhysicalKeys) {
                edges.add(new Edge(key, _physicalKey));

                WordNode currNode = graph.get(key);

                toBeAddedNode.addSonNode(currNode);
                currNode.setMyHypernym(toBeAddedNode);
            }
        }

        GDFPrinter gdfPrinter = new GDFPrinter(filePathNameAndStage, "UTF-8");

        gdfPrinter.printGDFHeader("name VARCHAR, synset VARCHAR, fullName VARCHAR, reducedName VARCHAR, verb VARCHAR," +
                                      " stage VARCHAR, color VARCHAR");

        ArrayList<String> tempNodesList = new ArrayList<>();
        for(WordNode wn : graph.values()) {

            WordContainer wc = wn.getWordContainer();
            String codedWord = Utils.wordCoder(wc.getReducedWord(), wc.getSynset());

            tempNodesList.add(codedWord + "," + wc.getSynset() + "," + wc.getFullWord() +
                              "," + wc.getReducedWord() +"," +
                              wc.getVerb() + "," + wc.getStage() + "," +
                              wc.getHexColor());
        }

        gdfPrinter.printGDFNodes(tempNodesList);

        gdfPrinter.printGDFHeader("node1 VARCHAR, node2 VARCHAR");

        ArrayList<String> tempEdgesList = new ArrayList<>();
        for(Edge e : edges) {
            tempEdgesList.add(e.getGDFString());
        }

        gdfPrinter.printGDFEdges(tempEdgesList);

    }


    private void findHypernyms(String fullWordAndSynset, HashMap<String, WordNode> graph, ArrayList<Edge> edges) {

            ArrayList<AVPair> tempHypernymsList = new ArrayList<>();
            ArrayList<Double> similarityValues = new ArrayList<>();

            ArrayList<AVPair> currRelations = WordNet.wn.relations.get(graph.get(fullWordAndSynset).getSynset());
            if(currRelations != null) {
                tempHypernymsList = getHypernymsList(currRelations);

            if(tempHypernymsList.isEmpty()) {
                return;
            }

            AVPair currAVPair = getHighestSimilarityHypernym(tempHypernymsList, fullWordAndSynset, graph);

            WordNode newHypernymNode = null;
            WordNode currNode = graph.get(fullWordAndSynset);

            String codedWord = Utils.wordCoder(currAVPair.attribute, currAVPair.value);

            if(graph.containsKey(codedWord) == false) {
                WordContainer newHypernym = new WordContainer(currAVPair.attribute, currAVPair.attribute,
                                                                                                          currAVPair.value, "-", "-");

                newHypernymNode = new WordNode(new WordContainer(newHypernym));
                graph.put(codedWord, newHypernymNode);

                findHypernyms(codedWord, graph, edges);
            }

            newHypernymNode = graph.get(codedWord);

            currNode.setMyHypernym(newHypernymNode);
            newHypernymNode.addSonNode(currNode);

            edges.add(new Edge(fullWordAndSynset, codedWord));

            } else if(graph.get(fullWordAndSynset).getSynset().equals("0") == false){
                System.out.println("[WARNING] Not able to find any relations to this fullWord: " +
                                   graph.get(fullWordAndSynset).getFullWord());
            }

    }


    private ArrayList<AVPair> getHypernymsList(ArrayList<AVPair> relations) {
        ArrayList<AVPair> tempHypernymsList = new ArrayList<>();

        Iterator<AVPair> relationsIterator = relations.iterator();
        while(relationsIterator.hasNext()) {
            AVPair avp = relationsIterator.next();
            if(avp.attribute.equals("hypernym")) {
                AVPair wordSynset = new AVPair(WordNet.wn.synsetsToWords.get(avp.value).get(0),
                                               avp.value);
                tempHypernymsList.add(wordSynset); 
            }
        }

        return tempHypernymsList;

    }


    private AVPair getHighestSimilarityHypernym(ArrayList<AVPair> tempHypernymsList, String fullWordAndSynset,
                                                                                            HashMap<String, WordNode> graph) {
        Double maxValue = -1.0;
        int place = -1;
        String currSynsetWord = WordNet.wn.synsetsToWords.get(graph.get(fullWordAndSynset).getSynset()).get(0);
        double similarityValue = 0.0;

        for(int i = 0; i < tempHypernymsList.size(); i++) {
            String currAttr = tempHypernymsList.get(i).attribute;

            similarityValue = Utils.calcAvgWordSimilarity(currSynsetWord, currAttr);

            if(maxValue.compareTo(similarityValue) < 0) {
                maxValue = similarityValue;
                place = i;
            }
        }

        return tempHypernymsList.get(place);
    }


    public void setWordGraphNewInputs(HashMap<String, HashMap<String, WordContainer>> wordsAndSynsets,
                                      HashMap<String, String> pathDict) {
        _wordsAndSynsets = wordsAndSynsets;
        _pathDict = pathDict;
    }

    public void findMostImportantHypernyms(HashMap<String, WordNode> graph, String filePathNameAndStage) {
        HashMap<String, Integer> sonsCounterHash = new HashMap<>();
        int counter[] = new int[1];
        counter[0] = 0;

        for(String fullWordAndSynset : graph.keySet()) {
            countSons(fullWordAndSynset, counter, 0, graph);
            sonsCounterHash.put(fullWordAndSynset, new Integer(counter[0]));
            counter[0] = 0;
        }

        int maxValue = -10;
        ArrayList<String> mostImportantHypernyms = new ArrayList<>();
        ArrayList<Integer> mostImportantHypernymsScores = new ArrayList<>();

        for(String fullWordAndSynset : sonsCounterHash.keySet()) {
            if(sonsCounterHash.get(fullWordAndSynset) >= maxValue) {
                maxValue = sonsCounterHash.get(fullWordAndSynset);
            }
        }

        for(int i = maxValue; i > 0; i--) {
            for(String fullWordAndSynset : sonsCounterHash.keySet()) {
                if(i == sonsCounterHash.get(fullWordAndSynset)) {
                    mostImportantHypernyms.add(fullWordAndSynset);
                    mostImportantHypernymsScores.add(i);
                }
            }
        }

        PrintWriter writter = null;
        try {
            writter = new PrintWriter(filePathNameAndStage, "UTF-8");

            int inc = 0;
            for(String fullWordAndSynset : mostImportantHypernyms) {
                WordNode currNode = graph.get(fullWordAndSynset);
                writter.println("Full word = " + currNode.getFullWord());
                writter.println("Reduced word = " + currNode.getReducedWord());
                writter.println("Synset = " + currNode.getSynset());
                writter.println("Score = " + mostImportantHypernymsScores.get(inc));
                writter.println("\n-------------------\n");

                inc++;
            }
            writter.close();
        } catch(Exception e) {
            System.out.println("Path not found ...");
        }
    }

    public void countSons(String fullWordAndSynset, int[] counter, int depth, HashMap<String, WordNode> graph) {

        depth++;
        WordNode currNode = graph.get(fullWordAndSynset);

        if(currNode.getFromInterview() == true) {
            counter[0] = counter[0] + 1;
        }

        if (depth >= 200) {
            return;
        }

        ArrayList<WordNode> currNodeSons = currNode.getSonNodes();
        for(int i = 0; i < currNodeSons.size(); i++) {
            WordNode tempNode = currNodeSons.get(i);
            countSons(Utils.wordCoder(tempNode.getReducedWord(), tempNode.getSynset()), counter, depth, graph);
        }
    }
}