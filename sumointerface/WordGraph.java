
import com.articulate.sigma.WordNet;
import com.articulate.sigma.AVPair;
import com.articulate.sigma.WordNetUtilities;
import py4j.GatewayServer;

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


	private String _filePathAndName = null;
	private HashMap<String, HashMap<String, WordContainer>> _wordsAndSynsets = null;

	private String _abstractFullWord = Utils.wordCoder("abstraction", "100002137");
	private String _physicalFullWord = Utils.wordCoder("physical_entity", "100001930");

	public WordGraph(HashMap<String, HashMap<String, WordContainer>> wordsAndSynsets, String filePathAndName) {

		for(HashMap<String, WordContainer> a : wordsAndSynsets.values()) {
			for(WordContainer wc : a.values()) {
				System.out.println(wc.getFullWord());
				System.out.println(wc.getReducedWord());
				System.out.println(wc.getSynset());
				System.out.println(wc.getHexColor());
				System.out.println(wc.getVerb());
				System.out.println("\n\n");
			}
		}

		System.out.println("\n\n");

		setWordGraphNewInputs(wordsAndSynsets, filePathAndName);
		WordNet.initOnce();
	}

	public void wordGraphCreationHandler(boolean doForEachDepartment) {
		HashMap<String, WordNode> graph = new HashMap<>();
		ArrayList<Edge> edges = new ArrayList<>();
		HashMap<String, WordContainer> tempWordContainers = new HashMap<>();

		if(doForEachDepartment) {

			System.out.println("\n\n\n *** STARTING THE CODE TO GET GRAPHS FOR EACH DEPARTMENT *** \n\n\n");

			ArrayList<String> colors = Utils.getColorList();
			for(String color : colors) {
				System.out.println("*** COLOR *** = " + color + "\n\n");
				for(String stage : _wordsAndSynsets.keySet()) {
					System.out.println("* stage * = " + stage + "\n");
					for(String key : _wordsAndSynsets.get(stage).keySet()) {
						WordContainer currContainer = _wordsAndSynsets.get(stage).get(key);
						if(color.equals(currContainer.getHexColor())) {
							System.out.println("Word added");
							WordContainer tempContainer = new WordContainer(currContainer);
							tempContainer.setStage(stage);
							tempWordContainers.put(key, tempContainer);
						}
					}
				}

				String filePathNameAndDept = _filePathAndName.replace(".gdf", "_" + Utils.colorsToDepartment(color) + ".gdf");

				startWordGraph(graph, edges, tempWordContainers, filePathNameAndDept);
				graph.clear();
				edges.clear();
				tempWordContainers.clear();

				System.out.println("\n-----------------------------\n\n");
			}
		}

		System.out.println("\n\n\n *** STARTING THE CODE TO GENERATE THE GRAPHS USING ALL NOUNS *** \n\n\n");

		for(String stage : _wordsAndSynsets.keySet()) {
			System.out.println("*** STAGE *** = " + stage + "\n");
			String filePathNameAndStage = _filePathAndName.replace(".gdf", "_" + stage + ".gdf");
			startWordGraph(graph, edges, _wordsAndSynsets.get(stage), filePathNameAndStage);
			System.out.println("\n-----------------------------\n\n");
		}
	}

	private void startWordGraph(HashMap<String, WordNode> graph, ArrayList<Edge> edges, 
								HashMap<String, WordContainer> wordsAndSynsets, String filePathNameAndStage) {

		for(String fullWordAndSynset : wordsAndSynsets.keySet()) {
			graph.put(fullWordAndSynset, new WordNode(new WordContainer(wordsAndSynsets.get(fullWordAndSynset))));
		}

		for(String fullWordAndSynset : wordsAndSynsets.keySet()) {
				findHypernyms(fullWordAndSynset, graph, edges);
		}

		for(String fullWordAndSynset : graph.keySet()) {
			WordNode currNode = graph.get(fullWordAndSynset);
			if(currNode.getSynset().equals("0")) {

				WordNode currNodeHypernym = null;

				if(currNode.getNatureOfEntity().equals("a")) {
					currNodeHypernym = graph.get(_abstractFullWord);
					edges.add(new Edge(fullWordAndSynset, _abstractFullWord));
				} else {
					currNodeHypernym = graph.get(_physicalFullWord);
					edges.add(new Edge(fullWordAndSynset, _physicalFullWord));
				}

				currNodeHypernym.addSonNode(currNode);
				currNode.setMyHypernym(currNodeHypernym);
			}
		}

		GDFPrinter gdfPrinter = new GDFPrinter(filePathNameAndStage, "UTF-8");

		gdfPrinter.printGDFHeader("name VARCHAR, synset VARCHAR, fullName VARCHAR, reducedName VARCHAR, verb VARCHAR, stage VARCHAR," +
			                      " color VARCHAR");

		ArrayList<String> tempNodesList = new ArrayList<>();
		for(WordNode wn : graph.values()) {

			WordContainer wc = wn.getWordContainer();
			String codedWord = Utils.wordCoder(wc.getReducedWord(), wc.getSynset());

			tempNodesList.add(codedWord + "," + wc.getSynset() + "," + wc.getFullWord() + "," + wc.getReducedWord() +
			                  "," + wc.getVerb() + "," + wc.getStage() + "," + wc.getHexColor());
		}

		gdfPrinter.printGDFNodes(tempNodesList);

		gdfPrinter.printGDFHeader("node1 VARCHAR, node2 VARCHAR");

		ArrayList<String> tempEdgesList = new ArrayList<>();
		for(Edge e : edges) {
			tempEdgesList.add(e.getGDFString());
		}

		gdfPrinter.printGDFEdges(tempEdgesList);

		//for(String synset : graph.keySet()) {
		//	WordNode currNode = graph.get(synset);
		//	System.out.print("\n");
		//	System.out.println("full word = " + currNode.getFullWord());
		//	for(WordNode sonNode : currNode.getSonNodes()) {
		//		System.out.println("son full word = " + sonNode.getFullWord());
		//		System.out.println("son from interview = " + sonNode.getFromInterview());
		//	}
		//}

		findMostImportantHypernyms(graph, filePathNameAndStage);

	}


	private void findHypernyms(String fullWordAndSynset, HashMap<String, WordNode> graph, ArrayList<Edge> edges) {

		ArrayList<AVPair> tempHypernymsList = new ArrayList<>();
		ArrayList<Double> similarityValues = new ArrayList<>();

		ArrayList<AVPair> currRelations = WordNet.wn.relations.get(graph.get(fullWordAndSynset).getSynset());
		if(currRelations != null) {
			tempHypernymsList = getHypernymsList(currRelations);

        	if(tempHypernymsList.size() == 0) {
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

		return;
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


	public void setWordGraphNewInputs(HashMap<String, HashMap<String, WordContainer>> wordsAndSynsets, String filePathAndName) {
		_wordsAndSynsets = wordsAndSynsets;
		_filePathAndName = filePathAndName;
	}

	public void findMostImportantHypernyms(HashMap<String, WordNode> graph, String filePathNameAndStage) {
		HashMap<String, Integer> sonsCounterHash = new HashMap<String, Integer>();
		int counter[] = new int[1];
		counter[0] = 0;

		for(String fullWordAndSynset : graph.keySet()) {
			countSons(fullWordAndSynset, counter, 0, graph);
			sonsCounterHash.put(fullWordAndSynset, new Integer(counter[0]));
			counter[0] = 0;
		}

		int maxValue = -10;
		ArrayList<String> mostImportantHypernyms = new ArrayList<String>();
		ArrayList<Integer> mostImportantHypernymsScores = new ArrayList<Integer>();

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
			writter = new PrintWriter(filePathNameAndStage.replace(".gdf", "_mostImportantHypernyms_all_nodes_below.txt"), "UTF-8");

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