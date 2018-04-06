
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
	private HashMap<String, WordContainer> _wordsAndSynsets = null;
	private HashMap<String, WordContainer> _nodes = null;
	private ArrayList<Edge> _edges = null;
	private HashMap<String, WordNode> _graph = null;

	private String _abstractFullWord = wordCoder("abstraction", "100002137");
	private String _physicalFullWord = wordCoder("physical_entity", "100001930");

	public WordGraph(HashMap<String, WordContainer> wordsAndSynsets, String filePathAndName) {

		setWordGraphNewInputs(wordsAndSynsets, filePathAndName);

		_graph = new HashMap<String, WordNode>();

		_nodes = new HashMap<String, WordContainer>();
		_edges = new ArrayList<Edge>();

		WordNet.initOnce();
	}

	public void startWordGraph() { 

		for(String fullWordAndSynset : _wordsAndSynsets.keySet()) {
			if(_nodes.containsKey(fullWordAndSynset) == false) {
				_nodes.put(fullWordAndSynset, new WordContainer(_wordsAndSynsets.get(fullWordAndSynset)));
				_graph.put(fullWordAndSynset, new WordNode(new WordContainer(_wordsAndSynsets.get(fullWordAndSynset))));
				findHypernyms(fullWordAndSynset);
			}
		}

		for(String fullWordAndSynset : _graph.keySet()) {
			WordNode currNode = _graph.get(fullWordAndSynset);
			WordNode currNodeHypernym = null;
			if(currNode.getFullWord().equals("0")) {
				if(currNode.getNatureOfEntity().equals("a")) {
					currNodeHypernym = _graph.get(_abstractFullWord);
					_edges.add(new Edge(fullWordAndSynset, _abstractFullWord));
				} else {
					currNodeHypernym = _graph.get(_physicalFullWord);
					_edges.add(new Edge(fullWordAndSynset, _physicalFullWord));
				}

				currNodeHypernym.addSonNode(currNode);
				currNode.setMyHypernym(currNodeHypernym);
			}
		}

		GDFPrinter gdfPrinter = new GDFPrinter(_filePathAndName, "UTF-8");

		gdfPrinter.printGDFHeader("name VARCHAR, synset VARCHAR, fullName VARCHAR, reducedName VARCHAR, verb VARCHAR, color VARCHAR");

		ArrayList<String> tempNodesList = new ArrayList<>();
		for(WordContainer wc : _nodes.values()) {

			String codedWord = "";
			for(int j = 0; j < wc.getFullWord().length(); j++) {
				codedWord += String.valueOf((int)wc.getFullWord().charAt(j));
			}

			tempNodesList.add(codedWord + wc.getSynset() + "," + wc.getSynset() + "," + wc.getFullWord() + "," + wc.getReducedWord() +
			                  "," + wc.getVerb() + "," + wc.getHexColor());
		}

		gdfPrinter.printGDFNodes(tempNodesList);

		gdfPrinter.printGDFHeader("node1 VARCHAR, node2 VARCHAR");

		ArrayList<String> tempEdgesList = new ArrayList<>();
		for(Edge e : _edges) {
			tempEdgesList.add(e.getGDFString());
		}

		gdfPrinter.printGDFEdges(tempEdgesList);

		System.out.println("----------------------- END1 -------------------\n\n\n");

		//for(String synset : _graph.keySet()) {
		//	WordNode currNode = _graph.get(synset);
		//	System.out.print("\n");
		//	System.out.println("full word = " + currNode.getFullWord());
		//	for(WordNode sonNode : currNode.getSonNodes()) {
		//		System.out.println("son full word = " + sonNode.getFullWord());
		//		System.out.println("son from interview = " + sonNode.getFromInterview());
		//	}
		//}

		findMostImportantHypernyms();

		System.out.println("----------------------- END2 -------------------\n\n\n");
	}


	private void findHypernyms(String fullWordAndSynset) {

		ArrayList<AVPair> tempHypernymsList = new ArrayList<>();
		ArrayList<Double> similarityValues = new ArrayList<>();

		ArrayList<AVPair> currRelations = WordNet.wn.relations.get(_graph.get(fullWordAndSynset).getSynset());
		if(currRelations != null) {
			tempHypernymsList = getHypernymsList(currRelations);

        	if(tempHypernymsList.size() == 0) {
        		return;
        	}

        	AVPair currAVPair = getHighestSimilarityHypernym(tempHypernymsList, fullWordAndSynset);

        	WordNode newHypernymNode = null;
        	WordNode currNode = _graph.get(fullWordAndSynset);

        	System.out.println("value = " + currAVPair.value + " || attr = " + currAVPair.attribute);

        	String codedWord = "";
        	for(int i = 0; i < currAVPair.attribute.length(); i++) {
        		codedWord += String.valueOf((int)currAVPair.attribute.charAt(i));
        	}

        	if(_nodes.containsKey(codedWord + currAVPair.value) == false) {
        		WordContainer newHypernym = new WordContainer(currAVPair.attribute, currAVPair.attribute,
        													  currAVPair.value, "-", "-");
        		_nodes.put(codedWord + currAVPair.value, newHypernym);

        		newHypernymNode = new WordNode(new WordContainer(newHypernym));
        		_graph.put(codedWord + currAVPair.value, newHypernymNode);
        		
        		findHypernyms(codedWord + currAVPair.value);
        	}

        	newHypernymNode = _graph.get(codedWord + currAVPair.value);

        	currNode.setMyHypernym(newHypernymNode);
        	newHypernymNode.addSonNode(currNode);

        	_edges.add(new Edge(fullWordAndSynset, codedWord + currAVPair.value));

		} else if(_graph.get(fullWordAndSynset).getSynset().equals("0") == false){
			System.out.println("[WARNING] Not able to find any relations to this fullWordAndSynset: " + fullWordAndSynset);
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


	private AVPair getHighestSimilarityHypernym(ArrayList<AVPair> tempHypernymsList, String fullWordAndSynset) {
		Double maxValue = -1.0;
    	int place = -1;
    	String currSynsetWord = WordNet.wn.synsetsToWords.get(_graph.get(fullWordAndSynset).getSynset()).get(0);
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


	public void setWordGraphNewInputs(HashMap<String, WordContainer> wordsAndSynsets, String filePathAndName) {
		_wordsAndSynsets = wordsAndSynsets;
		_filePathAndName = filePathAndName;

		_nodes = new HashMap<String, WordContainer>();
		_edges = new ArrayList<Edge>();
	}

	public void findMostImportantHypernyms() {
		HashMap<String, Integer> sonsCounterHash = new HashMap<String, Integer>();
		int counter[] = new int[1];
		counter[0] = 0;

		for(String fullWordAndSynset : _graph.keySet()) {
			countSons(fullWordAndSynset, counter, 0);
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
					//System.out.println(_graph.get(fullWord).getFullWord() + " | " + i);
					mostImportantHypernymsScores.add(i);
				}
			}
		}

		PrintWriter writter = null;
		try {
			writter = new PrintWriter(_filePathAndName.replace(".gdf", "_mostImportantHypernyms_all_nodes_below.txt"), "UTF-8");

			int inc = 0;
			for(String fullWordAndSynset : mostImportantHypernyms) {
				WordNode currNode = _graph.get(fullWordAndSynset);
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

	public void countSons(String fullWordAndSynset, int[] counter, int depth) {

		depth++;
		WordNode currNode = _graph.get(fullWordAndSynset);

		if(currNode.getFromInterview() == true) {
			counter[0] = counter[0] + 1;
			//System.out.println("counter = " + counter[0]);
			//System.out.println("node full name = " + currNode.getFullWord());
		}

		if (depth >= 200) {
			return;
		}

		ArrayList<WordNode> currNodeSons = currNode.getSonNodes();
		for(int i = 0; i < currNodeSons.size(); i++) {
			//System.out.println(currNodeSons.get(i).getFullWord());
			WordNode tempNode = currNodeSons.get(i);

			String codedWord = "";
			for(int j = 0; j < tempNode.getFullWord().length(); j++) {
				codedWord += String.valueOf((int)tempNode.getFullWord().charAt(j));
			}
			countSons(codedWord + tempNode.getSynset(), counter, depth);
		}
	}

	public String wordCoder(String fullName, String synset) {

		String codedWord = "";
		for(int i = 0; i < fullName.length(); i++) {
			codedWord += String.valueOf((int)fullName.charAt(i));
		}

		codedWord += synset;

		return codedWord;
	}
}