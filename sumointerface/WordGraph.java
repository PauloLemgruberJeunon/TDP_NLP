
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

	public WordGraph(HashMap<String, WordContainer> wordsAndSynsets, String filePathAndName) {

		setWordGraphNewInputs(wordsAndSynsets, filePathAndName);

		_graph = new HashMap<String, WordNode>();

		_nodes = new HashMap<String, WordContainer>();
		_edges = new ArrayList<Edge>();

		WordNet.initOnce();
	}

	public void startWordGraph() { 

		for(String synset : _wordsAndSynsets.keySet()) {
			if(_nodes.containsKey(synset) == false) {
				_nodes.put(synset, new WordContainer(_wordsAndSynsets.get(synset)));
				_graph.put(synset, new WordNode(new WordContainer(_wordsAndSynsets.get(synset))));
				findHypernyms(synset);
			}
		}

		GDFPrinter gdfPrinter = new GDFPrinter(_filePathAndName, "UTF-8");

		gdfPrinter.printGDFHeader("name VARCHAR, fullName VARCHAR, reducedName VARCHAR, verb VARCHAR, color VARCHAR");

		ArrayList<String> tempNodesList = new ArrayList<>();
		for(WordContainer wc : _nodes.values()) {
			tempNodesList.add(wc.getSynset() + "," + wc.getFullWord() + "," + wc.getReducedWord() +
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

		findMostImportantsHypernym();

		System.out.println("----------------------- END2 -------------------\n\n\n");
	}


	private void findHypernyms(String synset) {

		ArrayList<AVPair> tempHypernymsList = new ArrayList<>();
		ArrayList<Double> similarityValues = new ArrayList<>();

		ArrayList<AVPair> currRelations = WordNet.wn.relations.get(synset);
		if(currRelations != null) {
			tempHypernymsList = getHypernymsList(currRelations);

        	if(tempHypernymsList.size() == 0) {
        		return;
        	}

        	AVPair currAVPair = getHighestSimilarityHypernym(tempHypernymsList, synset);

        	WordNode newHypernymNode = null;
        	WordNode currNode = _graph.get(synset);

        	if(_nodes.containsKey(currAVPair.value) == false) {
        		WordContainer newHypernym = new WordContainer(currAVPair.attribute, currAVPair.attribute,
        													  currAVPair.value, "-");
        		_nodes.put(currAVPair.value, newHypernym);

        		newHypernymNode = new WordNode(new WordContainer(newHypernym));
        		_graph.put(currAVPair.value, newHypernymNode);
        		
        		findHypernyms(currAVPair.value);
        	}

        	newHypernymNode = _graph.get(currAVPair.value);

        	currNode.setMyHypernym(newHypernymNode);
        	newHypernymNode.addSonNode(currNode);

        	_edges.add(new Edge(synset, currAVPair.value));

		} else {
			System.out.println("[WARNING] Not able to find any relations to this synset: " + synset);
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


	private AVPair getHighestSimilarityHypernym(ArrayList<AVPair> tempHypernymsList, String synset) {
		Double maxValue = -1.0;
    	int place = -1;
    	String currSynsetWord = WordNet.wn.synsetsToWords.get(synset).get(0);
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

	public void findMostImportantsHypernym() {
		HashMap<String, Integer> sonsCounterHash = new HashMap<String, Integer>();
		int counter[] = new int[1];
		counter[0] = 0;

		for(String synset : _graph.keySet()) {
			countSons(synset, counter, 0);
			sonsCounterHash.put(synset, new Integer(counter[0]));
			//System.out.println(_graph.get(synset).getFullWord() + " | counter = " + counter[0]); 
			counter[0] = 0;
		}

		int maxValue = -10;
		ArrayList<String> mostImportantHypernyms = new ArrayList<String>();
		ArrayList<Integer> mostImportantHypernymsScores = new ArrayList<Integer>();

		for(String synset : sonsCounterHash.keySet()) {
			if(sonsCounterHash.get(synset) >= maxValue) {
				maxValue = sonsCounterHash.get(synset);
			}
		}

		for(int i = 4; i > 2; i--) {
			for(String synset : sonsCounterHash.keySet()) {
				if(i == sonsCounterHash.get(synset)) {
					mostImportantHypernyms.add(synset);
					System.out.println(_graph.get(synset).getFullWord() + " | " + i);
					mostImportantHypernymsScores.add(i);
				}
			}
		}

		PrintWriter writter = null;
		try {
			writter = new PrintWriter(_filePathAndName.replace(".gdf", "_mostImportantHypernyms_score_3_4.txt"), "UTF-8");

			int inc = 0;
			for(String synset : mostImportantHypernyms) {
				WordNode currNode = _graph.get(synset);
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

	public void countSons(String synset, int[] counter, int depth) {

		depth++;
		WordNode currNode = _graph.get(synset);

		if(currNode.getFromInterview() == true) {
			counter[0] = counter[0] + 1;
			//System.out.println("counter = " + counter[0]);
			//System.out.println("node full name = " + currNode.getFullWord());
		}

		if (depth >= 4) {
			return;
		}

		ArrayList<WordNode> currNodeSons = currNode.getSonNodes();
		for(int i = 0; i < currNodeSons.size(); i++) {
			countSons(currNodeSons.get(i).getSynset(), counter, depth);
		}
	}
}