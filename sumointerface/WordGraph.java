
import com.articulate.sigma.WordNet;
import com.articulate.sigma.AVPair;
import com.articulate.sigma.WordNetUtilities;
import py4j.GatewayServer;

import java.util.*;

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

	private HashMap<String, WordContainer> _wordsAndSynsets = null;
	private HashMap<String, WordContainer> _nodes = null;
	private ArrayList<Edge> _edges = null;

	public WordGraph() {
		_nodes = new HashMap<String, WordContainer>();
		_edges = new ArrayList<Edge>();

		WordNet.initOnce();
	}

	public void startWordGraph(HashMap<String, WordContainer> wordsAndSynsets, String filePathAdName) {

		_wordsAndSynsets = wordsAndSynsets;
		GDFPrinter gdfPrinter = new GDFPrinter(filePathAdName, "UTF-8"); 

		gdfPrinter.printGDFHeader("name VARCHAR, fullName VARCHAR, reducedName VARCHAR, color VARCHAR");

		for(String synset : wordsAndSynsets.keySet()) {
			if(_nodes.containsKey(synset) == false) {
				_nodes.put(synset, new WordContainer(wordsAndSynsets.get(synset)));
				findHypernyms(synset);
			}
		}

		ArrayList<String> tempNodesList = new ArrayList<>();
		for(WordContainer wc : _nodes.values()) {
			tempNodesList.add(wc.getSynset() + "," + wc.getFullWord() + "," + wc.getReducedWord() +
			                  "," + wc.getHexColor());
		}

		gdfPrinter.printGDFNodes(tempNodesList);

		gdfPrinter.printGDFHeader("node1 VARCHAR, node2 VARCHAR");

		ArrayList<String> tempEdgesList = new ArrayList<>();
		for(Edge e : _edges) {
			tempEdgesList.add(e.getGDFString());
		}

		gdfPrinter.printGDFEdges(tempEdgesList);

		System.out.println("End");
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

        	if(_nodes.containsKey(currAVPair.value) == false) {
        		WordContainer newHypernym = new WordContainer(currAVPair.attribute, currAVPair.attribute,
        													  currAVPair.value);
        		_nodes.put(currAVPair.value, newHypernym);
        		findHypernyms(currAVPair.value);
        	}

        	_edges.add(new Edge(synset, currAVPair.value));

		} else {
			System.out.println("[WARNING] Not able to find any relations to this synset: " + synset);
			System.out.println("Synset meaning: " + WordNet.wn.synsetsToWords.get(synset).get(0));
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


	public static void main(String args[]) {
		GatewayServer gatewayServer = new GatewayServer(new WordGraph());
        gatewayServer.start();
        System.out.println("Gateway Server Started");
	}
}