import py4j.GatewayServer;
import java.util.*;

public class WordGraphHandler {

	private WordGraph _wordGraph;
	private boolean _isInstntiated;

	public WordGraphHandler() {
		_wordGraph = null;
		_isInstntiated = false;
	}

	public WordGraph getWordGraph(HashMap<String, HashMap<String, WordContainer>> wordsAndSynsets, HashMap<String,String> pathDict) {
		if(_isInstntiated) {
			_wordGraph.setWordGraphNewInputs(wordsAndSynsets, pathDict);
			return _wordGraph;
		} else {
			_wordGraph = new WordGraph(wordsAndSynsets, pathDict);
			_isInstntiated = true;
			return _wordGraph;
		}
	}

	public static void main(String[] args) {
		GatewayServer gatewayServer = new GatewayServer(new WordGraphHandler());
        gatewayServer.start();
        System.out.println("\n Gateway Server Started");
	}

	public static String wordCoder(String fullWord, String synset) {
		return Utils.wordCoder(fullWord, synset);
	}
}