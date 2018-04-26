import java.util.*;

public class WordNode {
	private WordContainer _word = null;
	private ArrayList<WordNode> _sonNodes = null;
	private WordNode _myHypernym = null;
	private boolean _fromInterview = true;
	private String _uniqueCode = null;

	public WordNode(WordContainer nodeWord) {
		_word = nodeWord;
		_fromInterview = !(nodeWord.getHexColor().equals("#000000"));
		_uniqueCode = Utils.wordCoder(_word.getReducedWord(), _word.getSynset());
		_sonNodes = new ArrayList<WordNode>();
	}

	public void setMyHypernym(WordNode hypernym) {
		_myHypernym = hypernym;
	}

	public void addSonNode(WordNode sonNode) {
		_sonNodes.add(sonNode);
	}

	public boolean getFromInterview() {
		return _fromInterview;
	}

	public ArrayList<WordNode> getSonNodes() {
		return _sonNodes;
	}

	public WordNode getMyHypernym() {
		return _myHypernym;
	}

	public WordContainer getWordContainer() {
		return _word;
	}

	public String getUniqueCode() {
		return _uniqueCode;
	}

	public String getSynset() {
		return _word.getSynset();
	}

	public String getHexColor() {
		return _word.getHexColor();
	}

	public String getFullWord() {
		return _word.getFullWord();
	}

	public String getReducedWord() {
		return _word.getReducedWord();
	}

	public String getNatureOfEntity() {
		return _word.getNatureOfEntity();
	}

	public String getVerb() {
		return _word.getVerb();
	}
} 