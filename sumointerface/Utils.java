
import java.util.*;
import java.io.*;

import edu.cmu.lti.ws4j.WS4J;


public class Utils {

	public static double calcAvgWordSimilarity(String word1, String word2) {
		double wupValue = WS4J.runWUP(word1, word2);
		double jcnValue = WS4J.runJCN(word1, word2);
		double linValue = WS4J.runLIN(word1, word2);
		double lchValue = WS4J.runLCH(word1, word2);

		double averageValue = (wupValue + jcnValue + linValue + lchValue)/4.0;
		if(averageValue > 1.0) {
			averageValue = 1.0;
		} else if(averageValue < 0.001) {
			averageValue = 0.001;
		}

		return averageValue;
	}
}