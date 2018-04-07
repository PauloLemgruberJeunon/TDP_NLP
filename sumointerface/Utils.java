
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
}