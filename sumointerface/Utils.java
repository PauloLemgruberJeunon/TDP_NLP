
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

	public static void calculateAvgPathBetweenDepts(HashMap<String, WordNode> graph, boolean enableSynset0, String fileToSave) {
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
					System.out.println(wn.getSynset());
				} 
			}			
		}

		System.out.println("\n");

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
		
		if(wn.getMyHypernym() == null) {
			return;
		} else {
			nodeHypernyms.add(Utils.wordCoder(wn.getMyHypernym().getReducedWord(), wn.getMyHypernym().getSynset()));
			createHypernymList(wn.getMyHypernym(), nodeHypernyms);
		}
	}
}