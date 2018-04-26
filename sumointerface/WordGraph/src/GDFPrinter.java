import java.util.*;
import java.io.*;


public class GDFPrinter {

	private int _state = 0;
	private PrintWriter writter = null;

	public GDFPrinter(String fileFullPathAndName, String encoding) {
		try {
			writter = new PrintWriter(fileFullPathAndName, encoding);
		} catch(FileNotFoundException | UnsupportedEncodingException e) {
			System.out.println("Path not found ...");
		}		
	}

	public void printGDFHeader(String attrs) {
            switch (_state) {
                case 0:
                    writter.println("nodedef>" + attrs);
                    _state += 1;
                    break;
                case 2:
                    writter.println("edgedef>" + attrs);
                    _state += 1;
                    break;
                default:
                    System.out.println("[WARNING] incorrect state for call");
                    break;
            }
	}

	public void printGDFNodes(List<String> nodeLine) {
		if(_state == 1) {
			for(String line : nodeLine) {
				writter.println(line);
			}
			_state += 1;
		} else {
			System.out.println("[WARNING] incorrect state for call");
		}
	}

	public void printGDFEdges(List<String> edgeLine) {
		if(_state == 3) {
			for(String line : edgeLine) {
				writter.println(line);
			}
			writter.close();
		} else {
			System.out.println("[WARNING] incorrect state for call");
		}
	}

}