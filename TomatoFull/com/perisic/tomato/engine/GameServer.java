package com.perisic.tomato.engine;

import java.awt.image.BufferedImage;
import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.net.MalformedURLException;
import java.net.URL;
import java.util.Base64;

import javax.imageio.ImageIO;

/**
 * Game that interfaces to an external Server to retrieve a game. 
 * A game consists of an image and an integer that denotes the solution of this game. 
 * 
 * @author Marc Conrad
 *
 */
public class GameServer {

	/**
	 * Basic utility method to read string for URL.
	 */

	private static String readUrl(String urlString)  {
		try {
			URL url = new URL(urlString);
			InputStream inputStream = url.openStream();

			// Choose anyone of
			// https://stackoverflow.com/questions/309424/how-do-i-read-convert-an-inputstream-into-a-string-in-java
			// to convert InputStream to String.
			ByteArrayOutputStream result = new ByteArrayOutputStream();
			byte[] buffer = new byte[1024];
			int length;
			while ((length = inputStream.read(buffer)) != -1) {
				result.write(buffer, 0, length);
			}
			return result.toString("UTF-8");
		} catch (Exception e) {
			/* To do: proper exception handling when URL cannot be read. */
			System.out.println("An Error occured: " + e.toString());
			e.printStackTrace();
			return null;
		}

	}

	/**
	 * Retrieves a random game from the web site.
	 * @return a random game or null if a game cannot be found. 
	 */
	public Game getRandomGame() {
		// See http://marconrad.com/uob/tomato for details of usage of the api. 
		
		String tomatoapi = "https://marcconrad.com/uob/tomato/api.php?out=csv&base64=yes";
		String dataraw = readUrl(tomatoapi);
		String[] data = dataraw.split(",");

		byte[] decodeImg = Base64.getDecoder().decode(data[0]);
		ByteArrayInputStream quest = new ByteArrayInputStream(decodeImg);

		int solution = Integer.parseInt(data[1]);

		BufferedImage img = null;
		try {
			img = ImageIO.read(quest);
			return new Game(img, solution);
		} catch (IOException e1) {
			// TODO Add proper exception handling. 
			e1.printStackTrace();
			return null;
		}
	}

}
