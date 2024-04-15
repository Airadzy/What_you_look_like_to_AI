import requests
import os
import json
import pandas as pd

def download_first_image(search_term, api_key, cx_id, download_dir="celeb_images"):
    """
    Downloads the first image based on a search term using Google Custom Search API.

    Args:
        search_term (str): The search term to query images for.
        api_key (str): API key for Google Custom Search.
        cx_id (str): Google Custom Search engine ID.
        download_dir (str): Directory to save the downloaded image.

    Returns:
        str: Path to the downloaded image or None if the download fails.
    """
    params = {
        'q': f"{search_term} + face hd",
        'num': 1,
        'start': 1,
        'imgSize': 'large',
        'searchType': 'image',
        'key': api_key,
        'cx': cx_id
    }

    response = requests.get('https://www.googleapis.com/customsearch/v1', params=params)
    if response.status_code == 200:
        response_json = response.json()
        if 'items' in response_json:
            image_url = response_json['items'][0]['link']
            image_response = requests.get(image_url)
            if image_response.status_code == 200:
                output_path = os.path.join(download_dir, search_term.replace(' ', '_') + '.jpg')
                os.makedirs(download_dir, exist_ok=True)
                with open(output_path, 'wb') as file:
                    file.write(image_response.content)
                print(f"Downloaded the first image for search term '{search_term}' successfully.")
                return output_path
            else:
                print("Failed to download the image.")
        else:
            print("No results found.")
    else:
        print(f"Error: {response.status_code}")
    return None


def find_best_match(dataframe,csv_path):
    """
        Generator that yields the image IDs from a CSV file that have the highest number of matching features
    with the given DataFrame's first row, one by one each time it's called.

        Args:
            dataframe (pd.DataFrame): DataFrame containing a row with feature values.
            csv_path (str): Path to the CSV file to compare against.

        Returns:
            int: Image ID of the best matching row.
    """

    # Load the CSV and calculate comparisons just once
    csv_df = pd.read_csv(csv_path)
    feature_row = dataframe.iloc[0]
    comparisons = csv_df.iloc[:, 1:] == feature_row.values
    match_counts = comparisons.sum(axis=1)

    # Create a sorted list of indices from highest to lowest match count
    sorted_indices = match_counts.sort_values(ascending=False).index

    # Yield each image ID one by one
    for index in sorted_indices:
        yield csv_df.iloc[index, 0]




