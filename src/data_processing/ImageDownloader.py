import os
import requests
import xml.etree.ElementTree as ET
import json

class ImageDownloader:
    def __init__(self, protein, output_dir="outputs", processed_proteins_file="processed_proteins.json"):
        self.url = f"https://www.proteinatlas.org/search/{protein}?format=xml&download=yes"
        self.output_dir = output_dir
        self.processed_proteins_file = os.path.join(self.output_dir, processed_proteins_file)
        self.root = None

        # Ensure the output directory exists
        os.makedirs(self.output_dir, exist_ok=True)

        # Load the list of processed proteins from the JSON file, if it exists
        self.processed_proteins = self._load_processed_proteins()

    def _load_processed_proteins(self):
        if os.path.exists(self.processed_proteins_file):
            with open(self.processed_proteins_file, "r") as file:
                return json.load(file)
        else:
            return []

    def _save_processed_protein(self, protein_name):
        if protein_name not in self.processed_proteins:
            self.processed_proteins.append(protein_name)
            with open(self.processed_proteins_file, "w") as file:
                json.dump(self.processed_proteins, file)

    def fetch_xml_data(self):
        response = requests.get(self.url)
        if response.status_code == 200:
            self.root = ET.fromstring(response.content)
            print("XML data fetched successfully.")
        else:
            print(f"Failed to retrieve XML data. Status code: {response.status_code}")
            self.root = None

    def download_images(self):
        if self.root is None:
            print("No XML data to process.")
            return

        protein_name = self._get_protein_name()
        if self._is_protein_processed(protein_name):
            print(f"Protein '{protein_name}' has already been processed. Skipping download.")
            return

        for image_elem in self.root.findall(".//image"):
            image_url_elem = image_elem.find("imageUrl")
            if image_url_elem is not None and image_url_elem.text.startswith('http'):
                image_url = image_url_elem.text
                print(f"Image URL: {image_url}")

                self._download_image(image_url)
            else:
                print(f"Invalid or empty image URL: {image_url_elem.text if image_url_elem is not None else 'None'}")

        # Mark the protein as processed
        self._save_processed_protein(protein_name)

    def _download_image(self, image_url):
        img_response = requests.get(image_url)
        img_filename = image_url.split("/")[-1]
        img_path = os.path.join(self.output_dir, img_filename)

        with open(img_path, "wb") as img_file:
            img_file.write(img_response.content)
        print(f"Downloaded: {img_path}")

    def _get_protein_name(self):
        """Extracts the protein name from the XML data."""
        protein_name_elem = self.root.find(".//protein/name")
        if protein_name_elem is not None:
            return protein_name_elem.text
        return "Unknown_Protein"

    def _is_protein_processed(self, protein_name):
        """Checks if the protein has already been processed."""
        return protein_name in self.processed_proteins
