import os
import requests
import xml.etree.ElementTree as ET

class ImageDownloader:
    def __init__(self, protein, output_dir="outputs"):
        self.url = f"https://www.proteinatlas.org/search/{protein}?format=xml&download=yes"
        self.output_dir = output_dir
        self.root = None

        # Ensure the output directory exists
        os.makedirs(self.output_dir, exist_ok=True)

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

        for image_elem in self.root.findall(".//image"):
            image_url_elem = image_elem.find("imageUrl")
            if image_url_elem is not None and image_url_elem.text.startswith('http'):
                image_url = image_url_elem.text
                print(f"Image URL: {image_url}")

                self._download_image(image_url)
            else:
                print(f"Invalid or empty image URL: {image_url_elem.text if image_url_elem is not None else 'None'}")

    def _download_image(self, image_url):
        img_response = requests.get(image_url)
        img_filename = image_url.split("/")[-1]
        img_path = os.path.join(self.output_dir, img_filename)

        with open(img_path, "wb") as img_file:
            img_file.write(img_response.content)
        print(f"Downloaded: {img_path}")

# Usage
protein = "SUCLG2"
downloader = ImageDownloader(protein=protein)
downloader.fetch_xml_data()
downloader.download_images()
