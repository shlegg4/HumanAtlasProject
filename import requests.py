import requests
import xml.etree.ElementTree as ET

# URL to download XML data for SUCLG2
url = "https://www.proteinatlas.org/search/SUCLG2?format=xml&download=yes"

# Send a GET request to the URL
response = requests.get(url)

# Check if the request was successful
if response.status_code == 200:
    # Parse the XML content
    root = ET.fromstring(response.content)

    # Find all 'image' elements in the XML
    for image_elem in root.findall(".//image"):
        # Find the 'imageUrl' element within each 'image' element
        image_url_elem = image_elem.find("imageUrl")

        # Check if the imageUrl element exists and contains text
        if image_url_elem is not None and image_url_elem.text.startswith('http'):
            image_url = image_url_elem.text
            print(f"Image URL: {image_url}")

            # Download and save the image
            img_response = requests.get(image_url)
            img_filename = image_url.split("/")[-1]  # Extract the filename from the URL
            with open(img_filename, "wb") as img_file:
                img_file.write(img_response.content)
            print(f"Downloaded: {img_filename}")
        else:
            print(f"Invalid or empty image URL: {image_url_elem.text if image_url_elem is not None else 'None'}")
else:
    print(f"Failed to retrieve XML data. Status code: {response.status_code}")
