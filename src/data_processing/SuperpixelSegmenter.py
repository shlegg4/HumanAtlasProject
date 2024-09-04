import os
import numpy as np
from skimage import io, segmentation, filters, measure, color
import matplotlib.pyplot as plt
from ..utils import log_message

class SuperpixelSegmenter:
    def __init__(self, image_path, n_segments=200, compactness=10, sigma=1):
        """
        Initialize the SuperpixelSegmenter with the image and segmentation parameters.
        """
        self.image_path = image_path
        self.image = io.imread(image_path)
        self.n_segments = n_segments
        self.compactness = compactness
        self.sigma = sigma
        self.segments = None
        self.output_dir = None
        self.variance_threshold = 0.0001  # Threshold for filtering out low-entropy segments

    def perform_slic_segmentation(self):
        """
        Perform SLIC superpixel segmentation on the image.
        """
        self.segments = segmentation.slic(
            self.image, 
            n_segments=self.n_segments, 
            compactness=self.compactness, 
            sigma=self.sigma, 
            start_label=1
        )

    def create_output_directory(self):
        """
        Create a directory based on the input image name to store the segments.
        """
        image_name = os.path.splitext(os.path.basename(self.image_path))[0]
        self.output_dir = os.path.join("segments", image_name)
        
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
    
    def calculate_variance(self, segment_image):
        """
        Calculate the variance of a segment image.
        """
        gray_image = color.rgb2gray(segment_image)
        variance = np.var(gray_image)
        return variance

    def extract_segment_path(self, mask):
        """
        Extract the path (coordinates) of the segment based on the mask.
        """
        coords = np.column_stack(np.where(mask))
        return coords
    
    def save_segments(self):
        """
        Save each segment as an individual image in the output directory, filtering out low-variance segments.
        Set the background to white instead of black. Track accepted and rejected segments.
        Returns a list of dictionaries containing the path and file path for each segment.
        """
        if self.segments is None:
            raise ValueError("Segmentation has not been performed yet.")
        
        num_segments = np.max(self.segments) + 1
        self.segment_status = np.zeros(num_segments, dtype=bool)  # Track which segments are accepted
        segments_info = []  # List to store path and file path of each segment
        
        for i in range(num_segments):
            mask = self.segments == i
            
            # Create a white image instead of a black one
            segment_image = np.ones_like(self.image) * 255  # White background
            
            # Copy the segment to the white image
            segment_image[mask] = self.image[mask]
            
            # Calculate variance and filter out low-variance segments
            variance = self.calculate_variance(segment_image)
            if variance > self.variance_threshold:
                output_path = os.path.join(self.output_dir, f'segment_{i}.png')
                io.imsave(output_path, segment_image)
                self.segment_status[i] = True  # Mark as accepted
                segment_path = self.extract_segment_path(mask)  # Get path (coordinates)
                segments_info.append({
                    "path": segment_path,
                    "file_path": output_path
                })
                print(f'Segment {i}/{num_segments} saved to {output_path} with variance {variance:.2f}')
            else:
                self.segment_status[i] = False  # Mark as rejected
                #print(f'Segment {i}/{num_segments} discarded due to low variance {variance:.2f}')
        
        return segments_info
    
    def segment_and_save(self):
        """
        Perform the full process: segment the image and save each segment.
        Returns a list of dictionaries containing the path and file path for each segment.
        """
        self.perform_slic_segmentation()
        self.create_output_directory()
        return self.save_segments()
        
    def display_segments(self):
        """
        Display the segmented image with superpixel boundaries overlaid.
        Accepted segments are colored green, and rejected segments are colored red.
        """
        if self.segments is None:
            raise ValueError("Segmentation has not been performed yet.")
        
        # Create a copy of the image for visualization
        display_image = self.image.copy()
        
        # Color accepted segments green and rejected segments red
        try:
            if self.segment_status is not None:
                for i, accepted in enumerate(self.segment_status):
                    mask = self.segments == i
                    if accepted:
                        display_image[mask] = 0.5 * display_image[mask] + 0.5 * [0, 255, 0]  # Green for accepted
                    else:
                        display_image[mask] = 0.5 * display_image[mask] + 0.5 * [255, 0, 0]  # Red for rejected
        except AttributeError:
            pass    
        dark_boundary_color = [255, 215, 0]  # RGB for black (or dark color)
        display_image[segmentation.find_boundaries(self.segments, mode='thick')] = dark_boundary_color
            
        # Display the image
        plt.figure(figsize=(10, 10))
        plt.imshow(display_image)
        plt.axis('off')
        plt.title("Superpixel Segmentation (Green = Accepted, Red = Rejected)")
        plt.show()
        
    def segment_and_display(self):
        self.perform_slic_segmentation()
        self.display_segments()
