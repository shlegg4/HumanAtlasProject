import os
import numpy as np
from skimage import io, segmentation, filters, measure, color
import matplotlib.pyplot as plt
from ..utils import log_message

class SuperpixelSegmenter:
    def __init__(self, n_segments=200, compactness=10, sigma=1):
        """
        Initialize the SuperpixelSegmenter with the image and segmentation parameters.
        """
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
        mean_coords = np.mean(coords, axis=0)
        return coords, mean_coords
        
        
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

        # Create a mask for each segment and compute variance for all segments in one step
        variances = np.array([
            self.calculate_variance(self.image[self.segments == i])
            for i in range(num_segments)
        ])
        
        # Determine which segments pass the variance threshold
        accepted_segments = variances > self.variance_threshold
        self.segment_status = accepted_segments  # Mark accepted/rejected segments

        # Now process only the accepted segments
        for i in range(num_segments):
            if accepted_segments[i]:
                mask = self.segments == i

                # Create a white image background
                segment_image = np.ones_like(self.image) * 255
                
                # Copy the segment to the white image
                segment_image[mask] = self.image[mask]

                # Extract the segment path (coordinates) and add the info to the list
                segment_path, mean_coords = self.extract_segment_path(mask)
                segments_info.append({
                    "path": mean_coords,
                    "image": segment_image
                })
                log_message('info', f'segment {i}/{num_segments} passed threshold')
            else:
                log_message('info', f'segment {i}/{num_segments} did not pass threshold')

        return segments_info
    
    def segment_and_save(self, image_path):
        """
        Perform the full process: segment the image and save each segment.
        Returns a list of dictionaries containing the path and file path for each segment.
        """
        self.image_path = image_path
        self.image = io.imread(image_path)
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
