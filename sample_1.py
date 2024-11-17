#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 16 10:41:32 2024

@author: soumyashekhar
"""

import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from datetime import datetime

class ThermalEffluentDetector:
    def __init__(self, temp_threshold=5, area_threshold=100, flow_direction='vertical', inlet_size_threshold=0.3):
        """
        Initialize the detector with thresholds
        
        Args:
            temp_threshold (float): Minimum temperature difference to flag as anomaly
            area_threshold (int): Minimum area to consider
            flow_direction (str): Direction of river flow ('vertical' or 'horizontal')
            inlet_size_threshold (float): Ratio of inlet size to total anomaly size (0-1)
        """
        self.temp_threshold = temp_threshold
        self.area_threshold = area_threshold
        self.flow_direction = flow_direction
        self.inlet_size_threshold = inlet_size_threshold
        self.baseline_temp = None
    
    def load_thermal_image(self, image_path):
        """Load and preprocess thermal image"""
        image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        if image is None:
            raise ValueError(f"Could not load image at {image_path}")
            
        image = image.astype(np.float32)
        if image.dtype == np.uint16:
            image = -20 + (image / 65535.0) * 140
        elif np.max(image) <= 255:
            image = -20 + (image / 255.0) * 140
            
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
        return image
    
    def set_baseline(self, thermal_image, roi=None):
        """Set baseline temperature from clean section"""
        if roi:
            x, y, w, h = roi
            region = thermal_image[y:y+h, x:x+w]
        else:
            region = thermal_image
        self.baseline_temp = np.mean(region)
        self.temp_std = np.std(region)

    def find_inlet_regions(self, thermal_image, temp_diff):
        """
        Detect inlet regions using temperature gradients
        Returns list of inlet regions with their characteristics
        """
        # Create gradient maps
        grad_y = cv2.Sobel(temp_diff, cv2.CV_32F, 0, 1, ksize=3)
        grad_x = cv2.Sobel(temp_diff, cv2.CV_32F, 1, 0, ksize=3)
        
        # Calculate gradient magnitude
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        # Threshold for significant gradients
        grad_threshold = np.max(gradient_magnitude) * 0.3
        significant_gradients = gradient_magnitude > grad_threshold
        
        # Create binary mask of temperature anomalies
        temp_mask = np.abs(temp_diff) > self.temp_threshold
        
        # Combine gradient and temperature information
        inlet_mask = significant_gradients & temp_mask
        
        # Clean up the mask
        kernel = np.ones((3,3), np.uint8)
        inlet_mask = cv2.morphologyEx(inlet_mask.astype(np.uint8), cv2.MORPH_OPEN, kernel)
        
        # Find contours of potential inlet regions
        contours, _ = cv2.findContours(inlet_mask.astype(np.uint8), 
                                     cv2.RETR_EXTERNAL, 
                                     cv2.CHAIN_APPROX_SIMPLE)
        
        inlet_regions = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < self.area_threshold:
                continue
                
            # Get bounding box
            x, y, w, h = cv2.boundingRect(contour)
            
            # Analyze temperature profile in the region
            region = thermal_image[y:y+h, x:x+w]
            temp_profile = np.mean(region, axis=1 if self.flow_direction == 'horizontal' else 0)
            
            # Find point of maximum gradient
            gradient_profile = np.gradient(temp_profile)
            max_grad_idx = np.argmax(np.abs(gradient_profile))
            
            # Calculate inlet point based on flow direction
            if self.flow_direction == 'horizontal':
                inlet_x = x
                inlet_y = y + max_grad_idx
                inlet_w = int(w * self.inlet_size_threshold)
                inlet_h = int(h * self.inlet_size_threshold)
            else:
                inlet_x = x + max_grad_idx
                inlet_y = y
                inlet_w = int(w * self.inlet_size_threshold)
                inlet_h = int(h * self.inlet_size_threshold)
            
            # Calculate confidence based on temperature gradient and area
            temp_diff_local = np.max(region) - self.baseline_temp
            grad_strength = np.max(np.abs(gradient_profile))
            confidence = min(100, (temp_diff_local / self.temp_threshold + 
                                 grad_strength / grad_threshold) * 50)
            
            inlet_regions.append({
                'inlet_bbox': (inlet_x, inlet_y, inlet_w, inlet_h),
                'full_bbox': (x, y, w, h),
                'temperature': np.mean(region),
                'max_temperature': np.max(region),
                'gradient_strength': grad_strength,
                'confidence': confidence
            })
        
        # Sort by confidence
        inlet_regions.sort(key=lambda x: x['confidence'], reverse=True)
        return inlet_regions

    def detect_anomalies(self, thermal_image):
        """Detect thermal anomalies and inlet points"""
        if self.baseline_temp is None:
            raise ValueError("Baseline temperature not set. Call set_baseline() first.")
            
        # Calculate temperature difference
        temp_diff = thermal_image - self.baseline_temp
        
        # Find inlet regions
        inlet_regions = self.find_inlet_regions(thermal_image, temp_diff)
        
        return inlet_regions, temp_diff
    
    def visualize_results(self, thermal_image, inlet_regions, temp_diff):
        """Visualize detected inlets and temperature differences"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 15))
        
        # Original thermal image
        im1 = axes[0,0].imshow(thermal_image, cmap='inferno')
        axes[0,0].set_title('Original Thermal Image')
        plt.colorbar(im1, ax=axes[0,0], label='Temperature (°C)')
        
        # Temperature difference map
        im2 = axes[0,1].imshow(temp_diff, cmap='RdBu_r')
        axes[0,1].set_title('Temperature Difference Map')
        plt.colorbar(im2, ax=axes[0,1], label='Temperature Difference (°C)')
        
        # Gradient magnitude
        grad_y = cv2.Sobel(temp_diff, cv2.CV_32F, 0, 1, ksize=3)
        grad_x = cv2.Sobel(temp_diff, cv2.CV_32F, 1, 0, ksize=3)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        im3 = axes[1,0].imshow(gradient_magnitude, cmap='viridis')
        axes[1,0].set_title('Temperature Gradient Magnitude')
        plt.colorbar(im3, ax=axes[1,0], label='Gradient Strength')
        
        # Annotated thermal image with inlet boxes
        axes[1,1].imshow(thermal_image, cmap='inferno')
        for inlet in inlet_regions:
            # Draw small box around inlet point
            x, y, w, h = inlet['inlet_bbox']
            rect = plt.Rectangle((x, y), w, h, fill=False, color='red', linewidth=2)
            axes[1,1].add_patch(rect)
            
            # Add information text
            text = f"Temp: {inlet['temperature']:.1f}°C\nConf: {inlet['confidence']:.0f}%"
            axes[1,1].text(x, y-20, text, color='white', fontsize=8,
                          bbox=dict(facecolor='red', alpha=0.5))
            
        axes[1,1].set_title('Detected Inlet Points')
        
        plt.tight_layout()
        return fig

def main():
    # Initialize detector
    detector = ThermalEffluentDetector(
        temp_threshold=5,
        area_threshold=100,
        flow_direction='vertical',
        inlet_size_threshold=0.3  # Adjust this to change inlet box size
    )
    
    # Load and process image
    image_path = "/Users/soumyashekhar/Desktop/inlet.jpg"  # Replace with your image path
    thermal_image = detector.load_thermal_image(image_path)
    
    # Set baseline
    roi = (0, 0, 100, 100)  # Adjust ROI to clean river section
    detector.set_baseline(thermal_image, roi)
    
    # Detect inlets
    inlet_regions, temp_diff = detector.detect_anomalies(thermal_image)
    
    # Visualize results
    fig = detector.visualize_results(thermal_image, inlet_regions, temp_diff)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    fig.savefig(f"thermal_analysis_{timestamp}.png")
    
    # Print detected inlets
    for i, inlet in enumerate(inlet_regions, 1):
        print(f"\nInlet Point {i}:")
        print(f"Location: {inlet['inlet_bbox'][:2]}")
        print(f"Size: {inlet['inlet_bbox'][2:]}")
        print(f"Temperature: {inlet['temperature']:.1f}°C")
        print(f"Maximum Temperature: {inlet['max_temperature']:.1f}°C")
        print(f"Gradient Strength: {inlet['gradient_strength']:.1f}")
        print(f"Confidence: {inlet['confidence']:.1f}%")

if __name__ == "__main__":
    main()