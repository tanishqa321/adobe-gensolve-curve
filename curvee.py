import cv2
import numpy as np
import svgwrite
from scipy.optimize import minimize
from scipy.spatial.distance import euclidean

def preprocess_image(image_path):
    # Load the image in grayscale
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Apply edge detection (Canny)
    edges = cv2.Canny(image, 50, 150)
    
    # Find contours in the edged image
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Approximate contours to polylines
    polylines = [cv2.approxPolyDP(cnt, epsilon=2.0, closed=False) for cnt in contours]
    
    return polylines

def bezier(t, p0, p1, p2, p3):
    return (1-t)**3 * p0 + 3*(1-t)**2 * t * p1 + 3*(1-t) * t**2 * p2 + t**3 * p3

def fit_bezier_curve(points):
    def objective(params, points):
        p0, p1, p2, p3 = params[:2], params[2:4], params[4:6], params[6:8]
        error = 0
        for i, t in enumerate(np.linspace(0, 1, len(points))):
            point_on_curve = bezier(t, p0, p1, p2, p3)
            error += euclidean(point_on_curve, points[i])
        return error
    
    p0, p3 = np.array(points[0]), np.array(points[-1])
    p1, p2 = (2/3)*p0 + (1/3)*p3, (1/3)*p0 + (2/3)*p3
    initial_guess = np.array([p0, p1, p2, p3]).flatten()
    
    result = minimize(objective, initial_guess, args=(points,))
    return result.x.reshape(4, 2)

def regularize_curve(bezier_curve):
    t_values = np.linspace(0, 1, 100)
    regularized_curve = [bezier(t, *bezier_curve) for t in t_values]
    return regularized_curve

def bezier_to_svg(bezier_curves, filename="output.svg"):
    dwg = svgwrite.Drawing(filename, profile='tiny')
    for curve in bezier_curves:
        p0, p1, p2, p3 = curve
        dwg.add(dwg.path(d=f"M{p0[0]},{p0[1]} C{p1[0]},{p1[1]} {p2[0]},{p2[1]} {p3[0]},{p3[1]}", fill="none", stroke="yellow"))
    dwg.save()

def process_image(image_path, output_svg_path):
    polylines = preprocess_image(image_path)
    
    bezier_curves = []
    for polyline in polylines:
        points = polyline[:, 0, :]  # Extract points from polyline
        bezier_curve = fit_bezier_curve(points)
        regularized_curve = regularize_curve(bezier_curve)
        bezier_curves.append(regularized_curve)
    
    bezier_to_svg(bezier_curves, output_svg_path)

# Example usage
image_path = "/content/frag0.csv"  # Input image file path
output_svg_path = "/content/output.svg"  # Desired output SVG file path
process_image(image_path, output_svg_path)
