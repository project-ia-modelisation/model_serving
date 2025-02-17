import torch
from models.model import Simple3DGenerator, Simple2DGenerator
import trimesh
import os
from scipy.spatial import ConvexHull
from visualisation.lecture import save_image
import numpy as np
import torch.nn as nn

def generate_model(model, noise_dim=100, min_vertices=100, max_vertices=1000):
    num_vertices = np.random.randint(min_vertices, max_vertices)
    model.fc4 = nn.Linear(1024, num_vertices * 3)  # Mettre à jour la couche finale pour correspondre au nombre de sommets
    model.num_vertices = num_vertices  # Mettre à jour le nombre de sommets du modèle
    noise = torch.randn(1, noise_dim)  # Random noise input
    with torch.no_grad():
        generated_vertices = model(noise).numpy().reshape(num_vertices, 3)
    # Vérifions que le nombre de sommets générés est correct
    if len(generated_vertices) < min_vertices or len(generated_vertices) > max_vertices:
        print(f"Erreur : Nombre de sommets générés ({len(generated_vertices)}) hors des limites spécifiées ({min_vertices}-{max_vertices}).")
        return None
    # Générer des faces pour le maillage
    if len(generated_vertices) >= 4:  # ConvexHull requires at least 4 points
        hull = ConvexHull(generated_vertices)
        faces = hull.simplices
        generated_model = trimesh.Trimesh(vertices=generated_vertices, faces=faces)
    else:
        print("Erreur : Pas assez de points pour générer un maillage.")
        return None
    return generated_model

def generate_image(model, noise_dim=100):
    noise = torch.randn(1, noise_dim)  # Random noise input
    with torch.no_grad():
        generated_image = model(noise).numpy().reshape(64, 64)  # Assuming 64x64 image size
    return generated_image

def is_valid_3d_shape(model):
    return model is not None and model.is_volume

def generate_and_save_model(model, output_dir="./data", max_attempts=10, is_2d=False, min_vertices=100, max_vertices=1000):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Trouver le prochain numéro de fichier disponible
    existing_files = [f for f in os.listdir(output_dir) if f.startswith("generated_model_") and (f.endswith(".obj") if not is_2d else f.endswith(".png"))]
    existing_numbers = [int(f.split('_')[2].split('.')[0]) for f in existing_files]
    next_number = max(existing_numbers, default=0) + 1
    
    # Générer un seul modèle
    generated_model = None
    attempts = 0
    while not is_valid_3d_shape(generated_model) and attempts < max_attempts:
        print(f"Tentative de génération {attempts + 1}/{max_attempts}")
        generated_model = generate_model(model, min_vertices=min_vertices, max_vertices=max_vertices) if not is_2d else generate_image(model)
        attempts += 1
    
    if generated_model is not None:
        output_path = os.path.join(output_dir, f"generated_model_{next_number}.obj" if not is_2d else f"generated_model_{next_number}.png")
        generated_model.export(output_path) if not is_2d else save_image(generated_model, output_path)
        print(f"Modèle sauvegardé à {output_path}")
    else:
        print("Échec de la génération d'un modèle valide après plusieurs tentatives.")

if __name__ == "__main__":
    model = Simple3DGenerator()
    model.load_state_dict(torch.load("./data/model.pth", map_location=torch.device('cpu'), weights_only=True))  # Load the trained model
    model.eval()
    generate_and_save_model(model)
