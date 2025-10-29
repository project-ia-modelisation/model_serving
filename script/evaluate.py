import pickle
import numpy as np
import trimesh
import os

def load_preprocessed_model(filepath):
    """ Charger un modèle 3D prétraité depuis un fichier pickle. """
    try:
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Le fichier spécifié n'existe pas : {filepath}")
        
        with open(filepath, 'rb') as f:
            model = pickle.load(f)

        # Si le modèle est un TrackedArray, le convertir en Trimesh
        if isinstance(model, np.ndarray):
            model = trimesh.Trimesh(vertices=model)
        elif isinstance(model, trimesh.caching.TrackedArray):
            model = trimesh.Trimesh(vertices=model.data)

        if not isinstance(model, trimesh.Trimesh):
            raise ValueError(f"Le fichier pickle ne contient pas un objet Trimesh valide. Type détecté : {type(model)}.")

        if len(model.vertices) == 0:
            raise ValueError("Le modèle prétraité est vide. Aucun sommet n'a été trouvé.")

        print(f"✅ Modèle prétraité chargé avec succès : {type(model)} avec {len(model.vertices)} sommets.")
        return model
    except FileNotFoundError as fnf_error:
        print(f"❌ Erreur : {fnf_error}")
        return None
    except pickle.UnpicklingError:
        print("❌ Erreur lors du dépickle du fichier. Il pourrait être corrompu.")
        return None
    except ValueError as val_error:
        print(f"❌ Erreur de validation : {val_error}")
        return None
    except Exception as e:
        print(f"❌ Erreur inattendue lors du chargement du modèle prétraité : {e}")
        return None


def resample_vertices(vertices, target_count, model_name="Modèle inconnu"):
    """
    Rééchantillonne un ensemble de sommets pour atteindre un nombre cible de points.
    """
    print(f"🔄 [resample_vertices] {model_name} : {len(vertices)} → Cible : {target_count}")

    if len(vertices) == 0:
        raise ValueError("❌ ERREUR : Les sommets d'entrée sont vides.")

    try:
        mesh = trimesh.Trimesh(vertices=vertices)
        if len(vertices) < target_count:
            points_to_add = target_count - len(vertices)

            if len(mesh.faces) == 0:
                raise ValueError("⚠️ Pas de faces, interpolation linéaire forcée...")

            new_points, _ = trimesh.sample.sample_surface_even(mesh, points_to_add)
            if new_points is None or len(new_points) == 0:
                raise ValueError("❌ ERREUR CRITIQUE : `sample_surface_even()` a retourné un tableau vide !")

            resampled = np.vstack((vertices, new_points))
        else:
            indices = np.linspace(0, len(vertices) - 1, target_count, dtype=int)
            resampled = vertices[indices]
    except Exception as e:
        print(f"⚠️ [resample_vertices] {e}")
        print("🛑 Passage en mode interpolation linéaire...")
        new_points = np.tile(vertices, (target_count // len(vertices) + 1, 1))[:target_count]
        resampled = new_points

    resampled = resampled[:target_count]  # 🔒 S'assure qu'on a exactement target_count sommets
    print(f"📊 [resample_vertices] {model_name} après rééchantillonnage : {len(resampled)} sommets")
    return resampled

def validate_faces(model):
    """ Vérifie que les indices des faces ne dépassent pas la taille des sommets. """
    if not isinstance(model, trimesh.Trimesh):
        raise ValueError("L'objet fourni n'est pas un modèle Trimesh valide.")

    if len(model.faces) == 0:
        raise ValueError("Le modèle ne contient aucune face.")

    max_index = len(model.vertices) - 1
    for i, face in enumerate(model.faces):
        if any(index > max_index or index < 0 for index in face):
            raise ValueError(f"❌ Indice de face invalide détecté à la face {i} : {face}")

    print("✅ Vérification des indices des faces terminée, aucun problème détecté.")

def evaluate_model(preprocessed_model, ground_truth_model):
    """
    Évalue la similarité entre le modèle prétraité et le modèle de vérité terrain.
    """
    try:
        if not isinstance(preprocessed_model, trimesh.Trimesh):
            raise ValueError(f"❌ ERREUR : preprocessed_model n'est pas un Trimesh mais {type(preprocessed_model)}")
        if not isinstance(ground_truth_model, trimesh.Trimesh):
            raise ValueError(f"❌ ERREUR : ground_truth_model n'est pas un Trimesh mais {type(ground_truth_model)}")

        # Vérification des tailles avant rééchantillonnage
        print(f"🔍 Vérification des tailles avant rééchantillonnage...")
        print(f"   🔹 Prédits : {len(preprocessed_model.vertices)} sommets")
        print(f"   🔹 Vérité terrain : {len(ground_truth_model.vertices)} sommets")

        target_count = max(len(preprocessed_model.vertices), len(ground_truth_model.vertices))
        preprocessed_resampled = resample_vertices(preprocessed_model.vertices, target_count, "Modèle prétraité")
        ground_truth_resampled = resample_vertices(ground_truth_model.vertices, target_count, "Modèle vérité terrain")

        print("✅ Rééchantillonnage terminé :", len(preprocessed_resampled), "sommets alignés")

        # Calculer la similarité entre les modèles rééchantillonnés
        # Ajoutez votre logique d'évaluation ici

        return {"similarity_score": 0.9}  # Exemple de score de similarité
    
    except Exception as e:
        print(f"❌ Erreur lors de l'évaluation du modèle : {e}")
        return None

def compute_metrics(predicted_vertices, ground_truth_vertices):
    """ Calcul des métriques entre sommets prédit et vérité terrain. """
    if not isinstance(predicted_vertices, np.ndarray) or not isinstance(ground_truth_vertices, np.ndarray):
        raise ValueError("❌ ERREUR : Les sommets doivent être des tableaux numpy.")

    if predicted_vertices.shape != ground_truth_vertices.shape:
        raise ValueError("❌ ERREUR : Les sommets doivent avoir la même forme après rééchantillonnage.")

    metrics = {
        "mean_squared_error": float(np.mean((predicted_vertices - ground_truth_vertices) ** 2)),
        "max_error": float(np.max(np.abs(predicted_vertices - ground_truth_vertices))),
        "average_distance": float(np.mean(np.linalg.norm(predicted_vertices - ground_truth_vertices, axis=1)))
    }
    return metrics
