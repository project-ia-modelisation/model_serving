import json
import os
import numpy as np
from typing import Dict, List, Union

class PromptHandler:
    def __init__(self, config_file: str = "data/prompt_config.json"):
        """
        Gestionnaire de prompts pour la génération d'objets 3D.
        
        Args:
            config_file (str): Chemin vers le fichier de configuration des prompts
        """
        self.config = self._load_config(config_file)
        self.forme_base = 1000  # Nombre de sommets de base
        
    def _load_config(self, config_file: str) -> Dict:
        """Charge la configuration des prompts depuis un fichier JSON."""
        if not os.path.exists(config_file):
            default_config = {
                "formes_base": ["cube", "sphere", "pyramide", "cylindre"],
                "modificateurs": ["grand", "petit", "large", "étroit"],
                "transformations": ["rotation", "échelle", "translation"],
                "proprietes": ["lisse", "rugueux", "symétrique"]
            }
            os.makedirs(os.path.dirname(config_file), exist_ok=True)
            with open(config_file, 'w', encoding='utf-8') as f:
                json.dump(default_config, f, indent=4, ensure_ascii=False)
            return default_config
        
        with open(config_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def analyser_prompt(self, prompt: str) -> Dict[str, Union[str, List[str]]]:
        """
        Analyse le prompt utilisateur pour extraire les paramètres de génération.
        
        Args:
            prompt (str): Description textuelle de l'objet à générer
            
        Returns:
            Dict: Paramètres extraits du prompt
        """
        prompt = prompt.lower()
        params = {
            "forme_base": None,
            "modificateurs": [],
            "transformations": [],
            "proprietes": []
        }
        
        # Analyse de la forme de base
        for forme in self.config["formes_base"]:
            if forme in prompt:
                params["forme_base"] = forme
                break
                
        # Analyse des modificateurs et propriétés
        for categorie in ["modificateurs", "transformations", "proprietes"]:
            for element in self.config[categorie]:
                if element in prompt:
                    params[categorie].append(element)
        
        return params
    
    def generer_parametres_forme(self, params: Dict) -> Dict:
        """
        Convertit les paramètres textuels en paramètres numériques pour la génération.
        
        Args:
            params (Dict): Paramètres extraits du prompt
            
        Returns:
            Dict: Paramètres numériques pour la génération
        """
        nb_sommets = self.forme_base
        
        # Ajustement du nombre de sommets selon les modificateurs
        if "grand" in params["modificateurs"]:
            nb_sommets *= 1.5
        elif "petit" in params["modificateurs"]:
            nb_sommets *= 0.75
            
        return {
            "nb_sommets": int(nb_sommets),
            "lissage": "lisse" in params["proprietes"],
            "symetrie": "symétrique" in params["proprietes"],
            "transformations": params["transformations"]
        }
    
    def appliquer_prompt(self, prompt: str) -> Dict:
        """
        Traite un prompt utilisateur complet.
        
        Args:
            prompt (str): Description textuelle de l'objet à générer
            
        Returns:
            Dict: Paramètres de génération complets
        """
        params = self.analyser_prompt(prompt)
        if not params["forme_base"]:
            raise ValueError("Aucune forme de base n'a été détectée dans le prompt.")
            
        return self.generer_parametres_forme(params)