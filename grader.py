"""
Exam Grader - Uses Open Source LLM API for grading
Supports: Hugging Face, Together AI, Replicate, or custom API endpoints
Scores: 0-20 (European grading scale)
"""

import os
import json
import logging
from typing import Dict, Optional
import requests

logger = logging.getLogger(__name__)


class OpenSourceLLMGrader:
    """
    Interface pour grader avec un LLM open source via API
    Supporte: Hugging Face, Together AI, Replicate, ou endpoint personnalisé
    Scores: 0-20 (échelle européenne)
    """
    
    def __init__(self, api_key: str = None, 
                 api_endpoint: str = None,
                 model_name: str = None):
        """
        Args:
            api_key: Clé API pour le service LLM (ou variable d'env)
            api_endpoint: URL de l'endpoint API
                Exemples:
                - Hugging Face: https://api-inference.huggingface.co/models/
                - Together AI: https://api.together.xyz/inference
                - Replicate: https://api.replicate.com/v1/predictions
                - Custom: http://localhost:8000/api/generate
            model_name: Nom du modèle (ex: mistralai/Mistral-7B-Instruct-v0.1)
        """
        self.api_key = api_key
        self.api_endpoint = api_endpoint
        self.model_name = model_name
        self.available = False
        
        self._init_llm()
    
    def _init_llm(self):
        """Initialise la connexion LLM"""
        
        # Par défaut : Hugging Face avec Mistral (open source)
        if not self.api_endpoint:
            self.api_endpoint = "https://api-inference.huggingface.co/models/"
        
        if not self.model_name:
            self.model_name = "mistralai/Mistral-7B-Instruct-v0.1"
        
        # Récupère la clé API
        self.api_key = self.api_key or os.getenv("LLM_API_KEY")
        
        if not self.api_key:
            # Essaie les clés d'env spécifiques
            self.api_key = (os.getenv("HUGGINGFACE_API_KEY") or 
                           os.getenv("TOGETHER_API_KEY") or 
                           os.getenv("REPLICATE_API_KEY"))
        
        if not self.api_key:
            logger.error("❌ API Key not found")
            logger.info("   Set environment variable: LLM_API_KEY or HUGGINGFACE_API_KEY")
            logger.info("   Or pass api_key parameter")
            return
        
        # Valide la connexion
        self._validate_connection()
    
    def _validate_connection(self):
        """Teste la connexion à l'API"""
        try:
            logger.info(f"Testing API connection...")
            logger.info(f"  Endpoint: {self.api_endpoint}")
            logger.info(f"  Model: {self.model_name}")
            
            # Test simple avec un prompt court
            test_prompt = "Hello, respond with 'OK'"
            response = self._query_api(test_prompt)
            
            if response and len(response) > 0:
                self.available = True
                logger.info(f"✅ LLM API is available and working")
            else:
                logger.warning(f"⚠️  LLM API responded but with no content")
        except Exception as e:
            logger.warning(f"⚠️  Could not validate API connection: {e}")
            logger.info(f"   Make sure your API key and endpoint are correct")
    
    def grade(self, exam_text: str, subject: str = "Matière") -> Dict[str, any]:
        """
        Note l'examen avec LLM open source et fournit des conseils
        Score: 0-20 (échelle européenne)
        
        Args:
            exam_text: Texte extrait du QA segmenter
            subject: Matière (GenAI, MLOps, etc.)
        
        Returns:
            {
                "score": 0-20,
                "grade_letter": "A/B/C/D/F",
                "strengths": ["..."],
                "weaknesses": ["..."],
                "feedback": "...",
                "improvements": ["..."]
            }
        """
        
        if not self.api_key:
            logger.warning("LLM API key not configured, returning default grading")
            return self._fallback_grade(exam_text)
        
        prompt = self._build_prompt(exam_text, subject)
        
        try:
            response = self._query_api(prompt)
            return self._parse_response(response)
        except Exception as e:
            logger.error(f"Error in grading: {e}")
            return self._fallback_grade(exam_text)
    
    def _build_prompt(self, exam_text: str, subject: str) -> str:
        """Construit le prompt pour le LLM"""
        return f"""Tu es un professeur en {subject}. Évalue l'examen suivant sur 20 et fournir:
1. Une note de 0-20
2. Une lettre (A/B/C/D/F) basée sur la note:
   - A: 18-20
   - B: 14-17
   - C: 10-13
   - D: 6-9
   - F: 0-5
3. Les points forts (2-3 items)
4. Les faiblesses (2-3 items)
5. Des conseils pour s'améliorer (3-4 items)

Format ta réponse EN JSON SEULEMENT, sans autre texte:
{{
  "score": X,
  "grade_letter": "Y",
  "strengths": ["...", "...", "..."],
  "weaknesses": ["...", "...", "..."],
  "feedback": "Analyse générale",
  "improvements": ["Amélioration 1", "Amélioration 2", "Amélioration 3"]
}}

EXAMEN À ÉVALUER:
{exam_text[:4000]}

RÉPONSE EN JSON:"""
    
    def _query_api(self, prompt: str) -> str:
        """Envoie une requête à l'API LLM"""
        
        # Détecte le type d'API basé sur l'endpoint
        if "huggingface" in self.api_endpoint:
            return self._query_huggingface(prompt)
        elif "together" in self.api_endpoint:
            return self._query_together(prompt)
        elif "replicate" in self.api_endpoint:
            return self._query_replicate(prompt)
        else:
            # Endpoint personnalisé (ex: Ollama compatible, LM Studio, etc.)
            return self._query_custom(prompt)
    
    def _query_huggingface(self, prompt: str) -> str:
        """Envoie une requête à Hugging Face API"""
        try:
            url = f"{self.api_endpoint}{self.model_name}"
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            payload = {
                "inputs": prompt,
                "parameters": {
                    "max_new_tokens": 1000,
                    "temperature": 0.7,
                    "top_p": 0.9,
                }
            }
            
            logger.info(f"Querying Hugging Face: {self.model_name}...")
            response = requests.post(url, headers=headers, json=payload, timeout=120)
            
            if response.status_code == 200:
                data = response.json()
                if isinstance(data, list) and len(data) > 0:
                    return data[0].get("generated_text", "")
                return ""
            else:
                logger.error(f"Hugging Face error {response.status_code}: {response.text}")
                return ""
        except Exception as e:
            logger.error(f"Hugging Face request failed: {e}")
            return ""
    
    def _query_together(self, prompt: str) -> str:
        """Envoie une requête à Together AI API"""
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "max_tokens": 1000,
                "temperature": 0.7,
                "top_p": 0.9,
            }
            
            logger.info(f"Querying Together AI: {self.model_name}...")
            response = requests.post(self.api_endpoint, headers=headers, json=payload, timeout=120)
            
            if response.status_code == 200:
                data = response.json()
                if "output" in data and "choices" in data["output"]:
                    return data["output"]["choices"][0].get("text", "")
                return ""
            else:
                logger.error(f"Together AI error {response.status_code}: {response.text}")
                return ""
        except Exception as e:
            logger.error(f"Together AI request failed: {e}")
            return ""
    
    def _query_replicate(self, prompt: str) -> str:
        """Envoie une requête à Replicate API"""
        try:
            headers = {
                "Authorization": f"Token {self.api_key}",
                "Content-Type": "application/json"
            }
            payload = {
                "version": self.model_name,
                "input": {
                    "prompt": prompt,
                    "max_tokens": 1000,
                    "temperature": 0.7,
                }
            }
            
            logger.info(f"Querying Replicate: {self.model_name}...")
            response = requests.post(self.api_endpoint, headers=headers, json=payload, timeout=120)
            
            if response.status_code == 201:
                result_url = response.json().get("urls", {}).get("get")
                if result_url:
                    # Poll for result
                    result = requests.get(result_url, headers=headers, timeout=120)
                    if result.status_code == 200:
                        output = result.json().get("output", [])
                        if output:
                            return "".join(output) if isinstance(output, list) else str(output)
                return ""
            else:
                logger.error(f"Replicate error {response.status_code}: {response.text}")
                return ""
        except Exception as e:
            logger.error(f"Replicate request failed: {e}")
            return ""
    
    def _query_custom(self, prompt: str) -> str:
        """Envoie une requête à un endpoint personnalisé"""
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            payload = {
                "prompt": prompt,
                "max_tokens": 1000,
                "temperature": 0.7,
            }
            
            logger.info(f"Querying custom endpoint: {self.api_endpoint}")
            response = requests.post(self.api_endpoint, headers=headers, json=payload, timeout=120)
            
            if response.status_code == 200:
                data = response.json()
                # Try various response formats
                if "text" in data:
                    return data["text"]
                elif "output" in data:
                    return data["output"]
                elif "response" in data:
                    return data["response"]
                elif "choices" in data and len(data["choices"]) > 0:
                    return data["choices"][0].get("text", "")
                return str(data)
            else:
                logger.error(f"Custom API error {response.status_code}: {response.text}")
                return ""
        except Exception as e:
            logger.error(f"Custom endpoint request failed: {e}")
            return ""
    
    def _parse_response(self, response: str) -> Dict[str, any]:
        """Parse la réponse JSON du LLM"""
        try:
            # Extrait le JSON de la réponse
            start = response.find("{")
            end = response.rfind("}") + 1
            if start >= 0 and end > start:
                json_str = response[start:end]
                result = json.loads(json_str)
                
                # Valide et normalise la réponse
                if "score" not in result:
                    result["score"] = 10
                
                # S'assure que score est 0-20
                score = result.get("score", 10)
                if score > 100:
                    # Si le LLM a donné 0-100, convertir
                    score = score / 5
                result["score"] = min(20, max(0, score))
                
                if "grade_letter" not in result:
                    result["grade_letter"] = self._score_to_grade(result["score"])
                
                for key in ["strengths", "weaknesses", "improvements"]:
                    if key not in result or not isinstance(result[key], list):
                        result[key] = []
                
                return result
        except Exception as e:
            logger.warning(f"Could not parse JSON response: {e}")
        
        return self._fallback_grade(response)
    
    def _fallback_grade(self, text: str) -> Dict[str, any]:
        """Note par défaut si le LLM échoue"""
        word_count = len(text.split())
        score = min(20, 10 + word_count / 100)
        
        return {
            "score": score,
            "grade_letter": self._score_to_grade(score),
            "strengths": [
                "Réponse fournie",
                "Tentative de réponse structurée",
                "Communication présente"
            ],
            "weaknesses": [
                "Détails insuffisants",
                "Clarté à améliorer",
                "Approche incomplète"
            ],
            "feedback": f"Examen avec {word_count} mots analysés. Assurez-vous que votre API est correctement configurée.",
            "improvements": [
                "Vérifier votre clé API (LLM_API_KEY)",
                "Fournir plus de détails techniques",
                "Améliorer la justification des réponses"
            ]
        }
    
    @staticmethod
    def _score_to_grade(score: float) -> str:
        """Convertit un score (0-20) en lettre (A-F)"""
        if score >= 18:
            return "A"
        elif score >= 14:
            return "B"
        elif score >= 10:
            return "C"
        elif score >= 6:
            return "D"
        else:
            return "F"
