"""
Question/Answer Segmenter - Extracts questions and answers from OCR text
"""

import re
from typing import List, Dict, Tuple
import logging

logger = logging.getLogger(__name__)


class QASegmenter:
    """Segmente le texte OCR en questions et réponses"""
    
    def __init__(self):
        # Patterns pour identifier les questions/réponses
        self.question_pattern = re.compile(r'^[\s]*(question|q|question|exercice|ex|problème)\s*[:\-\.]?\s*(\d+)?', 
                                          re.IGNORECASE | re.MULTILINE)
        self.answer_pattern = re.compile(r'^[\s]*(réponse|r|answer|a|solution|résp)\s*[:\-\.]?\s*(\d+)?', 
                                        re.IGNORECASE | re.MULTILINE)
        
    def segment(self, text: str) -> Dict[str, List[str]]:
        """
        Segmente le texte en questions et réponses
        Retourne: {"questions": [...], "answers": [...], "mixed": [...]}
        """
        if not text or not text.strip():
            return {"questions": [], "answers": [], "mixed": []}
        
        lines = text.split('\n')
        questions = []
        answers = []
        current_question = []
        current_answer = []
        mode = None  # 'question', 'answer', ou None
        
        for line in lines:
            line = line.strip()
            if not line or line == "--- Page Break ---":
                continue
                
            # Détecte si c'est une question
            if self.question_pattern.match(line):
                # Sauvegarde la réponse précédente
                if current_answer:
                    answers.append('\n'.join(current_answer))
                    current_answer = []
                # Sauvegarde la question précédente
                if current_question:
                    questions.append('\n'.join(current_question))
                current_question = [line]
                mode = 'question'
            # Détecte si c'est une réponse
            elif self.answer_pattern.match(line):
                # Sauvegarde la question précédente
                if current_question:
                    questions.append('\n'.join(current_question))
                    current_question = []
                current_answer = [line]
                mode = 'answer'
            # Continue la question/réponse courante
            elif mode == 'question' and current_question:
                current_question.append(line)
            elif mode == 'answer' and current_answer:
                current_answer.append(line)
            else:
                # Sans marqueur clair, on accumule
                if current_question and not current_answer:
                    current_question.append(line)
                elif current_answer:
                    current_answer.append(line)
                else:
                    current_question.append(line)
        
        # Sauvegarde les derniers éléments
        if current_question:
            questions.append('\n'.join(current_question))
        if current_answer:
            answers.append('\n'.join(current_answer))
        
        # Retour structuré
        return {
            "questions": [q.strip() for q in questions if q.strip()],
            "answers": [a.strip() for a in answers if a.strip()],
        }
    
    def extract_student_code(self, text: str) -> str:
        """
        Extrait le code/numéro de l'étudiant du texte
        Cherche les patterns : "Code: XYZ", "Student: XYZ", etc.
        """
        patterns = [
            r'(?:code|numéro|étudiant|student)[\s\:\-]+([\w\-]+)',
            r'^[\w\-]+(?:\s+[\w\-]+){0,2}$',  # Première ligne si elle ressemble à un code
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text.split('\n')[0], re.IGNORECASE)
            if match:
                return match.group(1)
        
        return "unknown"
    
    def format_qa_pairs(self, qa_dict: Dict[str, List[str]]) -> str:
        """
        Formate les Q/A en un texte structuré pour le LLM
        """
        output = []
        
        questions = qa_dict.get("questions", [])
        answers = qa_dict.get("answers", [])
        
        # Associe les questions aux réponses
        pairs = []
        for i in range(max(len(questions), len(answers))):
            q = questions[i] if i < len(questions) else f"Question {i+1}:"
            a = answers[i] if i < len(answers) else "[No answer provided]"
            pairs.append((q, a))
        
        # Formate avec numérotation
        for idx, (q, a) in enumerate(pairs, 1):
            output.append(f"\n{'='*60}")
            output.append(f"Question {idx}:")
            output.append(f"{'='*60}")
            output.append(q)
            output.append(f"\n--- Réponse fournie ---")
            output.append(a)
        
        return '\n'.join(output)
