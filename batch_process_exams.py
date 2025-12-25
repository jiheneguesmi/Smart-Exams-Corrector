"""
Batch Processing Script - Processes all exams from data/ folder
Extracts OCR → Segments Q/A → Grades with LLM
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import Dict, List
from datetime import datetime

from ocr_pipeline import ExamOCR, OCRConfig, ModelType
from qa_extractor import QASegmenter
from grader import OpenSourceLLMGrader

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class BatchExamProcessor:
    """Traite tous les examens du dossier data/"""
    
    def __init__(self, data_root: str = "data", 
                 ocr_model: str = "french",
                 api_key: str = None,
                 api_endpoint: str = None,
                 model_name: str = None,
                 skip_ocr: bool = False):
        """
        Args:
            data_root: Chemin vers le dossier contenant GenAI/ et MLOps/
            ocr_model: "french", "english", ou "french_printed"
            api_key: Clé API pour le LLM
            api_endpoint: URL de l'API (défaut: Hugging Face)
                - Hugging Face: https://api-inference.huggingface.co/models/
                - Together AI: https://api.together.xyz/inference
                - Custom: http://localhost:8000/api/generate
            model_name: Nom du modèle (défaut: mistralai/Mistral-7B-Instruct-v0.1)
            skip_ocr: Si True, utilise les fichiers OCR existants
        """
        self.data_root = Path(data_root)
        self.ocr_model = ocr_model
        self.api_key = api_key
        self.api_endpoint = api_endpoint
        self.model_name = model_name
        self.skip_ocr = skip_ocr
        
        # Dossiers de sortie - créer AVANT l'OCR
        self.output_root = Path("results") / datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_root.mkdir(parents=True, exist_ok=True)
        logger.info(f"Output directory: {self.output_root}")
        
        # Initialise les composants
        self._init_ocr()
        self.qa_segmenter = QASegmenter()
        self.grader = OpenSourceLLMGrader(api_key=api_key, api_endpoint=api_endpoint, model_name=model_name)
    
    def _init_ocr(self):
        """Initialise le pipeline OCR"""
        model_type = ModelType[self.ocr_model.upper()]
        config = OCRConfig(
            model_type=model_type,
            remove_watermark=True,
            remove_blue_lines=True,
            output_dir=str(self.output_root / "ocr_raw")
        )
        self.ocr = ExamOCR(config)
        logger.info(f"✅ OCR initialized with model: {self.ocr_model}")
    
    def process_all(self):
        """Lance le traitement complet"""
        logger.info(f"\n{'='*80}")
        logger.info(f"BATCH PROCESSING STARTED - {datetime.now()}")
        logger.info(f"{'='*80}\n")
        
        # Scan des matières (GenAI/, MLOps/, etc.)
        subjects = [d for d in self.data_root.iterdir() 
                   if d.is_dir() and d.name not in ['.', '__pycache__']]
        
        if not subjects:
            logger.error(f"No subject folders found in {self.data_root}")
            return
        
        all_results = {}
        
        for subject_dir in sorted(subjects):
            subject_name = subject_dir.name
            logger.info(f"\n{'='*80}")
            logger.info(f"SUBJECT: {subject_name.upper()}")
            logger.info(f"{'='*80}\n")
            
            subject_results = self._process_subject(subject_dir, subject_name)
            all_results[subject_name] = subject_results
        
        # Sauvegarde le rapport final
        self._save_summary_report(all_results)
        
        logger.info(f"\n{'='*80}")
        logger.info(f"✅ BATCH PROCESSING COMPLETED")
        logger.info(f"Results saved to: {self.output_root}")
        logger.info(f"{'='*80}\n")
    
    def _process_subject(self, subject_dir: Path, subject_name: str) -> Dict:
        """Traite tous les étudiants d'une matière"""
        subject_results = {}
        
        # Scan des copies d'étudiants
        student_dirs = sorted([d for d in subject_dir.iterdir() 
                              if d.is_dir() and d.name not in ['.', '__pycache__']])
        
        for student_dir in student_dirs:
            student_id = student_dir.name
            logger.info(f"▶ Processing student: {student_id}")
            
            # Récupère les images
            image_files = self._get_image_files(student_dir)
            
            if not image_files:
                logger.warning(f"  ⚠️ No images found in {student_dir}")
                continue
            
            logger.info(f"  Found {len(image_files)} page(s)")
            
            # Traite cet étudiant
            student_result = self._process_student(
                student_id=student_id,
                student_dir=student_dir,
                subject_name=subject_name,
                image_files=image_files
            )
            
            subject_results[student_id] = student_result
        
        return subject_results
    
    def _process_student(self, student_id: str, student_dir: Path, 
                        subject_name: str, image_files: List[str]) -> Dict:
        """Traite l'examen d'un étudiant"""
        
        # Étape 1: OCR
        logger.info(f"  [1/4] Running OCR...")
        ocr_result = self.ocr.process_exam(image_files, exam_id=student_id)
        
        if not ocr_result.full_text:
            logger.warning(f"  ❌ OCR failed - no text extracted")
            return {"status": "failed", "error": "OCR failed"}
        
        logger.info(f"  [1/4] ✅ OCR done - {len(ocr_result.full_text)} chars")
        
        # Étape 2: Segmentation Q/A
        logger.info(f"  [2/4] Segmenting Q/A...")
        qa_dict = self.qa_segmenter.segment(ocr_result.full_text)
        qa_text = self.qa_segmenter.format_qa_pairs(qa_dict)
        
        logger.info(f"  [3/4] ✅ Found {len(qa_dict['questions'])} Q, {len(qa_dict['answers'])} A")
        
        # Étape 3: Notation par LLM
        logger.info(f"  [3/4] Grading with LLM...")
        grade_result = self.grader.grade(qa_text, subject_name)
        
        logger.info(f"  [3/4] ✅ Grade: {grade_result.get('grade_letter', '?')} ({grade_result.get('score', 0)}/20)")
        
        # Étape 4: Sauvegarde
        logger.info(f"  [4/4] Saving results...")
        output_dir = self.output_root / subject_name / student_id
        output_dir.mkdir(parents=True, exist_ok=True)
        
        self._save_student_results(output_dir, ocr_result, qa_dict, grade_result)
        logger.info(f"  [4/4] ✅ Saved to {output_dir}")
        
        return {
            "status": "success",
            "student_id": student_id,
            "subject": subject_name,
            "grade": grade_result.get("grade_letter"),
            "score": grade_result.get("score"),
            "num_questions": len(qa_dict['questions']),
            "num_answers": len(qa_dict['answers']),
        }
    
    def _get_image_files(self, directory: Path) -> List[str]:
        """Récupère tous les fichiers image du dossier"""
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif'}
        images = []
        
        for ext in image_extensions:
            images.extend(sorted(directory.glob(f"*{ext}")))
            images.extend(sorted(directory.glob(f"*{ext.upper()}")))
        
        return [str(img) for img in sorted(set(images))]
    
    def _save_student_results(self, output_dir: Path, ocr_result, 
                             qa_dict: Dict, grade_result: Dict):
        """Sauvegarde les résultats d'un étudiant"""
        
        # 1. Texte OCR brut
        with open(output_dir / "01_ocr_raw.txt", "w", encoding="utf-8") as f:
            f.write(ocr_result.full_text)
        
        # 2. Q/A formatées
        with open(output_dir / "02_questions_answers.txt", "w", encoding="utf-8") as f:
            f.write(self.qa_segmenter.format_qa_pairs(qa_dict))
        
        # 3. Notation
        with open(output_dir / "03_grade.json", "w", encoding="utf-8") as f:
            json.dump(grade_result, f, ensure_ascii=False, indent=2)
        
        # 4. Rapport lisible
        report = self._format_report(ocr_result, qa_dict, grade_result)
        with open(output_dir / "04_report.txt", "w", encoding="utf-8") as f:
            f.write(report)
    
    def _format_report(self, ocr_result, qa_dict: Dict, grade_result: Dict) -> str:
        """Formate un rapport lisible pour l'étudiant"""
        lines = [
            "=" * 80,
            "EXAM GRADE REPORT",
            "=" * 80,
            "",
            f"SCORE: {grade_result.get('score', '?')}/20",
            f"GRADE: {grade_result.get('grade', '?')}",
            "",
            "STRENGTHS:",
            *[f"  • {s}" for s in grade_result.get('strengths', [])],
            "",
            "WEAKNESSES:",
            *[f"  • {w}" for w in grade_result.get('weaknesses', [])],
            "",
            "FEEDBACK:",
            f"{grade_result.get('feedback', 'N/A')}",
            "",
            "TO IMPROVE:",
            *[f"  {i+1}. {imp}" for i, imp in enumerate(grade_result.get('improvements', []))],
            "",
            "=" * 80,
        ]
        return '\n'.join(lines)
    
    def _save_summary_report(self, all_results: Dict):
        """Sauvegarde un résumé global"""
        summary = {
            "timestamp": datetime.now().isoformat(),
            "total_students": sum(len(v) for v in all_results.values()),
            "subjects": list(all_results.keys()),
            "by_subject": {}
        }
        
        # Calcule les statistiques par matière
        for subject, results in all_results.items():
            successful = [r for r in results.values() if r.get("status") == "success"]
            scores = [r.get("score", 0) for r in successful]
            
            summary["by_subject"][subject] = {
                "total": len(results),
                "successful": len(successful),
                "failed": len(results) - len(successful),
                "average_score": sum(scores) / len(scores) if scores else 0,
                "students": results
            }
        
        with open(self.output_root / "summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        
        # Rapport texte
        report_lines = [
            "=" * 80,
            "BATCH PROCESSING SUMMARY",
            "=" * 80,
            f"Timestamp: {summary['timestamp']}",
            f"Total Students: {summary['total_students']}",
            "",
        ]
        
        for subject, stats in summary["by_subject"].items():
            report_lines.extend([
                f"\n{subject.upper()}:",
                f"  Processed: {stats['successful']}/{stats['total']}",
                f"  Failed: {stats['failed']}",
                f"  Average Score: {stats['average_score']:.1f}/100",
            ])
        
        with open(self.output_root / "summary.txt", "w", encoding="utf-8") as f:
            f.write('\n'.join(report_lines))


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Batch process exams with OCR and grading using open-source LLM")
    parser.add_argument("--data", default="data", help="Path to data folder (default: data/)")
    parser.add_argument("--ocr-model", default="french", choices=["french", "english", "french_printed"],
                       help="OCR model to use")
    parser.add_argument("--api-key", default=None, 
                       help="LLM API key (or set LLM_API_KEY environment variable)")
    parser.add_argument("--api-endpoint", default=None,
                       help="LLM API endpoint (default: Hugging Face)")
    parser.add_argument("--model-name", default=None,
                       help="LLM model name (default: mistralai/Mistral-7B-Instruct-v0.1)")
    parser.add_argument("--skip-ocr", action="store_true", help="Skip OCR if already processed")
    
    args = parser.parse_args()
    
    # Vérifie que le dossier data existe
    if not Path(args.data).exists():
        logger.error(f"Data folder not found: {args.data}")
        sys.exit(1)
    
    # Lance le traitement
    processor = BatchExamProcessor(
        data_root=args.data,
        ocr_model=args.ocr_model,
        api_key=args.api_key,
        api_endpoint=args.api_endpoint,
        model_name=args.model_name,
        skip_ocr=args.skip_ocr
    )
    processor.process_all()


if __name__ == "__main__":
    main()
