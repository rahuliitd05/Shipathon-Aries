"""
Advanced Fact-Checking Analyzer using NLI + Semantic Similarity Ensemble
"""
import torch
import numpy as np
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import re


class Verdict(Enum):
    TRUE = "TRUE"
    MOSTLY_TRUE = "MOSTLY TRUE"
    MIXED = "MIXED"
    MOSTLY_FALSE = "MOSTLY FALSE"
    FALSE = "FALSE"
    UNVERIFIABLE = "UNVERIFIABLE"


@dataclass
class VerificationResult:
    verdict: Verdict
    confidence: float
    nli_score: float
    semantic_score: float
    supporting_evidence: List[Dict]
    contradicting_evidence: List[Dict]
    explanation: str


class FactCheckAnalyzer:
    """
    Ensemble fact-checker combining:
    1. NLI (Natural Language Inference) for entailment/contradiction detection
    2. Semantic similarity for evidence matching
    3. Multi-evidence aggregation for robust verdicts
    """
    
    def __init__(self):
        print("🔄 Loading NLI model (facebook/bart-large-mnli)...")
        self.nli_pipeline = pipeline(
            "zero-shot-classification",
            model="facebook/bart-large-mnli",
            device=0 if torch.cuda.is_available() else -1
        )
        
        print("🔄 Loading semantic similarity model...")
        self.semantic_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # For more nuanced NLI analysis
        print("🔄 Loading entailment model (roberta-large-mnli)...")
        self.nli_tokenizer = AutoTokenizer.from_pretrained("roberta-large-mnli")
        self.nli_classifier = AutoModelForSequenceClassification.from_pretrained("roberta-large-mnli")
        if torch.cuda.is_available():
            self.nli_classifier = self.nli_classifier.cuda()
        self.nli_classifier.eval()
        
        print("✅ All models loaded successfully!")
    
    def _compute_nli_scores(self, claim: str, evidence: str) -> Dict[str, float]:
        """
        Compute NLI scores: entailment, neutral, contradiction
        """
        inputs = self.nli_tokenizer(
            evidence, claim,
            return_tensors="pt",
            truncation=True,
            max_length=512
        )
        
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.nli_classifier(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)[0].cpu().numpy()
        
        # RoBERTa-large-MNLI order: contradiction, neutral, entailment
        return {
            "contradiction": float(probs[0]),
            "neutral": float(probs[1]),
            "entailment": float(probs[2])
        }
    
    def _compute_semantic_similarity(self, claim: str, evidence_texts: List[str]) -> np.ndarray:
        """
        Compute cosine similarity between claim and all evidence pieces
        """
        if not evidence_texts:
            return np.array([])
        
        claim_embedding = self.semantic_model.encode([claim])
        evidence_embeddings = self.semantic_model.encode(evidence_texts)
        similarities = cosine_similarity(claim_embedding, evidence_embeddings)[0]
        return similarities
    
    def _decompose_claim(self, claim: str) -> List[str]:
        """
        Break complex claims into simpler verifiable sub-claims
        """
        sub_claims = []
        
        # Split on "and", "also", commas for compound claims
        parts = re.split(r'\s+and\s+|\s+also\s+|,\s*(?=\w+\s+(?:is|was|were|are|has|had|have))', claim, flags=re.IGNORECASE)
        
        for part in parts:
            part = part.strip()
            if len(part) > 10:
                sub_claims.append(part)
        
        return sub_claims if len(sub_claims) > 1 else [claim]
    
    def _analyze_single_claim(self, claim: str, evidence_list: List[Dict]) -> Dict:
        """
        Analyze a single claim against evidence
        """
        if not evidence_list:
            return {
                "claim": claim,
                "nli_scores": {"entailment": 0, "neutral": 1, "contradiction": 0},
                "semantic_score": 0,
                "best_evidence": None,
                "supporting": [],
                "contradicting": []
            }
        
        evidence_texts = [e['text'] for e in evidence_list]
        
        # 1. Semantic similarity to find relevant evidence
        similarities = self._compute_semantic_similarity(claim, evidence_texts)
        
        # Get top-k most relevant evidence pieces
        top_k = min(5, len(evidence_list))
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        # 2. NLI analysis on top relevant evidence
        nli_results = []
        supporting = []
        contradicting = []
        
        for idx in top_indices:
            if similarities[idx] < 0.3:
                continue
                
            evidence = evidence_list[idx]
            nli_scores = self._compute_nli_scores(claim, evidence['text'])
            
            result = {
                "evidence": evidence,
                "semantic_score": float(similarities[idx]),
                "nli_scores": nli_scores
            }
            nli_results.append(result)
            
            if nli_scores['entailment'] > 0.5:
                supporting.append(result)
            elif nli_scores['contradiction'] > 0.5:
                contradicting.append(result)
        
        # 3. Aggregate scores
        if nli_results:
            avg_entailment = np.mean([r['nli_scores']['entailment'] for r in nli_results])
            avg_contradiction = np.mean([r['nli_scores']['contradiction'] for r in nli_results])
            max_semantic = max([r['semantic_score'] for r in nli_results])
            best_evidence = nli_results[0]['evidence']
        else:
            avg_entailment = 0
            avg_contradiction = 0
            max_semantic = 0
            best_evidence = evidence_list[0] if evidence_list else None
        
        return {
            "claim": claim,
            "nli_scores": {
                "entailment": avg_entailment,
                "contradiction": avg_contradiction,
                "neutral": 1 - avg_entailment - avg_contradiction
            },
            "semantic_score": max_semantic,
            "best_evidence": best_evidence,
            "supporting": supporting,
            "contradicting": contradicting,
            "all_results": nli_results
        }
    
    def _determine_verdict(self, entailment: float, contradiction: float, 
                           semantic: float, evidence_count: int) -> Tuple[Verdict, float]:
        """
        Determine final verdict based on aggregated scores
        """
        evidence_confidence = min(1.0, evidence_count / 3)
        
        support_score = (entailment * 0.7 + semantic * 0.3) * evidence_confidence
        oppose_score = contradiction * evidence_confidence
        
        if evidence_count == 0 or semantic < 0.3:
            return Verdict.UNVERIFIABLE, 0.2
        
        net_score = support_score - oppose_score
        
        if net_score > 0.5:
            verdict = Verdict.TRUE
            confidence = min(0.95, 0.6 + net_score * 0.4)
        elif net_score > 0.25:
            verdict = Verdict.MOSTLY_TRUE
            confidence = 0.5 + net_score * 0.3
        elif net_score > -0.25:
            verdict = Verdict.MIXED
            confidence = 0.4 + abs(net_score) * 0.2
        elif net_score > -0.5:
            verdict = Verdict.MOSTLY_FALSE
            confidence = 0.5 + abs(net_score) * 0.3
        else:
            verdict = Verdict.FALSE
            confidence = min(0.95, 0.6 + abs(net_score) * 0.4)
        
        return verdict, confidence
    
    def verify(self, claim: str, evidence_list: List[Dict]) -> VerificationResult:
        """
        Main verification method - analyzes claim against gathered evidence
        """
        if not evidence_list:
            return VerificationResult(
                verdict=Verdict.UNVERIFIABLE,
                confidence=0.1,
                nli_score=0,
                semantic_score=0,
                supporting_evidence=[],
                contradicting_evidence=[],
                explanation="No evidence found to verify this claim."
            )
        
        sub_claims = self._decompose_claim(claim)
        
        all_supporting = []
        all_contradicting = []
        total_entailment = 0
        total_contradiction = 0
        max_semantic = 0
        
        for sub_claim in sub_claims:
            result = self._analyze_single_claim(sub_claim, evidence_list)
            
            total_entailment += result['nli_scores']['entailment']
            total_contradiction += result['nli_scores']['contradiction']
            max_semantic = max(max_semantic, result['semantic_score'])
            
            all_supporting.extend(result['supporting'])
            all_contradicting.extend(result['contradicting'])
        
        n_claims = len(sub_claims)
        avg_entailment = total_entailment / n_claims
        avg_contradiction = total_contradiction / n_claims
        
        verdict, confidence = self._determine_verdict(
            avg_entailment, avg_contradiction, max_semantic, len(evidence_list)
        )
        
        explanation = self._build_explanation(
            verdict, confidence, avg_entailment, avg_contradiction,
            len(all_supporting), len(all_contradicting)
        )
        
        return VerificationResult(
            verdict=verdict,
            confidence=confidence,
            nli_score=avg_entailment - avg_contradiction,
            semantic_score=max_semantic,
            supporting_evidence=all_supporting[:3],
            contradicting_evidence=all_contradicting[:3],
            explanation=explanation
        )
    
    def _build_explanation(self, verdict: Verdict, confidence: float,
                           entailment: float, contradiction: float,
                           n_supporting: int, n_contradicting: int) -> str:
        """
        Generate human-readable explanation
        """
        verdict_text = verdict.value
        
        if verdict == Verdict.TRUE:
            base = f"The claim appears to be **{verdict_text}** with {confidence*100:.0f}% confidence."
            detail = f"Found {n_supporting} supporting evidence pieces with high entailment scores."
        elif verdict == Verdict.MOSTLY_TRUE:
            base = f"The claim is **{verdict_text}** with {confidence*100:.0f}% confidence."
            detail = f"Evidence largely supports the claim, though some details may vary."
        elif verdict == Verdict.MIXED:
            base = f"The claim has **{verdict_text}** evidence with {confidence*100:.0f}% confidence."
            detail = f"Found both supporting ({n_supporting}) and contradicting ({n_contradicting}) evidence."
        elif verdict == Verdict.MOSTLY_FALSE:
            base = f"The claim is **{verdict_text}** with {confidence*100:.0f}% confidence."
            detail = f"Evidence suggests key parts of the claim are inaccurate."
        elif verdict == Verdict.FALSE:
            base = f"The claim appears to be **{verdict_text}** with {confidence*100:.0f}% confidence."
            detail = f"Found {n_contradicting} contradicting evidence pieces."
        else:
            base = f"The claim is **{verdict_text}**."
            detail = "Could not find sufficient evidence to verify or refute this claim."
        
        return f"{base} {detail}"


# Global analyzer instance (lazy loaded)
_analyzer: Optional[FactCheckAnalyzer] = None


def get_analyzer() -> FactCheckAnalyzer:
    """Get or create the global analyzer instance"""
    global _analyzer
    if _analyzer is None:
        _analyzer = FactCheckAnalyzer()
    return _analyzer


def check_veracity(user_claim: str, evidence_list: List[Dict]) -> Tuple[float, Dict, str]:
    """
    Legacy API compatibility wrapper
    Returns: (score, best_evidence, label)
    """
    analyzer = get_analyzer()
    result = analyzer.verify(user_claim, evidence_list)
    
    score = (result.nli_score + 1) / 2
    
    if result.supporting_evidence:
        best_evidence = result.supporting_evidence[0]['evidence']
    elif evidence_list:
        best_evidence = evidence_list[0]
    else:
        best_evidence = {"text": "No evidence found", "source": "N/A", "url": ""}
    
    label = result.verdict.value
    
    return score, best_evidence, label


def verify_claim(claim: str, evidence_list: List[Dict]) -> VerificationResult:
    """
    Full verification with detailed results
    """
    analyzer = get_analyzer()
    return analyzer.verify(claim, evidence_list)