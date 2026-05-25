class ConfidenceEngine:
    def __init__(self, verifier_weight: float = 0.6, retrieval_weight: float = 0.4, depth_penalty: float = 0.05):
        self.verifier_weight = verifier_weight
        self.retrieval_weight = retrieval_weight
        self.depth_penalty = depth_penalty

    def calculate_score(self, retrieval_confidence: float, verifier_score: float, graph_depth: int) -> float:
        """
        Calculate a confidence score based on retrieval confidence, verifier score, and graph depth.
        """
        base_score = (verifier_score * self.verifier_weight) + (retrieval_confidence * self.retrieval_weight)
        penalty = graph_depth * self.depth_penalty
        score = base_score - penalty
        return max(0.0, min(1.0, score))
