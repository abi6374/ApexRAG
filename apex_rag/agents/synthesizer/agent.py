from apex_rag.core.evidence.models import EvidencePacket
from apex_rag.providers import AsyncLLM


class EvidenceSynthesizerAgent:
    """Agent responsible for synthesizing answers from verified evidence packets."""

    def __init__(self, llm: AsyncLLM):
        self.llm = llm

    async def synthesize(self, query: str, packets: list[EvidencePacket]) -> str:
        """
        Synthesize an answer to the query using ONLY the provided evidence packets.
        Must cite the packets using their node_id.
        """
        if not packets:
            return "I could not find enough evidence to answer your query."

        # Filter out unverified packets if there are any
        verified_packets = [p for p in packets if p.verification_result]

        if not verified_packets:
            return "No verified evidence was provided."

        context_blocks = []
        for packet in verified_packets:
            context_blocks.append(f"[Node ID: {packet.node_id}]\n{packet.content}")

        context_str = "\n\n".join(context_blocks)

        prompt = f"""You are an expert synthesizer. Your task is to answer the user's query ONLY using the provided evidence.

EVIDENCE:
{context_str}

USER QUERY:
{query}

INSTRUCTIONS:
1. Answer the query based strictly on the provided evidence.
2. If the evidence does not contain the answer, say "I cannot answer this based on the provided evidence."
3. You MUST cite the source of your information by including the relevant Node ID inline, formatted as [Node ID: <node_id>].
"""
        response = await self.llm.generate(prompt, max_tokens=1000)
        return response.strip()
